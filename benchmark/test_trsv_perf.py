import ctypes
import ctypes.util
from typing import Generator

import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.ops import (
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_DIAG_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
)
from flag_blas.utils import shape_utils

TRSV_SIZES = [64, 256, 512, 1024, 2048, 4096, 8192]


def load_cublas():
    lib_names = ["libcublas.so", "libcublas.so.12", "libcublas.so.11"]
    found_path = ctypes.util.find_library("cublas")
    if found_path:
        lib_names.insert(0, found_path)
    for name in lib_names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError("Unable to find libcublas.so on this system")


_cublas = load_cublas()

_CUBLAS_TRSV_FUNCS = {
    torch.float32: _cublas.cublasStrsv_v2,
    torch.float64: _cublas.cublasDtrsv_v2,
    torch.complex64: _cublas.cublasCtrsv_v2,
    torch.complex128: _cublas.cublasZtrsv_v2,
}


def cublas_trsv_baseline(
    A,
    x,
    uplo,
    trans,
    diag,
    n,
    lda,
    incx,
    handle,
    c_func,
    **kwargs,
):
    if n == 0:
        return x
    status = c_func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(trans),
        ctypes.c_int(diag),
        ctypes.c_int(n),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
    )
    if status != 0:
        raise RuntimeError(f"cublasXtrsv_v2 failed with status code: {status}")
    return x


def _gems_wrapper(op):
    def _impl(A, x, uplo, trans, diag, n, lda, incx, handle, **kwargs):
        op(uplo, trans, diag, n, A, lda, x, incx)
        return x

    return _impl


gems_strsv_wrapper = _gems_wrapper(flag_blas.strsv)
gems_dtrsv_wrapper = _gems_wrapper(flag_blas.dtrsv)
gems_ctrsv_wrapper = _gems_wrapper(flag_blas.ctrsv)
gems_ztrsv_wrapper = _gems_wrapper(flag_blas.ztrsv)


def _run_trsv_benchmark(
    op_name,
    wrapper,
    dtype,
    uplo,
    trans,
    diag,
    require_fp64=False,
):
    if require_fp64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = TrsvBenchmark(
        op_name=op_name,
        torch_op=cublas_trsv_baseline,
        gems_op=wrapper,
        dtypes=[dtype],
        uplo=uplo,
        trans=trans,
        diag=diag,
    )
    run_correctness_then_benchmark(bench)


def _generate_triangular_A(n, lda, uplo, diag, dtype, device):
    vals = torch.randn(n, n, dtype=dtype, device=device) * 0.02
    A = torch.zeros((n, lda), dtype=dtype, device=device)
    row_idx = torch.arange(n, device=device).view(1, n)
    col_idx = torch.arange(n, device=device).view(n, 1)
    if uplo == CUBLAS_FILL_MODE_UPPER:
        valid = row_idx <= col_idx
    else:
        valid = row_idx >= col_idx
    A[:, :n] = vals.masked_fill(~valid, 0.0)
    if diag == CUBLAS_DIAG_NON_UNIT:
        diag_vals = torch.diagonal(vals).clone()
        if dtype.is_complex:
            diag_vals = diag_vals + (2.0 + 0.25j)
        else:
            diag_vals = diag_vals + 2.0
        idx = torch.arange(n, device=device)
        A[idx, idx] = diag_vals
    return A.contiguous()


class TrsvBenchmark(Benchmark):
    def __init__(
        self,
        *args,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.trans = trans
        self.diag = diag

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in TRSV_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
        if cur_dtype not in _CUBLAS_TRSV_FUNCS:
            raise ValueError(f"Unsupported dtype: {cur_dtype}")
        c_func = _CUBLAS_TRSV_FUNCS[cur_dtype]
        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            lda = n
            A = _generate_triangular_A(
                n, lda, self.uplo, self.diag, cur_dtype, self.device
            )
            x = torch.randn(n, dtype=cur_dtype, device=self.device)
            yield A, x.clone(), {
                "uplo": self.uplo,
                "trans": self.trans,
                "diag": self.diag,
                "n": n,
                "lda": lda,
                "incx": 1,
                "handle": handle,
                "c_func": c_func,
            }

    def get_correctness_reduce_dim(self, args, kwargs):
        return max(1, kwargs["n"])

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        nnz = n * (n + 1) // 2
        A = args[0]
        if A.dtype in (torch.complex64, torch.complex128):
            return 4 * nnz
        return nnz

    def get_gbps(self, args, latency):
        A, x = args[0], args[1]
        n = x.numel()
        a_bytes = n * (n + 1) // 2 * A.element_size()
        io_amount = a_bytes + 2 * shape_utils.size_in_bytes(x)
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.strsv
def test_perf_strsv():
    _run_trsv_benchmark(
        op_name="strsv",
        wrapper=gems_strsv_wrapper,
        dtype=torch.float32,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.strsv
def test_perf_strsv_upper():
    _run_trsv_benchmark(
        op_name="strsv_upper",
        wrapper=gems_strsv_wrapper,
        dtype=torch.float32,
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.strsv
def test_perf_strsv_trans():
    _run_trsv_benchmark(
        op_name="strsv_trans",
        wrapper=gems_strsv_wrapper,
        dtype=torch.float32,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.strsv
def test_perf_strsv_unit():
    _run_trsv_benchmark(
        op_name="strsv_unit",
        wrapper=gems_strsv_wrapper,
        dtype=torch.float32,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )


@pytest.mark.dtrsv
def test_perf_dtrsv():
    _run_trsv_benchmark(
        op_name="dtrsv",
        wrapper=gems_dtrsv_wrapper,
        dtype=torch.float64,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
        require_fp64=True,
    )


@pytest.mark.dtrsv
def test_perf_dtrsv_upper():
    _run_trsv_benchmark(
        op_name="dtrsv_upper",
        wrapper=gems_dtrsv_wrapper,
        dtype=torch.float64,
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
        require_fp64=True,
    )


@pytest.mark.dtrsv
def test_perf_dtrsv_trans():
    _run_trsv_benchmark(
        op_name="dtrsv_trans",
        wrapper=gems_dtrsv_wrapper,
        dtype=torch.float64,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
        require_fp64=True,
    )


@pytest.mark.dtrsv
def test_perf_dtrsv_unit():
    _run_trsv_benchmark(
        op_name="dtrsv_unit",
        wrapper=gems_dtrsv_wrapper,
        dtype=torch.float64,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
        require_fp64=True,
    )


@pytest.mark.ctrsv
def test_perf_ctrsv():
    _run_trsv_benchmark(
        op_name="ctrsv",
        wrapper=gems_ctrsv_wrapper,
        dtype=torch.complex64,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.ctrsv
def test_perf_ctrsv_upper():
    _run_trsv_benchmark(
        op_name="ctrsv_upper",
        wrapper=gems_ctrsv_wrapper,
        dtype=torch.complex64,
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.ctrsv
def test_perf_ctrsv_trans():
    _run_trsv_benchmark(
        op_name="ctrsv_trans",
        wrapper=gems_ctrsv_wrapper,
        dtype=torch.complex64,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.ctrsv
def test_perf_ctrsv_conj():
    _run_trsv_benchmark(
        op_name="ctrsv_conj",
        wrapper=gems_ctrsv_wrapper,
        dtype=torch.complex64,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_C,
        diag=CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.ctrsv
def test_perf_ctrsv_unit():
    _run_trsv_benchmark(
        op_name="ctrsv_unit",
        wrapper=gems_ctrsv_wrapper,
        dtype=torch.complex64,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
    )


@pytest.mark.ztrsv
def test_perf_ztrsv():
    _run_trsv_benchmark(
        op_name="ztrsv",
        wrapper=gems_ztrsv_wrapper,
        dtype=torch.complex128,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
        require_fp64=True,
    )


@pytest.mark.ztrsv
def test_perf_ztrsv_upper():
    _run_trsv_benchmark(
        op_name="ztrsv_upper",
        wrapper=gems_ztrsv_wrapper,
        dtype=torch.complex128,
        uplo=CUBLAS_FILL_MODE_UPPER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
        require_fp64=True,
    )


@pytest.mark.ztrsv
def test_perf_ztrsv_trans():
    _run_trsv_benchmark(
        op_name="ztrsv_trans",
        wrapper=gems_ztrsv_wrapper,
        dtype=torch.complex128,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_T,
        diag=CUBLAS_DIAG_NON_UNIT,
        require_fp64=True,
    )


@pytest.mark.ztrsv
def test_perf_ztrsv_conj():
    _run_trsv_benchmark(
        op_name="ztrsv_conj",
        wrapper=gems_ztrsv_wrapper,
        dtype=torch.complex128,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_C,
        diag=CUBLAS_DIAG_NON_UNIT,
        require_fp64=True,
    )


@pytest.mark.ztrsv
def test_perf_ztrsv_unit():
    _run_trsv_benchmark(
        op_name="ztrsv_unit",
        wrapper=gems_ztrsv_wrapper,
        dtype=torch.complex128,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_UNIT,
        require_fp64=True,
    )
