import ctypes
import ctypes.util
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.ops import (
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
)
from flag_blas.utils import shape_utils

TPSV_SIZES = [
    64,
    96,
    127,
    128,
    129,
    192,
    255,
    256,
    257,
    384,
    511,
    512,
    513,
    768,
    1023,
    1024,
    1025,
    1536,
    2048,
    3072,
    4096,
]


def _load_cublas():
    names = ["libcublas.so.13"]
    found = ctypes.util.find_library("cublas")
    if found:
        names.append(found)
    names.extend(["libcublas.so", "libcublas.so.12", "libcublas.so.11"])
    for name in names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError("Unable to find libcublas.so")


_cublas = _load_cublas()
_cublas_handle = None
_CUBLAS_TPSV_FUNCS = {
    torch.float32: _cublas.cublasStpsv_v2,
    torch.float64: _cublas.cublasDtpsv_v2,
    torch.complex64: _cublas.cublasCtpsv_v2,
    torch.complex128: _cublas.cublasZtpsv_v2,
}


def _get_cublas_handle():
    global _cublas_handle
    if _cublas_handle is None:
        handle = ctypes.c_void_p()
        status = _cublas.cublasCreate_v2(ctypes.byref(handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status code: {status}")
        _cublas_handle = handle.value
    status = _cublas.cublasSetStream_v2(
        ctypes.c_void_p(_cublas_handle),
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )
    if status != 0:
        raise RuntimeError(f"cublasSetStream_v2 failed with status code: {status}")
    return _cublas_handle


def cublas_tpsv_baseline(AP, x, uplo, trans, diag, n, incx, handle, c_func, **kwargs):
    if n == 0:
        return x
    status = c_func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(trans),
        ctypes.c_int(diag),
        ctypes.c_int(n),
        ctypes.c_void_p(AP.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
    )
    if status != 0:
        raise RuntimeError(f"cublasXtpsv_v2 failed with status code: {status}")
    return x


def _gems_wrapper(op):
    def _impl(AP, x, uplo, trans, diag, n, incx, **kwargs):
        op(uplo, trans, diag, n, AP, x, incx)
        return x

    return _impl


GEMS_TPSV_WRAPPERS = {
    "stpsv": _gems_wrapper(flag_blas.stpsv),
    "dtpsv": _gems_wrapper(flag_blas.dtpsv),
    "ctpsv": _gems_wrapper(flag_blas.ctpsv),
    "ztpsv": _gems_wrapper(flag_blas.ztpsv),
}


def _pack_triangular(A, uplo):
    n = A.shape[0]
    values = []
    for col in range(n):
        rows = range(col + 1) if uplo == CUBLAS_FILL_MODE_UPPER else range(col, n)
        values.extend(A[row, col] for row in rows)
    return torch.stack(values).contiguous()


def _make_case(n, dtype, uplo, diag, device):
    A = torch.randn((n, n), dtype=dtype, device=device) * 0.02
    A = torch.triu(A) if uplo == CUBLAS_FILL_MODE_UPPER else torch.tril(A)
    if diag == CUBLAS_DIAG_NON_UNIT:
        idx = torch.arange(n, device=device)
        A[idx, idx] += (2.0 + 0.25j) if dtype.is_complex else 2.0
    x = torch.randn(n, dtype=dtype, device=device)
    return _pack_triangular(A, uplo), x.contiguous()


class TpsvBenchmark(Benchmark):
    DEFAULT_SHAPES = [(n,) for n in TPSV_SIZES]
    DEFAULT_SHAPE_DESC = "N"

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
        self.shapes = [(n,) for n in TPSV_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = _get_cublas_handle()
        c_func = _CUBLAS_TPSV_FUNCS[cur_dtype]
        for shape in self.shapes:
            n = shape[0]
            AP, x = _make_case(n, cur_dtype, self.uplo, self.diag, self.device)
            yield AP, x, {
                "uplo": self.uplo,
                "trans": self.trans,
                "diag": self.diag,
                "n": n,
                "incx": 1,
                "handle": handle,
                "c_func": c_func,
            }

    def get_correctness_reduce_dim(self, args, kwargs):
        return max(1, kwargs["n"])

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        return n * (n + 1)

    def get_gbps(self, args, latency):
        AP, x = args[0], args[1]
        io_amount = shape_utils.size_in_bytes(AP) + 2 * shape_utils.size_in_bytes(x)
        return io_amount * 1e-9 / (latency * 1e-3)


def _run_tpsv(op_name, dtype, uplo, trans, diag=CUBLAS_DIAG_NON_UNIT):
    if (
        dtype in (torch.float64, torch.complex128)
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("fp64 is not supported on this device")
    bench = TpsvBenchmark(
        op_name=f"{op_name}_{uplo}_{trans}_{diag}",
        torch_op=cublas_tpsv_baseline,
        gems_op=GEMS_TPSV_WRAPPERS[op_name],
        dtypes=[dtype],
        uplo=uplo,
        trans=trans,
        diag=diag,
    )
    run_correctness_then_benchmark(bench)


TPSV_PERF_CASES = [
    pytest.param(
        "stpsv",
        torch.float32,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        marks=pytest.mark.stpsv,
    ),
    pytest.param(
        "stpsv",
        torch.float32,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        marks=pytest.mark.stpsv,
    ),
    pytest.param(
        "stpsv",
        torch.float32,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        marks=pytest.mark.stpsv,
    ),
    pytest.param(
        "stpsv",
        torch.float32,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        marks=pytest.mark.stpsv,
    ),
    pytest.param(
        "dtpsv",
        torch.float64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        marks=pytest.mark.dtpsv,
    ),
    pytest.param(
        "dtpsv",
        torch.float64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        marks=pytest.mark.dtpsv,
    ),
    pytest.param(
        "dtpsv",
        torch.float64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        marks=pytest.mark.dtpsv,
    ),
    pytest.param(
        "dtpsv",
        torch.float64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        marks=pytest.mark.dtpsv,
    ),
    pytest.param(
        "ctpsv",
        torch.complex64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        marks=pytest.mark.ctpsv,
    ),
    pytest.param(
        "ctpsv",
        torch.complex64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        marks=pytest.mark.ctpsv,
    ),
    pytest.param(
        "ctpsv",
        torch.complex64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        marks=pytest.mark.ctpsv,
    ),
    pytest.param(
        "ctpsv",
        torch.complex64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        marks=pytest.mark.ctpsv,
    ),
    pytest.param(
        "ctpsv",
        torch.complex64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_C,
        marks=pytest.mark.ctpsv,
    ),
    pytest.param(
        "ctpsv",
        torch.complex64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_C,
        marks=pytest.mark.ctpsv,
    ),
    pytest.param(
        "ztpsv",
        torch.complex128,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        marks=pytest.mark.ztpsv,
    ),
    pytest.param(
        "ztpsv",
        torch.complex128,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        marks=pytest.mark.ztpsv,
    ),
    pytest.param(
        "ztpsv",
        torch.complex128,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        marks=pytest.mark.ztpsv,
    ),
    pytest.param(
        "ztpsv",
        torch.complex128,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        marks=pytest.mark.ztpsv,
    ),
    pytest.param(
        "ztpsv",
        torch.complex128,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_C,
        marks=pytest.mark.ztpsv,
    ),
    pytest.param(
        "ztpsv",
        torch.complex128,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_C,
        marks=pytest.mark.ztpsv,
    ),
]


@pytest.mark.parametrize("op_name,dtype,uplo,trans", TPSV_PERF_CASES)
def test_perf_tpsv(op_name, dtype, uplo, trans):
    _run_tpsv(op_name, dtype, uplo, trans)
