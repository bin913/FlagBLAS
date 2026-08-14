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
    CUBLAS_SIDE_LEFT,
    CUBLAS_SIDE_RIGHT,
)
from flag_blas.utils import shape_utils

REAL_TRMM_SHAPES = [
    (32, 32),
    (128, 128),
    (512, 256),
    (1024, 1024),
    (2048, 1024),
    (4096, 2048),
    (8192, 4096),
    (9999, 4096),
    (12288, 4096),
    (16384, 4096),
]

COMPLEX_TRMM_SHAPES = [
    (32, 32),
    (128, 128),
    (512, 256),
    (1024, 1024),
    (2048, 1024),
    (4096, 2048),
    (8191, 4095),
    (8192, 4096),
]


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

_CUBLAS_TRMM_FUNCS = {
    torch.float32: _cublas.cublasStrmm_v2,
    torch.float64: _cublas.cublasDtrmm_v2,
    torch.complex64: _cublas.cublasCtrmm_v2,
    torch.complex128: _cublas.cublasZtrmm_v2,
}


def cublas_alpha(alpha, dtype):
    if dtype == torch.float32:
        val = ctypes.c_float(float(alpha))
        return val, ctypes.byref(val)
    if dtype == torch.float64:
        val = ctypes.c_double(float(alpha))
        return val, ctypes.byref(val)
    if dtype == torch.complex64:
        val = (ctypes.c_float * 2)(float(alpha.real), float(alpha.imag))
        return val, ctypes.byref(val)
    if dtype == torch.complex128:
        val = (ctypes.c_double * 2)(float(alpha.real), float(alpha.imag))
        return val, ctypes.byref(val)
    raise ValueError(f"Unsupported dtype {dtype}")


def alpha_value(dtype):
    if dtype == torch.complex64:
        return 0.75 + 0.25j
    if dtype == torch.complex128:
        return 0.75 + 0.125j
    return 1.25


def make_triangular(k, uplo, diag, dtype, device):
    A = torch.randn((max(k, 1), max(k, 1)), dtype=dtype, device=device)
    if k == 0:
        return A.contiguous()
    rows = torch.arange(k, device=device).view(k, 1)
    cols = torch.arange(k, device=device).view(1, k)
    if uplo == CUBLAS_FILL_MODE_UPPER:
        valid = cols >= rows
    else:
        valid = cols <= rows
    if diag == CUBLAS_DIAG_UNIT:
        valid = valid & (cols != rows)
    if dtype.is_complex:
        torch.view_as_real(A[:k, :k]).masked_fill_(~valid.unsqueeze(-1), float("nan"))
    else:
        A[:k, :k].masked_fill_(~valid, float("nan"))
    return A.contiguous()


def compact_col_major(X, rows, cols):
    if rows == 0 or cols == 0:
        return torch.empty((max(cols, 1), max(rows, 1)), dtype=X.dtype, device=X.device)
    return X[:rows, :cols].t().contiguous()


def cublas_trmm_baseline(
    A_col,
    B_col,
    C_col,
    A_row,
    B_row,
    C_row,
    side,
    uplo,
    trans,
    diag,
    m,
    n,
    alpha,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    lda_flag,
    ldb_flag,
    ldc_flag,
    handle,
    c_func,
    alpha_ptr,
    alpha_obj,
):
    if m == 0 or n == 0:
        return C_col
    status = c_func(
        ctypes.c_void_p(handle),
        ctypes.c_int(side),
        ctypes.c_int(uplo),
        ctypes.c_int(trans),
        ctypes.c_int(diag),
        ctypes.c_int(m),
        ctypes.c_int(n),
        alpha_ptr,
        ctypes.c_void_p(A_col.data_ptr()),
        ctypes.c_int(lda_cublas),
        ctypes.c_void_p(B_col.data_ptr()),
        ctypes.c_int(ldb_cublas),
        ctypes.c_void_p(C_col.data_ptr()),
        ctypes.c_int(ldc_cublas),
    )
    if status != 0:
        raise RuntimeError(f"cublasXtrmm_v2 failed with status code: {status}")
    return C_col


def _gems_wrapper(op):
    def _impl(
        A_col,
        B_col,
        C_col,
        A_row,
        B_row,
        C_row,
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        alpha,
        lda_cublas,
        ldb_cublas,
        ldc_cublas,
        lda_flag,
        ldb_flag,
        ldc_flag,
        handle,
        c_func,
        alpha_ptr,
        alpha_obj,
    ):
        op(
            side,
            uplo,
            trans,
            diag,
            m,
            n,
            alpha,
            A_row,
            lda_flag,
            B_row,
            ldb_flag,
            C_row,
            ldc_flag,
        )
        return C_row

    return _impl


gems_strmm_wrapper = _gems_wrapper(flag_blas.strmm)
gems_dtrmm_wrapper = _gems_wrapper(flag_blas.dtrmm)
gems_ctrmm_wrapper = _gems_wrapper(flag_blas.ctrmm)
gems_ztrmm_wrapper = _gems_wrapper(flag_blas.ztrmm)


class TrmmBenchmark(Benchmark):
    def __init__(
        self,
        *args,
        side=CUBLAS_SIDE_LEFT,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.side = side
        self.uplo = uplo
        self.trans = trans
        self.diag = diag
        self.correctness_reference = "cuBLAS"

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        cur_dtype = self.dtypes[0]
        if cur_dtype in (torch.complex64, torch.complex128):
            self.shapes = COMPLEX_TRMM_SHAPES
        else:
            self.shapes = REAL_TRMM_SHAPES
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
        c_func = _CUBLAS_TRMM_FUNCS[cur_dtype]
        alpha = alpha_value(cur_dtype)
        alpha_obj, alpha_ptr = cublas_alpha(alpha, cur_dtype)
        for m, n in self.shapes:
            k = m if self.side == CUBLAS_SIDE_LEFT else n
            A_row = make_triangular(k, self.uplo, self.diag, cur_dtype, self.device)
            B_row = torch.randn(
                (max(m, 1), max(n, 1)), dtype=cur_dtype, device=self.device
            )
            C_row = torch.empty_like(B_row)
            A_col = compact_col_major(A_row, k, k)
            B_col = compact_col_major(B_row, m, n)
            C_col = torch.empty_like(B_col)
            yield A_col, B_col, C_col, A_row, B_row, C_row, {
                "side": self.side,
                "uplo": self.uplo,
                "trans": self.trans,
                "diag": self.diag,
                "m": m,
                "n": n,
                "alpha": alpha,
                "lda_cublas": max(1, k),
                "ldb_cublas": max(1, m),
                "ldc_cublas": max(1, m),
                "lda_flag": max(1, k),
                "ldb_flag": max(1, n),
                "ldc_flag": max(1, n),
                "handle": handle,
                "c_func": c_func,
                "alpha_ptr": alpha_ptr,
                "alpha_obj": alpha_obj,
            }

    def get_tflops(self, op, *args, **kwargs):
        m = kwargs.get("m", 0)
        n = kwargs.get("n", 0)
        k = m if kwargs.get("side") == CUBLAS_SIDE_LEFT else n
        ops = m * n * (k + 1)
        A_row = args[3]
        if A_row.dtype in (torch.complex64, torch.complex128):
            return 4 * ops
        return ops

    def get_gbps(self, args, latency):
        A_row, B_row, C_row = args[3], args[4], args[5]
        k = A_row.shape[0]
        a_bytes = k * (k + 1) // 2 * A_row.element_size()
        io_amount = (
            a_bytes
            + shape_utils.size_in_bytes(B_row)
            + shape_utils.size_in_bytes(C_row)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return max(
            1, kwargs["m"] if kwargs["side"] == CUBLAS_SIDE_LEFT else kwargs["n"]
        )

    def clone_correctness_inputs(self, args, kwargs):
        A_col, B_col, C_col, A_row, B_row, C_row = args
        ref_args = (A_col, B_col, C_col.clone(), A_row, B_row, C_row)
        blas_args = (A_col, B_col, C_col, A_row, B_row, C_row.clone())
        return ref_args, kwargs, blas_args, kwargs

    def validate_results(self, reference_result, blas_result, dtype, reduce_dim=1):
        ref = reference_result.t().contiguous()
        return super().validate_results(ref, blas_result, dtype, reduce_dim=reduce_dim)


def _run_trmm_benchmark(op_name, torch_op, gems_op, dtype, side, uplo, trans, diag):
    bench = TrmmBenchmark(
        op_name=op_name,
        torch_op=torch_op,
        gems_op=gems_op,
        dtypes=[dtype],
        side=side,
        uplo=uplo,
        trans=trans,
        diag=diag,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.strmm
def test_perf_strmm_left_lower():
    _run_trmm_benchmark(
        "strmm_left_lower",
        cublas_trmm_baseline,
        gems_strmm_wrapper,
        torch.float32,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.strmm
def test_perf_strmm_right_upper():
    _run_trmm_benchmark(
        "strmm_right_upper",
        cublas_trmm_baseline,
        gems_strmm_wrapper,
        torch.float32,
        CUBLAS_SIDE_RIGHT,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.strmm
def test_perf_strmm_trans_unit():
    _run_trmm_benchmark(
        "strmm_trans_unit",
        cublas_trmm_baseline,
        gems_strmm_wrapper,
        torch.float32,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        CUBLAS_DIAG_UNIT,
    )


@pytest.mark.dtrmm
def test_perf_dtrmm_left_lower():
    _run_trmm_benchmark(
        "dtrmm_left_lower",
        cublas_trmm_baseline,
        gems_dtrmm_wrapper,
        torch.float64,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.dtrmm
def test_perf_dtrmm_right_upper():
    _run_trmm_benchmark(
        "dtrmm_right_upper",
        cublas_trmm_baseline,
        gems_dtrmm_wrapper,
        torch.float64,
        CUBLAS_SIDE_RIGHT,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.dtrmm
def test_perf_dtrmm_trans_unit():
    _run_trmm_benchmark(
        "dtrmm_trans_unit",
        cublas_trmm_baseline,
        gems_dtrmm_wrapper,
        torch.float64,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        CUBLAS_DIAG_UNIT,
    )


@pytest.mark.ctrmm
def test_perf_ctrmm_left_lower():
    _run_trmm_benchmark(
        "ctrmm_left_lower",
        cublas_trmm_baseline,
        gems_ctrmm_wrapper,
        torch.complex64,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.ctrmm
def test_perf_ctrmm_right_upper():
    _run_trmm_benchmark(
        "ctrmm_right_upper",
        cublas_trmm_baseline,
        gems_ctrmm_wrapper,
        torch.complex64,
        CUBLAS_SIDE_RIGHT,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.ctrmm
def test_perf_ctrmm_trans_unit():
    _run_trmm_benchmark(
        "ctrmm_trans_unit",
        cublas_trmm_baseline,
        gems_ctrmm_wrapper,
        torch.complex64,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        CUBLAS_DIAG_UNIT,
    )


@pytest.mark.ctrmm
def test_perf_ctrmm_conj_unit():
    _run_trmm_benchmark(
        "ctrmm_conj_unit",
        cublas_trmm_baseline,
        gems_ctrmm_wrapper,
        torch.complex64,
        CUBLAS_SIDE_RIGHT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_C,
        CUBLAS_DIAG_UNIT,
    )


@pytest.mark.ztrmm
def test_perf_ztrmm_left_lower():
    _run_trmm_benchmark(
        "ztrmm_left_lower",
        cublas_trmm_baseline,
        gems_ztrmm_wrapper,
        torch.complex128,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.ztrmm
def test_perf_ztrmm_right_upper():
    _run_trmm_benchmark(
        "ztrmm_right_upper",
        cublas_trmm_baseline,
        gems_ztrmm_wrapper,
        torch.complex128,
        CUBLAS_SIDE_RIGHT,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
    )


@pytest.mark.ztrmm
def test_perf_ztrmm_trans_unit():
    _run_trmm_benchmark(
        "ztrmm_trans_unit",
        cublas_trmm_baseline,
        gems_ztrmm_wrapper,
        torch.complex128,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        CUBLAS_DIAG_UNIT,
    )


@pytest.mark.ztrmm
def test_perf_ztrmm_conj_unit():
    _run_trmm_benchmark(
        "ztrmm_conj_unit",
        cublas_trmm_baseline,
        gems_ztrmm_wrapper,
        torch.complex128,
        CUBLAS_SIDE_RIGHT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_C,
        CUBLAS_DIAG_UNIT,
    )
