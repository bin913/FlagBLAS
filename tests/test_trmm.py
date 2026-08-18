import ctypes
import ctypes.util

import cupy as cp
import numpy as np
import pytest
import torch
from cupy_backends.cuda.libs import cublas
from scipy.linalg import blas as cpu_blas

import flag_blas
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

from .accuracy_utils import blas_assert_close
from .conftest import TO_CPU


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

TRMM_CASES = [
    (0, 5, 0, 0, 0),
    (1, 1, 0, 0, 0),
    (8, 7, 1, 2, 3),
    (31, 33, 0, 1, 0),
    (32, 32, 0, 0, 0),
    (64, 32, 2, 0, 1),
    (127, 65, 1, 1, 2),
    (128, 128, 0, 0, 0),
    (257, 129, 1, 0, 1),
    (512, 256, 0, 2, 0),
    (1023, 513, 1, 0, 1),
    (2048, 1024, 0, 0, 0),
    (4096, 2048, 0, 0, 0),
]

REAL_CASES = TRMM_CASES + [
    (8192, 4096, 0, 0, 0),
    (9999, 4096, 0, 1, 0),
    (12288, 4096, 0, 0, 0),
    (16384, 4096, 1, 0, 1),
]

COMPLEX_CASES = TRMM_CASES + [
    (8191, 4095, 0, 1, 0),
    (8192, 4096, 0, 0, 0),
]
SIDES = [CUBLAS_SIDE_LEFT, CUBLAS_SIDE_RIGHT]
UPLOS = [CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER]
DIAGS = [CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT]
REAL_TRANS = [CUBLAS_OP_N, CUBLAS_OP_T]
COMPLEX_TRANS = [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


def make_triangular(k, lda, uplo, diag, dtype, device):
    if k == 0:
        return torch.empty((max(k, 1), lda), dtype=dtype, device=device).contiguous()
    A = torch.randn((k, lda), dtype=dtype, device=device)
    rows = torch.arange(k, device=device).view(k, 1)
    cols = torch.arange(lda, device=device).view(1, lda)
    if uplo == CUBLAS_FILL_MODE_UPPER:
        valid = (cols >= rows) & (cols < k)
    else:
        valid = cols <= rows
    if diag == CUBLAS_DIAG_UNIT:
        valid = valid & (cols != rows)
    if dtype.is_complex:
        torch.view_as_real(A).masked_fill_(~valid.unsqueeze(-1), float("nan"))
    else:
        A.masked_fill_(~valid, float("nan"))
    return A.contiguous()


def make_dense(rows, cols, ld, dtype, device):
    X = torch.randn((max(rows, 1), ld), dtype=dtype, device=device)
    if ld > cols:
        if dtype.is_complex:
            torch.view_as_real(X[:, cols:]).fill_(float("nan"))
        else:
            X[:, cols:].fill_(float("nan"))
    return X.contiguous()


def alpha_value(dtype):
    if dtype == torch.complex64:
        return 0.75 + 0.25j
    if dtype == torch.complex128:
        return 0.75 + 0.125j
    return 1.25


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


def compact_col_major(X, rows, cols):
    if rows == 0 or cols == 0:
        return torch.empty((max(cols, 1), max(rows, 1)), dtype=X.dtype, device=X.device)
    return X[:rows, :cols].t().contiguous()


def cublas_trmm_reference(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    if m == 0 or n == 0:
        return torch.empty((max(m, 1), max(n, 1)), dtype=B.dtype, device=B.device)
    C_col = torch.empty((max(n, 1), max(m, 1)), dtype=B.dtype, device=B.device)
    k = m if side == CUBLAS_SIDE_LEFT else n
    A_col = compact_col_major(A, k, k)
    B_col = compact_col_major(B, m, n)
    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    func = _CUBLAS_TRMM_FUNCS[B.dtype]
    alpha_obj, alpha_ptr = cublas_alpha(alpha, B.dtype)
    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(side),
        ctypes.c_int(uplo),
        ctypes.c_int(trans),
        ctypes.c_int(diag),
        ctypes.c_int(m),
        ctypes.c_int(n),
        alpha_ptr,
        ctypes.c_void_p(A_col.data_ptr()),
        ctypes.c_int(max(1, k)),
        ctypes.c_void_p(B_col.data_ptr()),
        ctypes.c_int(max(1, m)),
        ctypes.c_void_p(C_col.data_ptr()),
        ctypes.c_int(max(1, m)),
    )
    if status != 0:
        raise RuntimeError(f"cublasXtrmm_v2 failed with status code: {status}")
    cp.cuda.runtime.deviceSynchronize()
    return C_col.t().contiguous()


def _to_cpu_fortran_blas_array(inp, torch_dtype, np_dtype):
    cpu_tensor = inp.detach().to(device="cpu", dtype=torch_dtype)
    return np.array(cpu_tensor.numpy(), dtype=np_dtype, order="F", copy=True)


def cpu_trmm_reference(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    torch_dtype = torch.complex128 if B.dtype.is_complex else torch.float64
    np_dtype = np.complex128 if B.dtype.is_complex else np.float64
    if m == 0 or n == 0:
        return torch.empty((max(m, 1), max(n, 1)), dtype=torch_dtype)
    k = m if side == CUBLAS_SIDE_LEFT else n
    a_np = _to_cpu_fortran_blas_array(A[:k, :k], torch_dtype, np_dtype)
    b_np = _to_cpu_fortran_blas_array(B[:m, :n], torch_dtype, np_dtype)
    func = cpu_blas.ztrmm if B.dtype.is_complex else cpu_blas.dtrmm
    out = func(
        alpha,
        a_np,
        b_np,
        side=side,
        lower=(uplo == CUBLAS_FILL_MODE_LOWER),
        trans_a=trans,
        diag=(diag == CUBLAS_DIAG_UNIT),
        overwrite_b=1,
    )
    return torch.from_numpy(out)


def trmm_reference(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    if TO_CPU:
        return cpu_trmm_reference(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    return cublas_trmm_reference(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


def run_case(op, dtype, side, uplo, trans, diag, case):
    m, n, lda_extra, ldb_extra, ldc_extra = case
    if dtype in (torch.float64, torch.complex128):
        check_fp64_support()
    device = flag_blas.device
    k = m if side == CUBLAS_SIDE_LEFT else n
    lda = max(1, k + lda_extra)
    ldb = max(1, n + ldb_extra)
    ldc = max(1, n + ldc_extra)
    A = make_triangular(k, lda, uplo, diag, dtype, device)
    B = make_dense(m, n, ldb, dtype, device)
    C = torch.empty((max(m, 1), ldc), dtype=dtype, device=device)
    if ldc > n:
        if dtype.is_complex:
            torch.view_as_real(C[:, n:]).fill_(float("nan"))
        else:
            C[:, n:].fill_(float("nan"))
    alpha = alpha_value(dtype)
    ref = trmm_reference(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    op(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    if m == 0 or n == 0:
        return
    blas_assert_close(C[:m, :n], ref[:m, :n], dtype, reduce_dim=max(1, k))


@pytest.mark.strmm
@pytest.mark.parametrize("case", REAL_CASES)
@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
def test_accuracy_strmm(case, side, uplo, trans, diag):
    run_case(flag_blas.strmm, torch.float32, side, uplo, trans, diag, case)


@pytest.mark.dtrmm
@pytest.mark.parametrize("case", REAL_CASES)
@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
def test_accuracy_dtrmm(case, side, uplo, trans, diag):
    run_case(flag_blas.dtrmm, torch.float64, side, uplo, trans, diag, case)


@pytest.mark.ctrmm
@pytest.mark.parametrize("case", COMPLEX_CASES)
@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
def test_accuracy_ctrmm(case, side, uplo, trans, diag):
    run_case(flag_blas.ctrmm, torch.complex64, side, uplo, trans, diag, case)


@pytest.mark.ztrmm
@pytest.mark.parametrize("case", COMPLEX_CASES)
@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
def test_accuracy_ztrmm(case, side, uplo, trans, diag):
    run_case(flag_blas.ztrmm, torch.complex128, side, uplo, trans, diag, case)


@pytest.mark.strmm
def test_strmm_alpha_zero():
    m, n = 17, 19
    dtype = torch.float32
    A = make_triangular(
        m, m, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, dtype, flag_blas.device
    )
    B = make_dense(m, n, n, dtype, flag_blas.device)
    C = torch.full((m, n), float("nan"), dtype=dtype, device=flag_blas.device)
    flag_blas.strmm(
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
        m,
        n,
        0.0,
        A,
        m,
        B,
        n,
        C,
        n,
    )
    ref = torch.zeros((m, n), dtype=dtype, device="cpu" if TO_CPU else flag_blas.device)
    blas_assert_close(C, ref, dtype, reduce_dim=1)


def _run_alpha_zero_padded(op, dtype):
    if dtype in (torch.float64, torch.complex128):
        check_fp64_support()
    m, n, ldc_extra = 17, 19, 5
    device = flag_blas.device
    A = make_triangular(
        m, m, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, dtype, device
    )
    B = make_dense(m, n, n, dtype, device)
    ldc = n + ldc_extra
    sentinel = (3.5 + 1.5j) if dtype.is_complex else 3.5
    C = torch.full((m, ldc), sentinel, dtype=dtype, device=device)
    alpha = 0j if dtype.is_complex else 0.0
    op(
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
        m,
        n,
        alpha,
        A,
        m,
        B,
        n,
        C,
        ldc,
    )
    assert torch.all(C[:, :n] == 0)
    assert torch.all(C[:, n:] == sentinel)


@pytest.mark.strmm
def test_strmm_alpha_zero_padded():
    _run_alpha_zero_padded(flag_blas.strmm, torch.float32)


@pytest.mark.dtrmm
def test_dtrmm_alpha_zero_padded():
    _run_alpha_zero_padded(flag_blas.dtrmm, torch.float64)


@pytest.mark.ctrmm
def test_ctrmm_alpha_zero_padded():
    _run_alpha_zero_padded(flag_blas.ctrmm, torch.complex64)


@pytest.mark.ztrmm
def test_ztrmm_alpha_zero_padded():
    _run_alpha_zero_padded(flag_blas.ztrmm, torch.complex128)


@pytest.mark.ctrmm
def test_ctrmm_inplace_alias():
    m, n = 17, 19
    dtype = torch.complex64
    A = make_triangular(
        m, m, CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_UNIT, dtype, flag_blas.device
    )
    B = make_dense(m, n, n, dtype, flag_blas.device)
    B_ref = B.clone()
    alpha = 0.75 + 0.25j
    ref = trmm_reference(
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_C,
        CUBLAS_DIAG_UNIT,
        m,
        n,
        alpha,
        A,
        m,
        B_ref,
        n,
    )
    flag_blas.ctrmm(
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_C,
        CUBLAS_DIAG_UNIT,
        m,
        n,
        alpha,
        A,
        m,
        B,
        n,
        B,
        n,
    )
    blas_assert_close(B, ref, dtype, reduce_dim=m)


@pytest.mark.ztrmm
def test_ztrmm_inplace_alias():
    m, n = 17, 19
    dtype = torch.complex128
    A = make_triangular(
        m, m, CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_UNIT, dtype, flag_blas.device
    )
    B = make_dense(m, n, n, dtype, flag_blas.device)
    B_ref = B.clone()
    alpha = 0.75 + 0.25j
    ref = trmm_reference(
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_C,
        CUBLAS_DIAG_UNIT,
        m,
        n,
        alpha,
        A,
        m,
        B_ref,
        n,
    )
    flag_blas.ztrmm(
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_C,
        CUBLAS_DIAG_UNIT,
        m,
        n,
        alpha,
        A,
        m,
        B,
        n,
        B,
        n,
    )
    blas_assert_close(B, ref, dtype, reduce_dim=m)
