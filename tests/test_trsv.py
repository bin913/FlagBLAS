import ctypes
import ctypes.util

import cupy as cp
import pytest
import torch
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
)

from .accuracy_utils import blas_assert_close, to_cpu_blas_tensor
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


def cublas_trsv_reference(uplo, trans, diag, n, A, lda, x, incx):
    if n == 0:
        return
    handle = cp.cuda.device.get_cublas_handle()
    dtype = A.dtype
    if dtype == torch.float32:
        func = _cublas.cublasStrsv_v2
    elif dtype == torch.float64:
        func = _cublas.cublasDtrsv_v2
    elif dtype == torch.complex64:
        func = _cublas.cublasCtrsv_v2
    elif dtype == torch.complex128:
        func = _cublas.cublasZtrsv_v2
    else:
        raise ValueError(f"Unsupported dtype {dtype}")
    status = func(
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
        raise RuntimeError(f"cublasXtrsv_v2 execution failed with error code: {status}")
    cp.cuda.runtime.deviceSynchronize()


def cpu_trsv_reference(uplo, trans, diag, n, A, lda, x, incx):
    ref_x = to_cpu_blas_tensor(x)
    if n == 0:
        return ref_x

    ref_A = to_cpu_blas_tensor(A)
    func = cpu_blas.ztrsv if ref_A.dtype.is_complex else cpu_blas.dtrsv
    xout = func(
        ref_A[:n, :n].T.numpy(),
        ref_x.numpy(),
        incx=incx,
        lower=int(uplo == CUBLAS_FILL_MODE_LOWER),
        trans=trans,
        diag=diag,
        overwrite_x=1,
    )
    return torch.from_numpy(xout)


def trsv_reference(uplo, trans, diag, n, A, lda, x, incx):
    if TO_CPU:
        return cpu_trsv_reference(uplo, trans, diag, n, A, lda, x, incx)

    ref_x = x.clone()
    cublas_trsv_reference(uplo, trans, diag, n, A, lda, ref_x, incx)
    return ref_x


TRSV_CASES = [
    (0, 0),
    (1, 0),
    (64, 0),
    (256, 0),
    (512, 0),
    (1024, 0),
    (2048, 0),
    (4096, 0),
    (8192, 0),
    (255, 7),
    (1024, 16),
]
TRSV_STRIDE_CASES = [(64, 0), (1023, 0)]
INCS = [1, 2, 3]
UPLOS = [CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER]
DIAGS = [CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT]
REAL_TRANS = [CUBLAS_OP_N, CUBLAS_OP_T]
COMPLEX_TRANS = [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]


def make_triangular(n, lda, uplo, diag, dtype, device):
    if n == 0:
        return torch.zeros((n, lda), dtype=dtype, device=device).contiguous()
    A = torch.empty((n, lda), dtype=dtype, device=device)
    if dtype.is_complex:
        torch.view_as_real(A).fill_(float("nan"))
    else:
        A.fill_(float("nan"))
    vals = torch.randn((n, n), dtype=dtype, device=device) * 0.02
    row_idx = torch.arange(n, device=device).view(1, n)
    col_idx = torch.arange(n, device=device).view(n, 1)
    if uplo == CUBLAS_FILL_MODE_UPPER:
        valid = row_idx <= col_idx
    else:
        valid = row_idx >= col_idx
    if diag == CUBLAS_DIAG_UNIT:
        valid = valid & (row_idx != col_idx)
    A[:, :n] = vals.masked_fill(~valid, float("nan"))
    if diag == CUBLAS_DIAG_NON_UNIT:
        diag_vals = torch.diagonal(vals).clone()
        if dtype.is_complex:
            diag_vals = diag_vals + (2.0 + 0.25j)
        else:
            diag_vals = diag_vals + 2.0
        idx = torch.arange(n, device=device)
        A[idx, idx] = diag_vals
    return A.contiguous()


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


@pytest.mark.strsv
@pytest.mark.parametrize("n, lda_extra", TRSV_CASES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
def test_accuracy_strsv(n, uplo, trans, diag, lda_extra):
    dtype = torch.float32
    lda = max(1, n + lda_extra)
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(max(n, 1), dtype=dtype, device=flag_blas.device)
    ref_x = trsv_reference(uplo, trans, diag, n, A, lda, x, 1)
    flag_blas.strsv(uplo, trans, diag, n, A, lda, x, 1)
    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.strsv
@pytest.mark.parametrize("n, lda_extra", TRSV_STRIDE_CASES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
def test_accuracy_strsv_stride(n, uplo, trans, diag, incx, lda_extra):
    dtype = torch.float32
    lda = n + lda_extra
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = trsv_reference(uplo, trans, diag, n, A, lda, x, incx)
    flag_blas.strsv(uplo, trans, diag, n, A, lda, x, incx)
    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.dtrsv
@pytest.mark.parametrize("n, lda_extra", TRSV_CASES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
def test_accuracy_dtrsv(n, uplo, trans, diag, lda_extra):
    check_fp64_support()
    dtype = torch.float64
    lda = max(1, n + lda_extra)
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(max(n, 1), dtype=dtype, device=flag_blas.device)
    ref_x = trsv_reference(uplo, trans, diag, n, A, lda, x, 1)
    flag_blas.dtrsv(uplo, trans, diag, n, A, lda, x, 1)
    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.dtrsv
@pytest.mark.parametrize("n, lda_extra", TRSV_STRIDE_CASES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
def test_accuracy_dtrsv_stride(n, uplo, trans, diag, incx, lda_extra):
    check_fp64_support()
    dtype = torch.float64
    lda = n + lda_extra
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = trsv_reference(uplo, trans, diag, n, A, lda, x, incx)
    flag_blas.dtrsv(uplo, trans, diag, n, A, lda, x, incx)
    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.ctrsv
@pytest.mark.parametrize("n, lda_extra", TRSV_CASES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
def test_accuracy_ctrsv(n, uplo, trans, diag, lda_extra):
    dtype = torch.complex64
    lda = max(1, n + lda_extra)
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(max(n, 1), dtype=dtype, device=flag_blas.device)
    ref_x = trsv_reference(uplo, trans, diag, n, A, lda, x, 1)
    flag_blas.ctrsv(uplo, trans, diag, n, A, lda, x, 1)
    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.ctrsv
@pytest.mark.parametrize("n, lda_extra", TRSV_STRIDE_CASES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
def test_accuracy_ctrsv_stride(n, uplo, trans, diag, incx, lda_extra):
    dtype = torch.complex64
    lda = n + lda_extra
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = trsv_reference(uplo, trans, diag, n, A, lda, x, incx)
    flag_blas.ctrsv(uplo, trans, diag, n, A, lda, x, incx)
    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.ztrsv
@pytest.mark.parametrize("n, lda_extra", TRSV_CASES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
def test_accuracy_ztrsv(n, uplo, trans, diag, lda_extra):
    check_fp64_support()
    dtype = torch.complex128
    lda = max(1, n + lda_extra)
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(max(n, 1), dtype=dtype, device=flag_blas.device)
    ref_x = trsv_reference(uplo, trans, diag, n, A, lda, x, 1)
    flag_blas.ztrsv(uplo, trans, diag, n, A, lda, x, 1)
    blas_assert_close(x, ref_x, dtype, reduce_dim=n)


@pytest.mark.ztrsv
@pytest.mark.parametrize("n, lda_extra", TRSV_STRIDE_CASES)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
def test_accuracy_ztrsv_stride(n, uplo, trans, diag, incx, lda_extra):
    check_fp64_support()
    dtype = torch.complex128
    lda = n + lda_extra
    A = make_triangular(n, lda, uplo, diag, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = trsv_reference(uplo, trans, diag, n, A, lda, x, incx)
    flag_blas.ztrsv(uplo, trans, diag, n, A, lda, x, incx)
    blas_assert_close(x, ref_x, dtype, reduce_dim=n)
