import ctypes
import ctypes.util

import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas

if flag_blas.vendor_name == "hygon":
    from .hipblas_reference import check_hipblas_status, get_hipblas_context
elif flag_blas.vendor_name != "ascend":
    import cupy as cp
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


_cublas = None if flag_blas.vendor_name in {"ascend", "hygon"} else load_cublas()


def hipblas_tbmv_reference(uplo, trans, diag, n, k, A, lda, x, incx):
    if n == 0:
        return x

    if A.dtype == torch.float32:
        symbol = "hipblasStbmv"
    elif A.dtype == torch.float64:
        symbol = "hipblasDtbmv"
    elif A.dtype == torch.complex64:
        symbol = "hipblasCtbmv_v2"
    elif A.dtype == torch.complex128:
        symbol = "hipblasZtbmv_v2"
    else:
        raise ValueError(f"Unsupported dtype for hipBLAS TBMV: {A.dtype}")

    hip_uplo = 121 if uplo == CUBLAS_FILL_MODE_UPPER else 122
    hip_trans = 111 + trans
    hip_diag = 131 + diag
    library, handle = get_hipblas_context(A)
    function = getattr(library, symbol)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    check_hipblas_status(
        function(
            handle,
            hip_uplo,
            hip_trans,
            hip_diag,
            n,
            k,
            ctypes.c_void_p(A.data_ptr()),
            lda,
            ctypes.c_void_p(x.data_ptr()),
            incx,
        ),
        symbol,
    )
    return x


def cublas_tbmv_reference(uplo, trans, diag, n, k, A, lda, x, incx):
    if n == 0:
        return
    handle = cp.cuda.device.get_cublas_handle()
    dtype = A.dtype
    if dtype == torch.float32:
        func = _cublas.cublasStbmv_v2
    elif dtype == torch.float64:
        func = _cublas.cublasDtbmv_v2
    elif dtype == torch.complex64:
        func = _cublas.cublasCtbmv_v2
    elif dtype == torch.complex128:
        func = _cublas.cublasZtbmv_v2
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    status = func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(trans),
        ctypes.c_int(diag),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
    )
    if status != 0:
        raise RuntimeError(f"cublasXtbmv_v2 execution failed with error code: {status}")


def cpu_tbmv_reference(uplo, trans, diag, n, k, A, lda, x, incx):
    ref_x = to_cpu_blas_tensor(x)
    if n == 0:
        return ref_x

    ref_A = to_cpu_blas_tensor(A)
    func = cpu_blas.ztbmv if ref_A.dtype.is_complex else cpu_blas.dtbmv
    xout = func(
        k,
        ref_A.numpy().T,
        ref_x.numpy(),
        incx=incx,
        lower=int(uplo == CUBLAS_FILL_MODE_LOWER),
        trans=trans,
        diag=diag,
        overwrite_x=1,
    )
    return torch.from_numpy(xout)


def npu_tbmv_dense_matrix(uplo, diag, n, k, A):
    if A.dtype.is_complex:
        dense_values = torch.zeros((n, n, 2), dtype=torch.float32, device=A.device)
    else:
        dense_values = torch.zeros((n, n), dtype=A.dtype, device=A.device)
    if n == 0:
        return (
            torch.view_as_complex(dense_values) if A.dtype.is_complex else dense_values
        )

    columns = torch.arange(n, device=A.device).view(n, 1)
    band_rows = torch.arange(k + 1, device=A.device).view(1, k + 1)
    if uplo == CUBLAS_FILL_MODE_UPPER:
        rows = columns + band_rows - k
        diagonal_band_row = k
    else:
        rows = columns + band_rows
        diagonal_band_row = 0

    valid = (rows >= 0) & (rows < n)
    if diag == CUBLAS_DIAG_UNIT:
        valid &= band_rows != diagonal_band_row

    column_indices = columns.expand(-1, k + 1)[valid]
    row_indices = rows[valid]
    if A.dtype.is_complex:
        band_values = torch.view_as_real(A)[:, : k + 1]
        dense_values[row_indices, column_indices, 0] = band_values[..., 0][valid]
        dense_values[row_indices, column_indices, 1] = band_values[..., 1][valid]
        dense = torch.view_as_complex(dense_values)
    else:
        dense_values[row_indices, column_indices] = A[:, : k + 1][valid]
        dense = dense_values
    if diag == CUBLAS_DIAG_UNIT:
        dense.diagonal().fill_(1)
    return dense


def npu_tbmv_reference(uplo, trans, diag, n, k, A, lda, x, incx):
    ref_x = x.clone()
    if n == 0:
        return ref_x

    dense = npu_tbmv_dense_matrix(uplo, diag, n, k, A)
    logical_x = x[::incx][:n].clone()
    if A.dtype.is_complex:
        if trans == CUBLAS_OP_N:
            real_matrix = dense.real
            imag_matrix = dense.imag
        elif trans == CUBLAS_OP_T:
            real_matrix = dense.real.T
            imag_matrix = dense.imag.T
        else:
            real_matrix = dense.real.T
            imag_matrix = -dense.imag.T

        real_x = logical_x.real
        imag_x = logical_x.imag
        real_out = torch.mv(real_matrix, real_x) - torch.mv(imag_matrix, imag_x)
        imag_out = torch.mv(real_matrix, imag_x) + torch.mv(imag_matrix, real_x)
        result = torch.view_as_complex(torch.stack((real_out, imag_out), dim=-1))
    else:
        matrix = dense if trans == CUBLAS_OP_N else dense.T
        result = torch.mv(matrix, logical_x)

    ref_x[::incx][:n].copy_(result)
    return ref_x


def tbmv_reference(uplo, trans, diag, n, k, A, lda, x, incx):
    if TO_CPU:
        return cpu_tbmv_reference(uplo, trans, diag, n, k, A, lda, x, incx)
    if flag_blas.vendor_name == "ascend":
        return npu_tbmv_reference(uplo, trans, diag, n, k, A, lda, x, incx)

    ref_x = x.clone()
    if flag_blas.vendor_name == "hygon":
        hipblas_tbmv_reference(uplo, trans, diag, n, k, A, lda, ref_x, incx)
    else:
        cublas_tbmv_reference(uplo, trans, diag, n, k, A, lda, ref_x, incx)
    return ref_x


TBMV_SIZES = [
    0,
    1,
    31,
    256,
    4096,
    16384,
]
TBMV_STRIDE_SIZES = [64, 256]
TBMV_KS = [0, 1, 16, 256]
INCS = [1, 2]
LDA_EXTRAS = [0, 2]
LDA_EXTRAS_STRIDE = [0, 1]


def tbmv_randn(shape, dtype, device):
    if flag_blas.vendor_name == "ascend" and dtype == torch.complex64:
        values = torch.randn((*shape, 2), dtype=torch.float32, device=device)
        return torch.view_as_complex(values)
    return torch.randn(shape, dtype=dtype, device=device)


def make_triangular_banded(n, k, lda, uplo, diag, dtype, device):
    if n == 0:
        return torch.zeros((n, lda), dtype=dtype, device=device).contiguous()

    A = tbmv_randn((n, lda), dtype, device)
    cols = torch.arange(lda, device=device).view(1, lda)
    j = torch.arange(n, device=device).view(n, 1)
    unit = diag == CUBLAS_DIAG_UNIT

    if uplo == CUBLAS_FILL_MODE_UPPER:
        valid = (cols >= torch.clamp(k - j, min=0)) & (cols <= k)
        if unit:
            valid &= cols != k
    else:
        valid = cols <= torch.clamp(n - 1 - j, max=k)
        if unit:
            valid &= cols != 0

    if dtype.is_complex:
        torch.view_as_real(A).masked_fill_(~valid.unsqueeze(-1), float("nan"))
    else:
        A.masked_fill_(~valid, float("nan"))
    return A.contiguous()


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


def tbmv_assert_close(result, reference, dtype, reduce_dim):
    if flag_blas.vendor_name == "ascend" and dtype.is_complex:
        result = result.cpu()
        reference = reference.cpu()
    blas_assert_close(result, reference, dtype, reduce_dim=reduce_dim)


UPLOS = [CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER]
DIAGS = [CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT]
REAL_TRANS = [CUBLAS_OP_N, CUBLAS_OP_T]
COMPLEX_TRANS = [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]


def _effective_k(n, k):
    return min(k, max(0, n - 1))


@pytest.mark.stbmv
@pytest.mark.parametrize("n", TBMV_SIZES)
@pytest.mark.parametrize("k", TBMV_KS)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS)
def test_accuracy_stbmv(n, k, uplo, trans, diag, lda_extra):
    k = _effective_k(n, k)
    dtype = torch.float32
    lda = k + 1 + lda_extra
    A = make_triangular_banded(n, k, lda, uplo, diag, dtype, flag_blas.device)
    x = tbmv_randn((max(n, 1),), dtype, flag_blas.device)
    ref_x = tbmv_reference(uplo, trans, diag, n, k, A, lda, x, 1)
    flag_blas.stbmv(uplo, trans, diag, n, k, A, lda, x, 1)

    tbmv_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.stbmv
@pytest.mark.parametrize("n", TBMV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS_STRIDE)
def test_accuracy_stbmv_stride(n, k, uplo, trans, diag, incx, lda_extra):
    k = _effective_k(n, k)
    dtype = torch.float32
    lda = k + 1 + lda_extra
    A = make_triangular_banded(n, k, lda, uplo, diag, dtype, flag_blas.device)
    x = tbmv_randn((1 + (n - 1) * incx,), dtype, flag_blas.device)
    ref_x = tbmv_reference(uplo, trans, diag, n, k, A, lda, x, incx)
    flag_blas.stbmv(uplo, trans, diag, n, k, A, lda, x, incx)

    tbmv_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.dtbmv
@pytest.mark.parametrize("n", TBMV_SIZES)
@pytest.mark.parametrize("k", TBMV_KS)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS)
def test_accuracy_dtbmv(n, k, uplo, trans, diag, lda_extra):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype = torch.float64
    lda = k + 1 + lda_extra
    A = make_triangular_banded(n, k, lda, uplo, diag, dtype, flag_blas.device)
    x = tbmv_randn((max(n, 1),), dtype, flag_blas.device)
    ref_x = tbmv_reference(uplo, trans, diag, n, k, A, lda, x, 1)
    flag_blas.dtbmv(uplo, trans, diag, n, k, A, lda, x, 1)

    tbmv_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.dtbmv
@pytest.mark.parametrize("n", TBMV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", REAL_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS_STRIDE)
def test_accuracy_dtbmv_stride(n, k, uplo, trans, diag, incx, lda_extra):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype = torch.float64
    lda = k + 1 + lda_extra
    A = make_triangular_banded(n, k, lda, uplo, diag, dtype, flag_blas.device)
    x = tbmv_randn((1 + (n - 1) * incx,), dtype, flag_blas.device)
    ref_x = tbmv_reference(uplo, trans, diag, n, k, A, lda, x, incx)
    flag_blas.dtbmv(uplo, trans, diag, n, k, A, lda, x, incx)

    tbmv_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.ctbmv
@pytest.mark.parametrize("n", TBMV_SIZES)
@pytest.mark.parametrize("k", TBMV_KS)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS)
def test_accuracy_ctbmv(n, k, uplo, trans, diag, lda_extra):
    k = _effective_k(n, k)
    dtype = torch.complex64
    lda = k + 1 + lda_extra
    A = make_triangular_banded(n, k, lda, uplo, diag, dtype, flag_blas.device)
    x = tbmv_randn((max(n, 1),), dtype, flag_blas.device)
    ref_x = tbmv_reference(uplo, trans, diag, n, k, A, lda, x, 1)
    flag_blas.ctbmv(uplo, trans, diag, n, k, A, lda, x, 1)

    tbmv_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.ctbmv
@pytest.mark.parametrize("n", TBMV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS_STRIDE)
def test_accuracy_ctbmv_stride(n, k, uplo, trans, diag, incx, lda_extra):
    k = _effective_k(n, k)
    dtype = torch.complex64
    lda = k + 1 + lda_extra
    A = make_triangular_banded(n, k, lda, uplo, diag, dtype, flag_blas.device)
    x = tbmv_randn((1 + (n - 1) * incx,), dtype, flag_blas.device)
    ref_x = tbmv_reference(uplo, trans, diag, n, k, A, lda, x, incx)
    flag_blas.ctbmv(uplo, trans, diag, n, k, A, lda, x, incx)

    tbmv_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.ztbmv
@pytest.mark.parametrize("n", TBMV_SIZES)
@pytest.mark.parametrize("k", TBMV_KS)
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS)
def test_accuracy_ztbmv(n, k, uplo, trans, diag, lda_extra):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype = torch.complex128
    lda = k + 1 + lda_extra
    A = make_triangular_banded(n, k, lda, uplo, diag, dtype, flag_blas.device)
    x = tbmv_randn((max(n, 1),), dtype, flag_blas.device)
    ref_x = tbmv_reference(uplo, trans, diag, n, k, A, lda, x, 1)
    flag_blas.ztbmv(uplo, trans, diag, n, k, A, lda, x, 1)

    tbmv_assert_close(x, ref_x, dtype, reduce_dim=k + 1)


@pytest.mark.ztbmv
@pytest.mark.parametrize("n", TBMV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("uplo", UPLOS)
@pytest.mark.parametrize("trans", COMPLEX_TRANS)
@pytest.mark.parametrize("diag", DIAGS)
@pytest.mark.parametrize("incx", INCS)
@pytest.mark.parametrize("lda_extra", LDA_EXTRAS_STRIDE)
def test_accuracy_ztbmv_stride(n, k, uplo, trans, diag, incx, lda_extra):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype = torch.complex128
    lda = k + 1 + lda_extra
    A = make_triangular_banded(n, k, lda, uplo, diag, dtype, flag_blas.device)
    x = tbmv_randn((1 + (n - 1) * incx,), dtype, flag_blas.device)
    ref_x = tbmv_reference(uplo, trans, diag, n, k, A, lda, x, incx)
    flag_blas.ztbmv(uplo, trans, diag, n, k, A, lda, x, incx)

    tbmv_assert_close(x, ref_x, dtype, reduce_dim=k + 1)
