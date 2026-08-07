import ctypes
import ctypes.util
import math
import pytest
import torch
from scipy.linalg import blas as cpu_blas
import flag_blas

from flag_blas.ops import (
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_DIAG_UNIT,
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


def cublas_tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx):
    if n == 0:
        return

    handle = ctypes.c_void_p()
    status = _cublas.cublasCreate_v2(ctypes.byref(handle))
    if status != 0:
        raise RuntimeError(f"cublasCreate_v2 failed with error code: {status}")
    dtype = A.dtype
    if dtype == torch.float32:
        func = _cublas.cublasStbsv_v2
    elif dtype == torch.float64:
        func = _cublas.cublasDtbsv_v2
    elif dtype == torch.complex64:
        func = _cublas.cublasCtbsv_v2
    elif dtype == torch.complex128:
        func = _cublas.cublasZtbsv_v2
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    try:
        status = func(
            handle,
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
            raise RuntimeError(
                f"cublasXtbsv_v2 execution failed with error code: {status}"
            )
    finally:
        _cublas.cublasDestroy_v2(handle)


def cublas_stbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx):
    cublas_tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx)


def cpu_tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx):
    ref_x = x.detach().cpu().contiguous()
    if n == 0:
        return ref_x

    ref_A = A.detach().cpu().contiguous()
    if ref_A.dtype == torch.float32:
        func = cpu_blas.stbsv
    elif ref_A.dtype == torch.float64:
        func = cpu_blas.dtbsv
    elif ref_A.dtype == torch.complex64:
        func = cpu_blas.ctbsv
    elif ref_A.dtype == torch.complex128:
        func = cpu_blas.ztbsv
    else:
        raise ValueError(f"Unsupported dtype {ref_A.dtype}")

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


def tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx):
    if TO_CPU:
        return cpu_tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx)

    ref_x = x.clone()
    cublas_tbsv_reference(uplo, trans, diag, n, k, A, lda, ref_x, incx)
    return ref_x


STBSV_SIZES = [1, 2, 32, 63, 64, 128, 256, 512, 1024, 4096]
STBSV_KS = [0, 1, 4, 16, 64]
STBSV_STRIDE_SIZES = [64, 127, 256]
COMPLEX_TBSV_SIZES = [0, 1, 2, 31, 32, 33, 64, 127, 128]

FILL_MODES = [CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER]
TRANS_MODES = [CUBLAS_OP_N, CUBLAS_OP_T]
COMPLEX_TRANS_MODES = [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
DIAG_MODES = [CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT]


def make_triangular_banded(n, k, lda, uplo, dtype, device, unit_diag=False):
    if n == 0:
        return torch.zeros((n, lda), dtype=dtype, device=device).contiguous()

    A = torch.randn((n, lda), dtype=dtype, device=device) * 0.1
    diag_floor = 2.0 * (k + 1) + 1.0
    cols = torch.arange(lda, device=device).view(1, lda)
    j = torch.arange(n, device=device).view(n, 1)

    if uplo == CUBLAS_FILL_MODE_UPPER:
        valid = (cols >= torch.clamp(k - j, min=0)) & (cols <= k)
        diag_col = k
    else:
        valid = cols <= torch.clamp(n - 1 - j, max=k)
        diag_col = 0

    A = A.masked_fill(~valid, 0.0)
    if unit_diag:
        A[:, diag_col] = 1.0
    else:
        real_dtype = (
            torch.float64
            if dtype in (torch.float64, torch.complex128)
            else torch.float32
        )
        sign = torch.where(
            torch.rand(n, device=device) < 0.5,
            torch.full((n,), -1.0, dtype=real_dtype, device=device),
            torch.full((n,), 1.0, dtype=real_dtype, device=device),
        )
        A[:, diag_col] = (
            sign * (diag_floor + torch.rand(n, dtype=real_dtype, device=device))
        ).to(dtype)
    return A.contiguous()


def _stbsv_tol(dtype, n, k):
    K = max(1, n)
    if dtype == torch.float32:
        return min(max(1e-4, 5e-6 * math.sqrt(K)), 5e-2)
    if dtype == torch.float64:
        return min(max(1e-12, 5e-14 * math.sqrt(K)), 1e-9)
    if dtype == torch.complex64:
        return min(max(2e-4, 1e-5 * math.sqrt(K)), 1e-1)
    if dtype == torch.complex128:
        return min(max(2e-12, 1e-13 * math.sqrt(K)), 1e-8)
    raise ValueError(f"Unsupported dtype {dtype}")


def _effective_k(n, k):
    return min(k, max(0, n - 1))


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("fp64 is not supported on this device")


def _make_x(length, dtype, device):
    if dtype.is_complex:
        return torch.randn(length, dtype=dtype, device=device) + 1j * torch.randn(
            length, dtype=dtype, device=device
        )
    return torch.randn(length, dtype=dtype, device=device)


@pytest.mark.stbsv
@pytest.mark.parametrize("n", STBSV_SIZES)
@pytest.mark.parametrize("k", STBSV_KS)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", TRANS_MODES)
@pytest.mark.parametrize("diag", DIAG_MODES)
def test_accuracy_stbsv(n, k, uplo, trans, diag):
    k = _effective_k(n, k)
    dtype = torch.float32
    lda = k + 1 + 2

    A = make_triangular_banded(
        n,
        k,
        lda,
        uplo,
        dtype,
        flag_blas.device,
        unit_diag=(diag == CUBLAS_DIAG_UNIT),
    )
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, 1)

    flag_blas.stbsv(uplo, trans, diag, n, k, A, lda, x, 1)

    tol = _stbsv_tol(dtype, n, k)
    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1, atol=tol)


@pytest.mark.stbsv
@pytest.mark.parametrize("n", STBSV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [0, 1, 8, 32])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", TRANS_MODES)
@pytest.mark.parametrize("incx", [1, 2, 3])
def test_accuracy_stbsv_stride(n, k, uplo, trans, incx):
    k = _effective_k(n, k)
    dtype = torch.float32
    diag = CUBLAS_DIAG_NON_UNIT
    lda = k + 1

    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx)

    flag_blas.stbsv(uplo, trans, diag, n, k, A, lda, x, incx)

    tol = _stbsv_tol(dtype, n, k)
    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1, atol=tol)


@pytest.mark.stbsv
def test_stbsv_n_zero():
    A = torch.empty((0, 1), dtype=torch.float32, device=flag_blas.device)
    x = torch.empty((0,), dtype=torch.float32, device=flag_blas.device)
    flag_blas.stbsv(
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
        0,
        0,
        A,
        1,
        x,
        1,
    )
    assert x.numel() == 0


@pytest.mark.stbsv
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", TRANS_MODES)
def test_stbsv_k_zero(uplo, trans):
    n, k = 256, 0
    lda = 1
    dtype = torch.float32
    diag = CUBLAS_DIAG_NON_UNIT

    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, 1)

    flag_blas.stbsv(uplo, trans, diag, n, k, A, lda, x, 1)

    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1, atol=1e-5)


@pytest.mark.dtbsv
@pytest.mark.parametrize("n", STBSV_SIZES)
@pytest.mark.parametrize("k", STBSV_KS)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", TRANS_MODES)
@pytest.mark.parametrize("diag", DIAG_MODES)
def test_accuracy_dtbsv(n, k, uplo, trans, diag):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype = torch.float64
    lda = k + 1 + 2

    A = make_triangular_banded(
        n,
        k,
        lda,
        uplo,
        dtype,
        flag_blas.device,
        unit_diag=(diag == CUBLAS_DIAG_UNIT),
    )
    x = _make_x(n, dtype, flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, 1)

    flag_blas.dtbsv(uplo, trans, diag, n, k, A, lda, x, 1)

    tol = _stbsv_tol(dtype, n, k)
    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1, atol=tol)


@pytest.mark.dtbsv
@pytest.mark.parametrize("n", STBSV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [0, 1, 8, 32])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", TRANS_MODES)
@pytest.mark.parametrize("incx", [1, 2, 3])
def test_accuracy_dtbsv_stride(n, k, uplo, trans, incx):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype = torch.float64
    diag = CUBLAS_DIAG_NON_UNIT
    lda = k + 1

    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = _make_x(1 + (n - 1) * incx, dtype, flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx)

    flag_blas.dtbsv(uplo, trans, diag, n, k, A, lda, x, incx)

    tol = _stbsv_tol(dtype, n, k)
    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1, atol=tol)


@pytest.mark.ctbsv
@pytest.mark.parametrize("n", COMPLEX_TBSV_SIZES)
@pytest.mark.parametrize("k", [0, 1, 8, 32])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", COMPLEX_TRANS_MODES)
@pytest.mark.parametrize("diag", DIAG_MODES)
def test_accuracy_ctbsv(n, k, uplo, trans, diag):
    k = _effective_k(n, k)
    dtype = torch.complex64
    lda = k + 1 + 2

    A = make_triangular_banded(
        n,
        k,
        lda,
        uplo,
        dtype,
        flag_blas.device,
        unit_diag=(diag == CUBLAS_DIAG_UNIT),
    )
    x = _make_x(n, dtype, flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, 1)

    flag_blas.ctbsv(uplo, trans, diag, n, k, A, lda, x, 1)

    tol = _stbsv_tol(dtype, n, k)
    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1, atol=tol)


@pytest.mark.ctbsv
@pytest.mark.parametrize("n", STBSV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [0, 1, 8, 32])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", COMPLEX_TRANS_MODES)
@pytest.mark.parametrize("incx", [1, 2, 3])
def test_accuracy_ctbsv_stride(n, k, uplo, trans, incx):
    k = _effective_k(n, k)
    dtype = torch.complex64
    diag = CUBLAS_DIAG_NON_UNIT
    lda = k + 1

    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = _make_x(1 + (n - 1) * incx, dtype, flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx)

    flag_blas.ctbsv(uplo, trans, diag, n, k, A, lda, x, incx)

    tol = _stbsv_tol(dtype, n, k)
    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1, atol=tol)


@pytest.mark.ztbsv
@pytest.mark.parametrize("n", COMPLEX_TBSV_SIZES)
@pytest.mark.parametrize("k", [0, 1, 8, 32])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", COMPLEX_TRANS_MODES)
@pytest.mark.parametrize("diag", DIAG_MODES)
def test_accuracy_ztbsv(n, k, uplo, trans, diag):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype = torch.complex128
    lda = k + 1 + 2

    A = make_triangular_banded(
        n,
        k,
        lda,
        uplo,
        dtype,
        flag_blas.device,
        unit_diag=(diag == CUBLAS_DIAG_UNIT),
    )
    x = _make_x(n, dtype, flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, 1)

    flag_blas.ztbsv(uplo, trans, diag, n, k, A, lda, x, 1)

    tol = _stbsv_tol(dtype, n, k)
    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1, atol=tol)


@pytest.mark.ztbsv
@pytest.mark.parametrize("n", STBSV_STRIDE_SIZES)
@pytest.mark.parametrize("k", [0, 1, 8, 32])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", COMPLEX_TRANS_MODES)
@pytest.mark.parametrize("incx", [1, 2, 3])
def test_accuracy_ztbsv_stride(n, k, uplo, trans, incx):
    check_fp64_support()
    k = _effective_k(n, k)
    dtype = torch.complex128
    diag = CUBLAS_DIAG_NON_UNIT
    lda = k + 1

    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = _make_x(1 + (n - 1) * incx, dtype, flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, incx)

    flag_blas.ztbsv(uplo, trans, diag, n, k, A, lda, x, incx)

    tol = _stbsv_tol(dtype, n, k)
    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1, atol=tol)


@pytest.mark.stbsv
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("trans", TRANS_MODES)
def test_stbsv_unit_diag_ignored(uplo, trans):
    n, k = 128, 8
    lda = k + 1 + 1
    dtype = torch.float32

    A_clean = make_triangular_banded(
        n, k, lda, uplo, dtype, flag_blas.device, unit_diag=True
    )
    A_dirty = A_clean.clone()
    diag_row = k if uplo == CUBLAS_FILL_MODE_UPPER else 0
    A_dirty[:, diag_row] = float("nan")

    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    x_clean = x.clone()
    x_dirty = x.clone()

    flag_blas.stbsv(uplo, trans, CUBLAS_DIAG_UNIT, n, k, A_clean, lda, x_clean, 1)
    flag_blas.stbsv(uplo, trans, CUBLAS_DIAG_UNIT, n, k, A_dirty, lda, x_dirty, 1)

    tol = _stbsv_tol(dtype, n, k)
    ref_x_clean = x_clean.cpu() if TO_CPU else x_clean
    blas_assert_close(x_dirty, ref_x_clean, dtype, reduce_dim=k + 1, atol=tol)


@pytest.mark.stbsv
def test_stbsv_solve_then_multiply_roundtrip():
    n, k = 512, 16
    lda = k + 1
    dtype = torch.float32
    uplo = CUBLAS_FILL_MODE_LOWER

    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x_orig = torch.randn(n, dtype=dtype, device=flag_blas.device)

    x_buf = x_orig.clone()
    flag_blas.stbsv(uplo, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, A, lda, x_buf, 1)

    A_dense = torch.zeros(n, n, dtype=dtype, device=flag_blas.device)
    rows = torch.arange(n, device=flag_blas.device).view(n, 1)
    cols = torch.arange(n, device=flag_blas.device).view(1, n)
    mask = (rows >= cols) & ((rows - cols) <= k)
    A_dense[mask] = A[cols.expand(n, n)[mask], (rows - cols)[mask]]
    Ay = A_dense @ x_buf

    tol = _stbsv_tol(dtype, n, k)
    ref_x_orig = x_orig.cpu() if TO_CPU else x_orig
    blas_assert_close(Ay, ref_x_orig, dtype, reduce_dim=k + 1, atol=tol)


TBSV_VARIANTS = [
    pytest.param(flag_blas.stbsv, torch.float32, CUBLAS_OP_T, id="stbsv"),
    pytest.param(flag_blas.dtbsv, torch.float64, CUBLAS_OP_T, id="dtbsv"),
    pytest.param(flag_blas.ctbsv, torch.complex64, CUBLAS_OP_C, id="ctbsv"),
    pytest.param(flag_blas.ztbsv, torch.complex128, CUBLAS_OP_C, id="ztbsv"),
]

NARROW_TBSV_VARIANTS = [
    pytest.param(
        flag_blas.dtbsv,
        torch.float64,
        CUBLAS_OP_N,
        id="dtbsv-n",
        marks=pytest.mark.dtbsv,
    ),
    pytest.param(
        flag_blas.dtbsv,
        torch.float64,
        CUBLAS_OP_T,
        id="dtbsv-t",
        marks=pytest.mark.dtbsv,
    ),
    pytest.param(
        flag_blas.ztbsv,
        torch.complex128,
        CUBLAS_OP_N,
        id="ztbsv-n",
        marks=pytest.mark.ztbsv,
    ),
    pytest.param(
        flag_blas.ztbsv,
        torch.complex128,
        CUBLAS_OP_T,
        id="ztbsv-t",
        marks=pytest.mark.ztbsv,
    ),
    pytest.param(
        flag_blas.ztbsv,
        torch.complex128,
        CUBLAS_OP_C,
        id="ztbsv-c",
        marks=pytest.mark.ztbsv,
    ),
]


@pytest.mark.parametrize("op,dtype,trans", NARROW_TBSV_VARIANTS)
@pytest.mark.parametrize("n", [127, 128, 129, 255, 256, 257, 513])
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("diag", DIAG_MODES)
def test_tbsv_k1_panel_boundaries(op, dtype, trans, n, uplo, diag):
    check_fp64_support()
    k, lda = 1, 4
    A = make_triangular_banded(
        n,
        k,
        lda,
        uplo,
        dtype,
        flag_blas.device,
        unit_diag=(diag == CUBLAS_DIAG_UNIT),
    )
    x = _make_x(n, dtype, flag_blas.device)
    ref_x = tbsv_reference(uplo, trans, diag, n, k, A, lda, x, 1)

    op(uplo, trans, diag, n, k, A, lda, x, 1)

    tol = _stbsv_tol(dtype, n, k)
    blas_assert_close(x, ref_x, dtype, reduce_dim=2, atol=tol)


@pytest.mark.dtbsv
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_dtbsv_large_k4_no_trans(uplo):
    check_fp64_support()
    n, k, lda = 8192, 4, 7
    dtype = torch.float64
    A = make_triangular_banded(n, k, lda, uplo, dtype, flag_blas.device)
    x = _make_x(n, dtype, flag_blas.device)
    ref_x = tbsv_reference(
        uplo, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, A, lda, x, 1
    )

    flag_blas.dtbsv(
        uplo, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, A, lda, x, 1
    )

    tol = _stbsv_tol(dtype, n, k)
    blas_assert_close(x, ref_x, dtype, reduce_dim=k + 1, atol=tol)


@pytest.mark.ztbsv
@pytest.mark.parametrize("trans", COMPLEX_TRANS_MODES)
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_ztbsv_k1_complex_diagonal_phase(trans, uplo):
    check_fp64_support()
    n, k, lda = 129, 1, 4
    A = make_triangular_banded(
        n, k, lda, uplo, torch.complex128, flag_blas.device
    )
    diag_col = k if uplo == CUBLAS_FILL_MODE_UPPER else 0
    diagonal_imag = torch.linspace(
        0.25, 0.75, n, dtype=torch.float64, device=flag_blas.device
    )
    A[:, diag_col] = torch.complex(A[:, diag_col].real, diagonal_imag)
    x = _make_x(n, torch.complex128, flag_blas.device)
    ref_x = tbsv_reference(
        uplo, trans, CUBLAS_DIAG_NON_UNIT, n, k, A, lda, x, 1
    )

    flag_blas.ztbsv(uplo, trans, CUBLAS_DIAG_NON_UNIT, n, k, A, lda, x, 1)

    tol = _stbsv_tol(torch.complex128, n, k)
    blas_assert_close(x, ref_x, torch.complex128, reduce_dim=2, atol=tol)


@pytest.mark.parametrize("op,dtype,trans", TBSV_VARIANTS)
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_tbsv_unit_diag_ignores_stored_diagonal(op, dtype, trans, uplo):
    if dtype in (torch.float64, torch.complex128):
        check_fp64_support()
    n, k, lda = 33, 8, 11
    clean = make_triangular_banded(
        n, k, lda, uplo, dtype, flag_blas.device, unit_diag=True
    )
    dirty = clean.clone()
    diag_col = k if uplo == CUBLAS_FILL_MODE_UPPER else 0
    if dtype.is_complex:
        dirty[:, diag_col] = complex(float("nan"), float("nan"))
    else:
        dirty[:, diag_col] = float("nan")
    x = _make_x(n, dtype, flag_blas.device)
    clean_x = x.clone()
    dirty_x = x.clone()

    op(uplo, trans, CUBLAS_DIAG_UNIT, n, k, clean, lda, clean_x, 1)
    op(uplo, trans, CUBLAS_DIAG_UNIT, n, k, dirty, lda, dirty_x, 1)

    tol = _stbsv_tol(dtype, n, k)
    torch.testing.assert_close(dirty_x, clean_x, rtol=tol, atol=tol)
