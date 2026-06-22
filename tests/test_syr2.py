import ctypes
import ctypes.util

import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER

from .accuracy_utils import blas_assert_close, to_reference
from .conftest import TO_CPU


def load_cublas():
    lib_names = ["libcublas.so.13"]
    found_path = ctypes.util.find_library("cublas")
    if found_path:
        lib_names.append(found_path)
    lib_names.extend(["libcublas.so", "libcublas.so.12", "libcublas.so.11"])

    for name in lib_names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError("Unable to find libcublas.so on this system")


_cublas = None
_cublas_handle = None
_CUBLAS_SYR2_FUNCS = None
CUBLAS_POINTER_MODE_HOST = 0


def _configure_cublas_signatures():
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    for name in ("cublasSsyr2_v2", "cublasDsyr2_v2"):
        func = getattr(_cublas, name)
        func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        func.restype = ctypes.c_int


def _ensure_cublas():
    global _cublas, _CUBLAS_SYR2_FUNCS
    if _cublas is None:
        _cublas = load_cublas()
        _configure_cublas_signatures()
        _CUBLAS_SYR2_FUNCS = {
            torch.float32: (_cublas.cublasSsyr2_v2, ctypes.c_float, False),
            torch.float64: (_cublas.cublasDsyr2_v2, ctypes.c_double, False),
        }
    return _cublas


def _make_scalar(ctor, is_complex, value):
    value = value.item() if isinstance(value, torch.Tensor) else value
    if is_complex:
        return ctor(value.real, value.imag)
    return ctor(value)


def _get_cublas_handle():
    global _cublas_handle
    if _cublas_handle is not None:
        return _cublas_handle

    cublas = _ensure_cublas()
    _cublas_handle = ctypes.c_void_p()
    status = cublas.cublasCreate_v2(ctypes.byref(_cublas_handle))
    if status != 0:
        raise RuntimeError(f"cublasCreate_v2 failed with error code: {status}")
    status = cublas.cublasSetPointerMode_v2(_cublas_handle, CUBLAS_POINTER_MODE_HOST)
    if status != 0:
        raise RuntimeError(f"cublasSetPointerMode_v2 failed with error code: {status}")
    return _cublas_handle


def cublas_syr2_reference(uplo, n, alpha, x, incx, y, incy, A, lda):
    if n == 0:
        return

    handle = _get_cublas_handle()
    _ensure_cublas()
    func, ctor, is_complex = _CUBLAS_SYR2_FUNCS[A.dtype]
    alpha_c = _make_scalar(ctor, is_complex, alpha)
    status = func(
        handle,
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
    )
    if status != 0:
        raise RuntimeError(f"cublasXsyr2_v2 execution failed with error code: {status}")
    torch.cuda.synchronize(A.device)


def cpu_syr2_reference(uplo, n, alpha, x, incx, y, incy, A, lda):
    ref_A = A.detach().to("cpu").contiguous()
    if n == 0:
        return ref_A

    ref_x = x.detach().to("cpu").contiguous()
    ref_y = y.detach().to("cpu").contiguous()
    logical_A = ref_A[:n, :n].T.numpy().copy(order="F")
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    lower = int(uplo == CUBLAS_FILL_MODE_LOWER)
    syr2 = cpu_blas.ssyr2 if A.dtype == torch.float32 else cpu_blas.dsyr2
    updated = syr2(
        alpha,
        ref_x.numpy(),
        ref_y.numpy(),
        a=logical_A,
        n=n,
        incx=incx,
        incy=incy,
        lower=lower,
        overwrite_a=1,
    )

    ref_A[:n, :n] = torch.from_numpy(updated.T.copy())
    return ref_A


def syr2_reference(uplo, n, alpha, x, incx, y, incy, A, lda):
    if TO_CPU:
        return cpu_syr2_reference(uplo, n, alpha, x, incx, y, incy, A, lda)

    ref_A = A.clone()
    cublas_syr2_reference(uplo, n, alpha, x, incx, y, incy, ref_A, lda)
    return ref_A


SYR2_SIZES = [
    1,
    2,
    3,
    4,
    7,
    8,
    15,
    16,
    17,
    31,
    32,
    33,
    47,
    48,
    49,
    63,
    64,
    65,
    95,
    96,
    97,
    127,
    128,
    129,
    191,
    192,
    193,
    255,
    256,
    257,
    383,
    384,
    385,
    511,
    512,
    513,
    767,
    768,
    769,
]
SYR2_STRIDE_SIZES = [
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    127,
    128,
    129,
    191,
    192,
    193,
    255,
    256,
    257,
]
FILL_MODES = [CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER]
STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]


def create_syr2_data(n, lda, dtype, device):
    A = torch.zeros((n, lda), dtype=dtype, device=device)
    if n > 0:
        A[:, :n] = torch.randn(n, n, dtype=dtype, device=device)
    return A.contiguous()


def _syr2_reduce_dim(n):
    return max(1, n)


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


def _run_syr2_case(op, dtype, alpha, uplo, n, incx=1, incy=1, lda_extra=2):
    if dtype == torch.float64:
        check_fp64_support()

    lda = max(1, n) + lda_extra
    A = create_syr2_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(1 + max(0, n - 1) * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(1 + max(0, n - 1) * incy, dtype=dtype, device=flag_blas.device)
    ref_A = syr2_reference(uplo, n, alpha, x, incx, y, incy, A, lda)
    op(uplo, n, alpha, x, incx, y, incy, A, lda)

    blas_assert_close(A, ref_A, dtype, reduce_dim=_syr2_reduce_dim(n))


@pytest.mark.ssyr2
@pytest.mark.parametrize(
    "uplo,n",
    [(CUBLAS_FILL_MODE_LOWER, 64), (CUBLAS_FILL_MODE_UPPER, 64)],
    ids=["lower-64", "upper-64"],
)
def test_accuracy_ssyr2(uplo, n):
    dtype, alpha = torch.float32, 1.5
    lda = n + 2

    A = create_syr2_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    ref_A = syr2_reference(uplo, n, alpha, x, 1, y, 1, A, lda)
    flag_blas.ops.ssyr2(uplo, n, alpha, x, 1, y, 1, A, lda)

    blas_assert_close(A, ref_A, dtype, reduce_dim=_syr2_reduce_dim(n))


@pytest.mark.ssyr2
@pytest.mark.parametrize("n", SYR2_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_accuracy_ssyr2_sizes(n, uplo):
    dtype, alpha = torch.float32, -0.75
    lda = max(1, n) + 3

    A = create_syr2_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(max(1, n), dtype=dtype, device=flag_blas.device)
    y = torch.randn(max(1, n), dtype=dtype, device=flag_blas.device)
    ref_A = syr2_reference(uplo, n, alpha, x, 1, y, 1, A, lda)
    flag_blas.ops.ssyr2(uplo, n, alpha, x, 1, y, 1, A, lda)

    blas_assert_close(A, ref_A, dtype, reduce_dim=_syr2_reduce_dim(n))


@pytest.mark.ssyr2
@pytest.mark.parametrize("n", SYR2_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_ssyr2_stride(n, uplo, incx, incy):
    dtype, alpha = torch.float32, 2.0
    lda = n

    A = create_syr2_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(1 + (n - 1) * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(1 + (n - 1) * incy, dtype=dtype, device=flag_blas.device)
    ref_A = syr2_reference(uplo, n, alpha, x, incx, y, incy, A, lda)
    flag_blas.ops.ssyr2(uplo, n, alpha, x, incx, y, incy, A, lda)

    blas_assert_close(A, ref_A, dtype, reduce_dim=_syr2_reduce_dim(n))


@pytest.mark.ssyr2
def test_ssyr2_alpha_zero():
    n, lda = 128, 130
    dtype = torch.float32
    A = create_syr2_data(n, lda, dtype, flag_blas.device)
    A_orig = A.clone()
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)

    ref_A = syr2_reference(CUBLAS_FILL_MODE_UPPER, n, 0.0, x, 1, y, 1, A, lda)
    flag_blas.ops.ssyr2(CUBLAS_FILL_MODE_UPPER, n, 0.0, x, 1, y, 1, A, lda)

    blas_assert_close(A, ref_A, dtype, reduce_dim=_syr2_reduce_dim(n))
    blas_assert_close(A, to_reference(A_orig, upcast=TO_CPU), dtype)


@pytest.mark.ssyr2
def test_ssyr2_n_zero():
    dtype = torch.float32
    A = torch.empty((0, 1), dtype=dtype, device=flag_blas.device)
    x = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    y = torch.empty((0,), dtype=dtype, device=flag_blas.device)

    flag_blas.ops.ssyr2(CUBLAS_FILL_MODE_LOWER, 0, 1.0, x, 1, y, 1, A, 1)
    assert A.numel() == 0


@pytest.mark.dsyr2
@pytest.mark.parametrize(
    "uplo,n",
    [(CUBLAS_FILL_MODE_LOWER, 64), (CUBLAS_FILL_MODE_UPPER, 64)],
    ids=["lower-64", "upper-64"],
)
def test_accuracy_dsyr2(uplo, n):
    check_fp64_support()
    dtype, alpha = torch.float64, 1.5
    lda = n + 2

    A = create_syr2_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n, dtype=dtype, device=flag_blas.device)
    ref_A = syr2_reference(uplo, n, alpha, x, 1, y, 1, A, lda)
    flag_blas.ops.dsyr2(uplo, n, alpha, x, 1, y, 1, A, lda)

    blas_assert_close(A, ref_A, dtype, reduce_dim=_syr2_reduce_dim(n))


@pytest.mark.parametrize(
    "op,dtype,alpha",
    [
        pytest.param(
            flag_blas.ops.dsyr2,
            torch.float64,
            -0.75,
            marks=pytest.mark.dsyr2,
        ),
    ],
)
@pytest.mark.parametrize("n", SYR2_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_accuracy_syr2_variants_sizes(op, dtype, alpha, n, uplo):
    _run_syr2_case(op, dtype, alpha, uplo, n, lda_extra=3)


@pytest.mark.parametrize(
    "op,dtype,alpha",
    [
        pytest.param(flag_blas.ops.dsyr2, torch.float64, 2.0, marks=pytest.mark.dsyr2),
    ],
)
@pytest.mark.parametrize("n", SYR2_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_syr2_variants_stride(op, dtype, alpha, n, uplo, incx, incy):
    _run_syr2_case(op, dtype, alpha, uplo, n, incx=incx, incy=incy, lda_extra=0)


@pytest.mark.parametrize(
    "op,dtype,alpha,uplo",
    [
        pytest.param(
            flag_blas.ops.dsyr2,
            torch.float64,
            0.0,
            CUBLAS_FILL_MODE_UPPER,
            marks=pytest.mark.dsyr2,
        ),
    ],
)
def test_syr2_variants_alpha_zero(op, dtype, alpha, uplo):
    _run_syr2_case(op, dtype, alpha, uplo, 128)


@pytest.mark.parametrize(
    "op,dtype,alpha",
    [
        pytest.param(flag_blas.ops.dsyr2, torch.float64, 1.0, marks=pytest.mark.dsyr2),
    ],
)
def test_syr2_variants_n_zero(op, dtype, alpha):
    if dtype == torch.float64:
        check_fp64_support()

    A = torch.empty((0, 1), dtype=dtype, device=flag_blas.device)
    x = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    y = torch.empty((0,), dtype=dtype, device=flag_blas.device)

    op(CUBLAS_FILL_MODE_LOWER, 0, alpha, x, 1, y, 1, A, 1)
    assert A.numel() == 0
