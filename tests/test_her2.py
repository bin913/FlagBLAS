import ctypes
import ctypes.util

import numpy as np
import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER

from .accuracy_utils import blas_assert_close, to_cpu_blas_tensor
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
_CUBLAS_HER2_FUNCS = None
CUBLAS_POINTER_MODE_HOST = 0


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def _configure_cublas_signatures():
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    for name in ("cublasCher2_v2", "cublasZher2_v2"):
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
    global _cublas, _CUBLAS_HER2_FUNCS
    if _cublas is None:
        _cublas = load_cublas()
        _configure_cublas_signatures()
        _CUBLAS_HER2_FUNCS = {
            torch.complex64: (_cublas.cublasCher2_v2, cuComplex),
            torch.complex128: (_cublas.cublasZher2_v2, cuDoubleComplex),
        }
    return _cublas


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


def check_fp64_support():
    if not getattr(flag_blas.runtime.device, "support_fp64", True):
        pytest.skip("No FP64 support on this device")


def _make_scalar(ctor, value):
    value = value.item() if isinstance(value, torch.Tensor) else value
    return ctor(value.real, value.imag)


def cublas_her2_reference(uplo, n, alpha, x, incx, y, incy, A, lda):
    if n == 0:
        return
    handle = _get_cublas_handle()
    _ensure_cublas()
    func, ctor = _CUBLAS_HER2_FUNCS[A.dtype]
    alpha_c = _make_scalar(ctor, alpha)
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
        raise RuntimeError(f"cublasXher2_v2 execution failed with error code: {status}")
    torch.cuda.synchronize(A.device)


def cpu_her2_reference(uplo, n, alpha, x, incx, y, incy, A, lda):
    ref_A = to_cpu_blas_tensor(A)
    if n == 0:
        return ref_A
    ref_x = to_cpu_blas_tensor(x)
    ref_y = to_cpu_blas_tensor(y)
    logical_A = np.array(ref_A[:n, :n].T.numpy(), order="F", copy=True)
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    her2 = cpu_blas.cher2 if A.dtype == torch.complex64 else cpu_blas.zher2
    updated = her2(
        alpha,
        ref_x.numpy(),
        ref_y.numpy(),
        lower=int(uplo == CUBLAS_FILL_MODE_LOWER),
        incx=incx,
        incy=incy,
        n=n,
        a=logical_A,
        overwrite_a=1,
    )
    ref_A[:n, :n] = torch.from_numpy(np.array(updated.T, order="C", copy=True))
    return ref_A


def her2_reference(uplo, n, alpha, x, incx, y, incy, A, lda):
    if TO_CPU:
        return cpu_her2_reference(uplo, n, alpha, x, incx, y, incy, A, lda)
    ref_A = A.clone()
    cublas_her2_reference(uplo, n, alpha, x, incx, y, incy, ref_A, lda)
    return ref_A


HER2_SIZES = [
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
]
HER2_STRIDE_SIZES = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256]
FILL_MODES = [CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER]
STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]


def create_hermitian_data(n, lda, dtype, device):
    A = torch.zeros((n, lda), dtype=dtype, device=device)
    if n > 0:
        data = torch.randn(n, n, dtype=dtype, device=device)
        diag_real = data.diagonal().real.clone()
        data.diagonal().copy_(diag_real.to(dtype))
        A[:, :n] = data
    return A.contiguous()


def _run_her2_case(op, dtype, alpha, uplo, n, incx=1, incy=1, lda_extra=2):
    if dtype == torch.complex128:
        check_fp64_support()
    lda = max(1, n) + lda_extra
    A = create_hermitian_data(n, lda, dtype, flag_blas.device)
    x = torch.randn(1 + max(0, n - 1) * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(1 + max(0, n - 1) * incy, dtype=dtype, device=flag_blas.device)
    ref_A = her2_reference(uplo, n, alpha, x, incx, y, incy, A, lda)
    op(uplo, n, alpha, x, incx, y, incy, A, lda)
    blas_assert_close(A, ref_A, dtype, reduce_dim=max(1, n))


@pytest.mark.cher2
@pytest.mark.parametrize("n", HER2_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_accuracy_cher2_sizes(n, uplo):
    _run_her2_case(flag_blas.ops.cher2, torch.complex64, 1.5 + 0.5j, uplo, n)


@pytest.mark.cher2
@pytest.mark.parametrize("n", HER2_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_cher2_stride(n, uplo, incx, incy):
    _run_her2_case(
        flag_blas.ops.cher2, torch.complex64, -0.75 + 0.25j, uplo, n, incx, incy
    )


@pytest.mark.zher2
@pytest.mark.parametrize("n", HER2_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_accuracy_zher2_sizes(n, uplo):
    _run_her2_case(flag_blas.ops.zher2, torch.complex128, 1.5 + 0.5j, uplo, n)


@pytest.mark.zher2
@pytest.mark.parametrize("n", HER2_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_zher2_stride(n, uplo, incx, incy):
    _run_her2_case(
        flag_blas.ops.zher2, torch.complex128, -0.75 + 0.25j, uplo, n, incx, incy
    )


@pytest.mark.parametrize(
    "dtype,op,alpha",
    [
        (torch.complex64, flag_blas.ops.cher2, 1.0 + 0.5j),
        (torch.complex128, flag_blas.ops.zher2, 1.0 + 0.5j),
    ],
)
def test_her2_n_zero(dtype, op, alpha):
    if dtype == torch.complex128:
        check_fp64_support()
    A = torch.empty((0, 1), dtype=dtype, device=flag_blas.device)
    x = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    y = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    op(CUBLAS_FILL_MODE_UPPER, 0, alpha, x, 1, y, 1, A, 1)
    assert A.numel() == 0
