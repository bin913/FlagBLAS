import ctypes
import ctypes.util

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
_CUBLAS_SPR2_FUNCS = None
CUBLAS_POINTER_MODE_HOST = 0


def _configure_cublas_signatures():
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    for name in ("cublasSspr2_v2", "cublasDspr2_v2"):
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
        ]
        func.restype = ctypes.c_int


def _ensure_cublas():
    global _cublas, _CUBLAS_SPR2_FUNCS
    if _cublas is None:
        _cublas = load_cublas()
        _configure_cublas_signatures()
        _CUBLAS_SPR2_FUNCS = {
            torch.float32: (_cublas.cublasSspr2_v2, ctypes.c_float),
            torch.float64: (_cublas.cublasDspr2_v2, ctypes.c_double),
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


def cublas_spr2_reference(uplo, n, alpha, x, incx, y, incy, AP):
    if n == 0:
        return
    handle = _get_cublas_handle()
    _ensure_cublas()
    func, ctor = _CUBLAS_SPR2_FUNCS[AP.dtype]
    alpha_c = ctor(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    status = func(
        handle,
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
        ctypes.c_void_p(AP.data_ptr()),
    )
    if status != 0:
        raise RuntimeError(f"cublasXspr2_v2 execution failed with error code: {status}")
    torch.cuda.synchronize(AP.device)


def cpu_spr2_reference(uplo, n, alpha, x, incx, y, incy, AP):
    ref_AP = to_cpu_blas_tensor(AP)
    if n == 0:
        return ref_AP
    ref_x = to_cpu_blas_tensor(x)
    ref_y = to_cpu_blas_tensor(y)
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    spr2 = cpu_blas.sspr2 if AP.dtype == torch.float32 else cpu_blas.dspr2
    updated = spr2(
        n,
        alpha,
        ref_x.numpy(),
        ref_y.numpy(),
        ref_AP.numpy(),
        incx=incx,
        incy=incy,
        lower=int(uplo == CUBLAS_FILL_MODE_LOWER),
        overwrite_ap=1,
    )
    return torch.from_numpy(updated)


def spr2_reference(uplo, n, alpha, x, incx, y, incy, AP):
    if TO_CPU:
        return cpu_spr2_reference(uplo, n, alpha, x, incx, y, incy, AP)
    ref_AP = AP.clone()
    cublas_spr2_reference(uplo, n, alpha, x, incx, y, incy, ref_AP)
    return ref_AP


SPR2_SIZES = [
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
SPR2_STRIDE_SIZES = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256]
FILL_MODES = [CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER]
STRIDES = [(1, 1), (2, 1), (1, 2), (2, 2)]


def make_packed(n, dtype, device):
    return torch.randn(n * (n + 1) // 2, dtype=dtype, device=device)


def _run_spr2_case(op, dtype, alpha, uplo, n, incx=1, incy=1):
    if dtype is torch.float64:
        check_fp64_support()
    AP = make_packed(n, dtype, flag_blas.device)
    x = torch.randn(1 + max(0, n - 1) * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(1 + max(0, n - 1) * incy, dtype=dtype, device=flag_blas.device)
    ref_AP = spr2_reference(uplo, n, alpha, x, incx, y, incy, AP)
    op(uplo, n, alpha, x, incx, y, incy, AP)
    blas_assert_close(AP, ref_AP, dtype, reduce_dim=max(1, n))


@pytest.mark.sspr2
@pytest.mark.parametrize("n", SPR2_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_accuracy_sspr2_sizes(n, uplo):
    _run_spr2_case(flag_blas.ops.sspr2, torch.float32, 1.5, uplo, n)


@pytest.mark.sspr2
@pytest.mark.parametrize("n", SPR2_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_sspr2_stride(n, uplo, incx, incy):
    _run_spr2_case(flag_blas.ops.sspr2, torch.float32, -0.75, uplo, n, incx, incy)


@pytest.mark.dspr2
@pytest.mark.parametrize("n", SPR2_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
def test_accuracy_dspr2_sizes(n, uplo):
    _run_spr2_case(flag_blas.ops.dspr2, torch.float64, 1.5, uplo, n)


@pytest.mark.dspr2
@pytest.mark.parametrize("n", SPR2_STRIDE_SIZES)
@pytest.mark.parametrize("uplo", FILL_MODES)
@pytest.mark.parametrize("incx,incy", STRIDES)
def test_accuracy_dspr2_stride(n, uplo, incx, incy):
    _run_spr2_case(flag_blas.ops.dspr2, torch.float64, -0.75, uplo, n, incx, incy)


@pytest.mark.parametrize(
    "dtype,op,alpha",
    [
        (torch.float32, flag_blas.ops.sspr2, 1.0),
        (torch.float64, flag_blas.ops.dspr2, 1.0),
    ],
)
def test_spr2_n_zero(dtype, op, alpha):
    if dtype is torch.float64:
        check_fp64_support()
    AP = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    x = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    y = torch.empty((0,), dtype=dtype, device=flag_blas.device)
    op(CUBLAS_FILL_MODE_UPPER, 0, alpha, x, 1, y, 1, AP)
    assert AP.numel() == 0
