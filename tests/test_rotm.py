import ctypes
import ctypes.util

import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas
from .accuracy_utils import (
    L1_PAIR_STRIDES,
    L1_STRIDE_SHAPES,
    blas_assert_close,
    to_reference,
)
from .conftest import TO_CPU

ROTM_FLAGS = [-2.0, -1.0, 0.0, 1.0]
ROTM_OPS = ["srotm", "drotm"]


def _build_param(dtype, device, flag):
    return torch.tensor([flag, 0.75, -0.5, 0.25, 1.5], dtype=dtype, device=device)


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
    raise RuntimeError("Cannot find libcublas.so in the system")


_cublas = None
_cublas_handle = None


def _get_cublas():
    global _cublas
    if _cublas is not None:
        return _cublas

    _cublas = load_cublas()
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    for rotm_name in ("cublasSrotm_v2", "cublasDrotm_v2"):
        rotm_func = getattr(_cublas, rotm_name)
        rotm_func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        rotm_func.restype = ctypes.c_int
    return _cublas


def _get_cublas_handle():
    global _cublas_handle
    if _cublas_handle is not None:
        return _cublas_handle

    _cublas_handle = ctypes.c_void_p()
    status = _get_cublas().cublasCreate_v2(ctypes.byref(_cublas_handle))
    if status != 0:
        raise RuntimeError(f"cuBLAS handle creation failed, error code: {status}")
    return _cublas_handle


def _rotm_host_param(dtype, param):
    values = param.detach().cpu().tolist()
    if dtype == torch.float32:
        return (ctypes.c_float * 5)(*values)
    return (ctypes.c_double * 5)(*values)


def _cublas_rotm_reference(n, x, incx, y, incy, param):
    ref_x = x.clone()
    ref_y = y.clone()
    if n <= 0:
        return ref_x, ref_y

    func = (
        _get_cublas().cublasSrotm_v2
        if x.dtype == torch.float32
        else _get_cublas().cublasDrotm_v2
    )
    param_host = _rotm_host_param(x.dtype, param)
    status = func(
        _get_cublas_handle(),
        ctypes.c_int(n),
        ctypes.c_void_p(ref_x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.c_void_p(ref_y.data_ptr()),
        ctypes.c_int(incy),
        ctypes.cast(param_host, ctypes.c_void_p),
    )
    if status != 0:
        raise RuntimeError(f"cuBLAS rotm execution failed, error code: {status}")
    return ref_x, ref_y


def _scipy_rotm_reference(n, x, incx, y, incy, param):
    ref_x = x.detach().cpu().clone().contiguous()
    ref_y = y.detach().cpu().clone().contiguous()
    if n <= 0:
        return ref_x, ref_y

    x_np = ref_x.numpy()
    y_np = ref_y.numpy()
    param_np = param.detach().cpu().numpy()
    func = cpu_blas.srotm if x.dtype == torch.float32 else cpu_blas.drotm
    func(
        x_np,
        y_np,
        param_np,
        n=n,
        incx=incx,
        incy=incy,
        overwrite_x=1,
        overwrite_y=1,
    )
    return ref_x, ref_y


def _rotm_reference(n, x, incx, y, incy, param):
    if TO_CPU:
        return _scipy_rotm_reference(n, x, incx, y, incy, param)
    return _cublas_rotm_reference(n, x, incx, y, incy, param)


def _call_rotm(dtype, n, x, incx, y, incy, param):
    if dtype == torch.float32:
        flag_blas.ops.srotm(n, x, incx, y, incy, param)
    else:
        flag_blas.ops.drotm(n, x, incx, y, incy, param)


@pytest.mark.rotm
@pytest.mark.parametrize("op_name", ROTM_OPS)
def test_accuracy_rotm_flag_blas_api_symbols(op_name):
    assert callable(getattr(flag_blas.ops, op_name))
    assert callable(getattr(flag_blas, op_name))
    assert getattr(flag_blas.ops, op_name) is getattr(flag_blas, op_name)


@pytest.mark.rotm
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", L1_STRIDE_SHAPES[:4])
@pytest.mark.parametrize("incx,incy", L1_PAIR_STRIDES)
@pytest.mark.parametrize("flag", ROTM_FLAGS)
def test_accuracy_rotm_real(dtype, shape, incx, incy, flag):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = shape[0]
    param = _build_param(dtype, flag_blas.device, flag)
    x = torch.randn(n * incx, dtype=dtype, device=flag_blas.device)
    y = torch.randn(n * incy, dtype=dtype, device=flag_blas.device)
    ref_x, ref_y = _rotm_reference(n, x, incx, y, incy, param)

    _call_rotm(dtype, n, x, incx, y, incy, param)

    blas_assert_close(x, ref_x, dtype)
    blas_assert_close(y, ref_y, dtype)


@pytest.mark.rotm
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accuracy_rotm_empty_tensor(dtype):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_blas.device)
    y = torch.randn(0, dtype=dtype, device=flag_blas.device)
    param = _build_param(dtype, flag_blas.device, -1.0)
    _call_rotm(dtype, 0, x, 1, y, 1, param)
    assert x.numel() == 0
    assert y.numel() == 0


@pytest.mark.rotm
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n,vec_size", [(1, 10), (5, 10), (10, 20), (100, 1000)])
@pytest.mark.parametrize("flag", ROTM_FLAGS)
def test_accuracy_rotm_different_n_real(dtype, n, vec_size, flag):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    param = _build_param(dtype, flag_blas.device, flag)
    x = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    y = torch.randn(vec_size, dtype=dtype, device=flag_blas.device)
    ref_x, ref_y = _rotm_reference(n, x, 1, y, 1, param)

    _call_rotm(dtype, n, x, 1, y, 1, param)

    blas_assert_close(x, ref_x, dtype)
    blas_assert_close(y, ref_y, dtype)


@pytest.mark.rotm
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accuracy_rotm_flag_blas_api(dtype):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=flag_blas.device)
    y = torch.tensor([4.0, 5.0, 6.0], dtype=dtype, device=flag_blas.device)
    param = torch.tensor(
        [-1.0, 0.75, -0.5, 0.25, 1.5],
        dtype=dtype,
        device=flag_blas.device,
    )

    if dtype == torch.float32:
        flag_blas.srotm(3, x, 1, y, 1, param)
    else:
        flag_blas.drotm(3, x, 1, y, 1, param)

    expected_x = to_reference(
        torch.tensor([1.75, 2.75, 3.75], dtype=dtype, device=flag_blas.device)
    )
    expected_y = to_reference(
        torch.tensor([5.5, 6.5, 7.5], dtype=dtype, device=flag_blas.device)
    )
    blas_assert_close(x, expected_x, dtype)
    blas_assert_close(y, expected_y, dtype)


@pytest.mark.rotm
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("incx,incy", [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_rotm_rejects_nonpositive_strides(dtype, incx, incy):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(4, dtype=dtype, device=flag_blas.device)
    y = torch.randn(4, dtype=dtype, device=flag_blas.device)
    param = _build_param(dtype, flag_blas.device, -1.0)

    with pytest.raises(AssertionError, match="inc.*must be positive"):
        _call_rotm(dtype, 4, x, incx, y, incy, param)


@pytest.mark.rotm
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotm_rejects_short_vectors(dtype):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    n = 4
    incx = 2
    incy = 2
    x = torch.randn(6, dtype=dtype, device=flag_blas.device)
    y = torch.randn(1 + (n - 1) * incy, dtype=dtype, device=flag_blas.device)
    param = _build_param(dtype, flag_blas.device, -1.0)

    with pytest.raises(AssertionError, match="x is too short"):
        _call_rotm(dtype, n, x, incx, y, incy, param)


@pytest.mark.rotm
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotm_rejects_bad_param_shape(dtype):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(4, dtype=dtype, device=flag_blas.device)
    y = torch.randn(4, dtype=dtype, device=flag_blas.device)
    param = torch.zeros(4, dtype=dtype, device=flag_blas.device)

    with pytest.raises(AssertionError, match="param must have exactly five elements"):
        _call_rotm(dtype, 4, x, 1, y, 1, param)


@pytest.mark.rotm
def test_srotm_rejects_mismatched_param_dtype():
    x = torch.randn(4, dtype=torch.float32, device=flag_blas.device)
    y = torch.randn(4, dtype=torch.float32, device=flag_blas.device)
    param = _build_param(torch.float64, flag_blas.device, -1.0)

    with pytest.raises(AssertionError, match="param must be float32"):
        flag_blas.ops.srotm(4, x, 1, y, 1, param)
