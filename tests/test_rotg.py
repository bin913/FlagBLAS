import ctypes
import ctypes.util
import glob
import os

import pytest
import scipy
import torch

import flag_blas

from .accuracy_utils import blas_assert_close, to_reference
from .conftest import TO_CPU

ROTG_REAL_CASES = [
    (3.0, 4.0),
    (4.0, 3.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 0.0),
    (-3.0, 4.0),
    (3.0, -4.0),
    (-3.0, -4.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (1e-8, 1e-8),
    (1e-20, 1.0),
    (1.0, 1e-20),
    (1e8, 1e8),
    (1e20, -1e20),
]

ROTG_COMPLEX_CASES = [
    (1.0 + 2.0j, 3.0 + 4.0j),
    (0.0 + 0.0j, 1.0 + 0.0j),
    (0.0 + 0.0j, 0.0 + 1.0j),
    (1.0 + 0.0j, 0.0 + 0.0j),
    (0.0 + 1.0j, 0.0 + 0.0j),
    (0.0 + 0.0j, 0.0 + 0.0j),
    (-2.0 + 3.0j, 4.0 - 5.0j),
    (-2.0 - 3.0j, -4.0 + 5.0j),
    (3.0 + 0.0j, 0.0 + 4.0j),
    (0.0 + 3.0j, 4.0 + 0.0j),
    (1.0 - 1.0j, -1.0 + 1.0j),
    (1e-8 + 1e-8j, 1e-8 - 1e-8j),
    (1e-20 + 0.0j, 1.0 + 0.0j),
    (1.0 + 0.0j, 1e-20 + 0.0j),
    (1e8 + 0.0j, 0.0 + 1e8j),
    (1e20 + 1e20j, -1e20 + 1e20j),
]

ROTG_OPS = ["srotg", "drotg", "crotg", "zrotg"]
CUBLAS_POINTER_MODE_DEVICE = 1


def load_cpu_blas():
    lib_paths = glob.glob(
        os.path.join(
            os.path.dirname(scipy.__file__),
            "..",
            "scipy.libs",
            "libscipy_openblas*.so",
        )
    )
    for path in lib_paths:
        try:
            return ctypes.cdll.LoadLibrary(path)
        except OSError:
            continue
    raise RuntimeError("Cannot find SciPy OpenBLAS library")


_cpu_blas_lib = load_cpu_blas()
_cublas = None
_cublas_handle = None


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


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


def _get_cublas():
    global _cublas
    if _cublas is not None:
        return _cublas

    _cublas = load_cublas()
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    for rotg_name in (
        "cublasSrotg_v2",
        "cublasDrotg_v2",
        "cublasCrotg_v2",
        "cublasZrotg_v2",
    ):
        rotg_func = getattr(_cublas, rotg_name)
        rotg_func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        rotg_func.restype = ctypes.c_int
    return _cublas


def _get_cublas_handle():
    global _cublas_handle
    cublas_lib = _get_cublas()
    if _cublas_handle is not None:
        return _cublas_handle

    _cublas_handle = ctypes.c_void_p()
    status = cublas_lib.cublasCreate_v2(ctypes.byref(_cublas_handle))
    if status != 0:
        raise RuntimeError(f"cuBLAS handle creation failed, error code: {status}")
    status = cublas_lib.cublasSetPointerMode_v2(
        _cublas_handle, CUBLAS_POINTER_MODE_DEVICE
    )
    if status != 0:
        raise RuntimeError(f"cuBLAS pointer mode setup failed, error code: {status}")
    return _cublas_handle


def _call_rotg(dtype, a, b, c, s):
    if dtype == torch.float32:
        flag_blas.ops.srotg(a, b, c, s)
    elif dtype == torch.float64:
        flag_blas.ops.drotg(a, b, c, s)
    elif dtype == torch.complex64:
        flag_blas.ops.crotg(a, b, c, s)
    else:
        flag_blas.ops.zrotg(a, b, c, s)


def _rotg_real_dtype(dtype):
    return torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64


def _scipy_rotg_reference(dtype, a, b):
    if dtype == torch.float32:
        a_ref = ctypes.c_float(a)
        b_ref = ctypes.c_float(b)
        c_ref = ctypes.c_float(0.0)
        s_ref = ctypes.c_float(0.0)
        _cpu_blas_lib.scipy_srotg_(
            ctypes.byref(a_ref),
            ctypes.byref(b_ref),
            ctypes.byref(c_ref),
            ctypes.byref(s_ref),
        )
        return a_ref.value, b_ref.value, c_ref.value, s_ref.value

    if dtype == torch.float64:
        a_ref = ctypes.c_double(a)
        b_ref = ctypes.c_double(b)
        c_ref = ctypes.c_double(0.0)
        s_ref = ctypes.c_double(0.0)
        _cpu_blas_lib.scipy_drotg_(
            ctypes.byref(a_ref),
            ctypes.byref(b_ref),
            ctypes.byref(c_ref),
            ctypes.byref(s_ref),
        )
        return a_ref.value, b_ref.value, c_ref.value, s_ref.value

    if dtype == torch.complex64:
        a_ref = cuComplex(a.real, a.imag)
        b_ref = cuComplex(b.real, b.imag)
        c_ref = ctypes.c_float(0.0)
        s_ref = cuComplex(0.0, 0.0)
        _cpu_blas_lib.scipy_crotg_(
            ctypes.byref(a_ref),
            ctypes.byref(b_ref),
            ctypes.byref(c_ref),
            ctypes.byref(s_ref),
        )
        return (
            complex(a_ref.x, a_ref.y),
            complex(b_ref.x, b_ref.y),
            c_ref.value,
            complex(s_ref.x, s_ref.y),
        )

    a_ref = cuDoubleComplex(a.real, a.imag)
    b_ref = cuDoubleComplex(b.real, b.imag)
    c_ref = ctypes.c_double(0.0)
    s_ref = cuDoubleComplex(0.0, 0.0)
    _cpu_blas_lib.scipy_zrotg_(
        ctypes.byref(a_ref),
        ctypes.byref(b_ref),
        ctypes.byref(c_ref),
        ctypes.byref(s_ref),
    )
    return (
        complex(a_ref.x, a_ref.y),
        complex(b_ref.x, b_ref.y),
        c_ref.value,
        complex(s_ref.x, s_ref.y),
    )


def _cublas_rotg_reference(dtype, a_val, b_val):
    real_dtype = _rotg_real_dtype(dtype)
    a = torch.tensor([a_val], dtype=dtype, device=flag_blas.device)
    b = torch.tensor([b_val], dtype=dtype, device=flag_blas.device)
    c = torch.zeros(1, dtype=real_dtype, device=flag_blas.device)
    s = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    if dtype == torch.float32:
        func = _get_cublas().cublasSrotg_v2
    elif dtype == torch.float64:
        func = _get_cublas().cublasDrotg_v2
    elif dtype == torch.complex64:
        func = _get_cublas().cublasCrotg_v2
    else:
        func = _get_cublas().cublasZrotg_v2

    status = func(
        _get_cublas_handle(),
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c.data_ptr()),
        ctypes.c_void_p(s.data_ptr()),
    )
    if status != 0:
        raise RuntimeError(f"cuBLAS rotg execution failed, error code: {status}")
    return a.cpu()[0].item(), b.cpu()[0].item(), c.cpu()[0].item(), s.cpu()[0].item()


def _rotg_reference(dtype, a, b):
    if TO_CPU:
        return _scipy_rotg_reference(dtype, a, b)
    return _cublas_rotg_reference(dtype, a, b)


def _reference_tensor(values, dtype):
    return to_reference(torch.tensor(values, dtype=dtype, device=flag_blas.device))


def _known_scipy_cublas_complex_divergence(dtype, a, b):
    if not TO_CPU or dtype not in (torch.complex64, torch.complex128):
        return False
    if a == 0.0j and (b == 0.0j or b.imag != 0.0):
        return True
    return dtype == torch.complex64 and (abs(a) > 1e19 or abs(b) > 1e19)


@pytest.mark.rotg
@pytest.mark.parametrize("op_name", ROTG_OPS)
def test_accuracy_rotg_flag_blas_api_symbols(op_name):
    assert callable(getattr(flag_blas.ops, op_name))
    assert callable(getattr(flag_blas, op_name))
    assert getattr(flag_blas.ops, op_name) is getattr(flag_blas, op_name)


@pytest.mark.rotg
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("a_val,b_val", ROTG_REAL_CASES)
def test_accuracy_rotg_real(dtype, a_val, b_val):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    a = torch.tensor([a_val], dtype=dtype, device=flag_blas.device)
    b = torch.tensor([b_val], dtype=dtype, device=flag_blas.device)
    c = torch.zeros(1, dtype=dtype, device=flag_blas.device)
    s = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    ref_a, ref_b, ref_c, ref_s = _rotg_reference(dtype, a_val, b_val)
    if dtype == torch.float32:
        flag_blas.ops.srotg(a, b, c, s)
    else:
        flag_blas.ops.drotg(a, b, c, s)

    expected = _reference_tensor([ref_a, ref_b, ref_c, ref_s], dtype)
    actual = torch.stack((a[0], b[0], c[0], s[0]))
    blas_assert_close(actual, expected, dtype)


@pytest.mark.rotg
def test_accuracy_rotg_flag_blas_api():
    a = torch.tensor([3.0], dtype=torch.float32, device=flag_blas.device)
    b = torch.tensor([4.0], dtype=torch.float32, device=flag_blas.device)
    c = torch.zeros(1, dtype=torch.float32, device=flag_blas.device)
    s = torch.zeros(1, dtype=torch.float32, device=flag_blas.device)

    flag_blas.srotg(a, b, c, s)

    blas_assert_close(a, _reference_tensor([5.0], torch.float32), torch.float32)
    blas_assert_close(b, _reference_tensor([5.0 / 3.0], torch.float32), torch.float32)
    blas_assert_close(c, _reference_tensor([0.6], torch.float32), torch.float32)
    blas_assert_close(s, _reference_tensor([0.8], torch.float32), torch.float32)


@pytest.mark.rotg
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("a_val,b_val", ROTG_COMPLEX_CASES)
def test_accuracy_rotg_complex(dtype, a_val, b_val):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    if _known_scipy_cublas_complex_divergence(dtype, a_val, b_val):
        pytest.xfail("SciPy OpenBLAS complex rotg diverges from cuBLAS semantics")

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    a = torch.tensor([a_val], dtype=dtype, device=flag_blas.device)
    b = torch.tensor([b_val], dtype=dtype, device=flag_blas.device)
    c = torch.zeros(1, dtype=real_dtype, device=flag_blas.device)
    s = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    ref_a, ref_b, ref_c, ref_s = _rotg_reference(dtype, a_val, b_val)
    if dtype == torch.complex64:
        flag_blas.ops.crotg(a, b, c, s)
    else:
        flag_blas.ops.zrotg(a, b, c, s)

    blas_assert_close(a, _reference_tensor([ref_a], dtype), dtype)
    blas_assert_close(b, _reference_tensor([ref_b], dtype), dtype)
    blas_assert_close(c, _reference_tensor([ref_c], real_dtype), real_dtype)
    blas_assert_close(s, _reference_tensor([ref_s], dtype), dtype)


@pytest.mark.rotg
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_rotg_rejects_non_scalar_inputs(dtype):
    if (
        dtype in (torch.float64, torch.complex128)
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("Device does not support float64")

    real_dtype = _rotg_real_dtype(dtype)
    a = torch.ones(2, dtype=dtype, device=flag_blas.device)
    b = torch.ones(1, dtype=dtype, device=flag_blas.device)
    c = torch.zeros(1, dtype=real_dtype, device=flag_blas.device)
    s = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    with pytest.raises(AssertionError, match="a must have exactly one element"):
        _call_rotg(dtype, a, b, c, s)


@pytest.mark.rotg
def test_srotg_rejects_mismatched_input_dtype():
    a = torch.ones(1, dtype=torch.float32, device=flag_blas.device)
    b = torch.ones(1, dtype=torch.float64, device=flag_blas.device)
    c = torch.zeros(1, dtype=torch.float32, device=flag_blas.device)
    s = torch.zeros(1, dtype=torch.float32, device=flag_blas.device)

    with pytest.raises(AssertionError, match="b must be float32"):
        flag_blas.ops.srotg(a, b, c, s)


@pytest.mark.rotg
def test_crotg_rejects_mismatched_c_dtype():
    a = torch.ones(1, dtype=torch.complex64, device=flag_blas.device)
    b = torch.ones(1, dtype=torch.complex64, device=flag_blas.device)
    c = torch.zeros(1, dtype=torch.float64, device=flag_blas.device)
    s = torch.zeros(1, dtype=torch.complex64, device=flag_blas.device)

    with pytest.raises(AssertionError, match="c must be float32"):
        flag_blas.ops.crotg(a, b, c, s)
