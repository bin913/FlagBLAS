import ctypes
import ctypes.util

import numpy as np
import pytest
import torch
from scipy.linalg import blas as cpu_blas

import flag_blas
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER

from .accuracy_utils import blas_assert_close
from .conftest import TO_CPU

HER_SIZES = [
    64,
    96,
    127,
    128,
    129,
    192,
    255,
    256,
    257,
    384,
    511,
    512,
    513,
    768,
    1023,
    1024,
    1025,
    1536,
    2048,
    3072,
    4096,
]
HER_UPLOS = [CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER]
CUBLAS_POINTER_MODE_HOST = 0
_cublas = None
_cublas_handle = None
_CUBLAS_HER_FUNCS = None


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


def _configure_cublas_signatures():
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    for name in ("cublasCher_v2", "cublasZher_v2"):
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
        ]
        func.restype = ctypes.c_int


def _ensure_cublas():
    global _cublas, _CUBLAS_HER_FUNCS
    if _cublas is None:
        _cublas = load_cublas()
        _configure_cublas_signatures()
        _CUBLAS_HER_FUNCS = {
            "cher": (_cublas.cublasCher_v2, ctypes.c_float),
            "zher": (_cublas.cublasZher_v2, ctypes.c_double),
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


def _make_her_inputs(dtype, n, incx, seed):
    torch.manual_seed(seed)
    real_dtype = torch.float32 if dtype is torch.complex64 else torch.float64
    device = flag_blas.device
    x_len = 1 + (n - 1) * incx
    x = torch.randn(x_len, dtype=dtype, device=device)
    A = torch.randn((n, n), dtype=dtype, device=device)
    A = torch.tril(A) + torch.tril(A, -1).mH
    diag = A.diagonal()
    diag.copy_(diag.real.to(dtype))
    alpha = torch.tensor(0.75, dtype=real_dtype, device=device)
    return alpha, x, A.contiguous()


def _cublas_her_reference(name, uplo, n, alpha, x, incx, A, lda):
    handle = _get_cublas_handle()
    _ensure_cublas()
    func, ctor = _CUBLAS_HER_FUNCS[name]
    alpha_c = ctor(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    status = func(
        handle,
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
    )
    if status != 0:
        raise RuntimeError(f"cublasXher_v2 execution failed with error code: {status}")
    torch.cuda.synchronize(A.device)
    return A


def _scipy_her(name, uplo, n, alpha, x, incx, A):
    lower = int(uplo == CUBLAS_FILL_MODE_LOWER)
    x_cpu = x.detach().cpu().contiguous()
    ref = A.detach().cpu().contiguous()
    logical_A = ref[:n, :n].T.numpy().copy(order="F")
    x_view = x_cpu.numpy()[::incx][:n]
    if name == "cher":
        out = cpu_blas.cher(
            float(alpha.cpu()), x_view, a=logical_A, lower=lower, overwrite_a=1
        )
    else:
        out = cpu_blas.zher(
            float(alpha.cpu()), x_view, a=logical_A, lower=lower, overwrite_a=1
        )
    ref[:n, :n] = torch.from_numpy(np.array(out.T, copy=True))
    return ref


def _reference(name, uplo, n, alpha, x, incx, A, lda):
    ref = A.clone()
    if TO_CPU:
        return _scipy_her(name, uplo, n, alpha, x, incx, ref)
    return _cublas_her_reference(name, uplo, n, alpha, x, incx, ref, lda)


def _run_her(name, dtype, uplo, n, incx, seed):
    alpha, x, A = _make_her_inputs(dtype, n, incx, seed)
    ref = _reference(name, uplo, n, alpha, x, incx, A, n)
    if name == "cher":
        flag_blas.ops.cher(uplo, n, alpha, x, incx, A, n)
    else:
        flag_blas.ops.zher(uplo, n, alpha, x, incx, A, n)
    blas_assert_close(A, ref, dtype, reduce_dim=n)


@pytest.mark.parametrize("n", HER_SIZES)
@pytest.mark.parametrize("uplo", HER_UPLOS)
@pytest.mark.cher
def test_accuracy_cher_sizes(uplo, n):
    _run_her("cher", torch.complex64, uplo, n, 1, 0)


@pytest.mark.parametrize("incx,uplo,n", [(2, 1, 64), (3, 0, 128)])
@pytest.mark.cher
def test_accuracy_cher_stride(incx, uplo, n):
    _run_her("cher", torch.complex64, uplo, n, incx, 0)


@pytest.mark.parametrize("n", HER_SIZES)
@pytest.mark.parametrize("uplo", HER_UPLOS)
@pytest.mark.zher
def test_accuracy_zher_sizes(uplo, n):
    _run_her("zher", torch.complex128, uplo, n, 1, 0)


@pytest.mark.parametrize("incx,uplo,n", [(2, 1, 64), (3, 0, 128)])
@pytest.mark.zher
def test_accuracy_zher_stride(incx, uplo, n):
    _run_her("zher", torch.complex128, uplo, n, incx, 0)


@pytest.mark.parametrize(
    "name,dtype,alpha_dtype",
    [
        ("cher", torch.complex64, torch.float32),
        ("zher", torch.complex128, torch.float64),
    ],
)
def test_her_padded_lda(name, dtype, alpha_dtype):
    n, lda = 7, 11
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)
    A = torch.randn((n, lda), dtype=dtype, device=flag_blas.device)
    alpha = torch.tensor(0.75, dtype=alpha_dtype, device=flag_blas.device)
    ref = _reference(
        name,
        CUBLAS_FILL_MODE_LOWER,
        n,
        alpha,
        x,
        1,
        A.clone(),
        lda,
    )

    getattr(flag_blas, name)(CUBLAS_FILL_MODE_LOWER, n, alpha, x, 1, A, lda)

    blas_assert_close(A, ref, dtype, reduce_dim=n)


def test_her_n_zero_is_noop():
    A = torch.empty((0, 1), dtype=torch.complex64, device=flag_blas.device)
    x = torch.empty(0, dtype=torch.complex64, device=flag_blas.device)

    result = flag_blas.cher(CUBLAS_FILL_MODE_LOWER, 0, 0.75, x, 1, A, 1)

    assert result is A


def test_zher_preserves_double_scalar_precision():
    n = 31
    alpha = torch.tensor(
        0.12345678901234568, dtype=torch.float64, device=flag_blas.device
    )
    x = torch.randn(n, dtype=torch.complex128, device=flag_blas.device)
    A = torch.randn((n, n), dtype=torch.complex128, device=flag_blas.device)
    ref = _reference(
        "zher",
        CUBLAS_FILL_MODE_UPPER,
        n,
        alpha,
        x,
        1,
        A.clone(),
        n,
    ).to(A.device)

    flag_blas.zher(CUBLAS_FILL_MODE_UPPER, n, alpha, x, 1, A, n)

    torch.testing.assert_close(A, ref, rtol=2e-13, atol=2e-13)


def test_her_rejects_noncontiguous_matrix():
    base = torch.randn((8, 8), dtype=torch.complex64, device=flag_blas.device)
    A = base.T
    x = torch.randn(8, dtype=torch.complex64, device=flag_blas.device)

    with pytest.raises(AssertionError):
        flag_blas.cher(CUBLAS_FILL_MODE_LOWER, 8, 0.75, x, 1, A, 8)


HER_BALANCED_SIZES = (1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 64, 127)
HER_VARIANTS = [
    pytest.param("cher", torch.complex64, torch.float32, 0.75, id="cher"),
    pytest.param(
        "zher",
        torch.complex128,
        torch.float64,
        0.12345678901234568,
        id="zher",
    ),
]


@pytest.mark.parametrize("name,dtype,alpha_dtype,alpha_value", HER_VARIANTS)
@pytest.mark.parametrize("uplo", HER_UPLOS)
@pytest.mark.parametrize("n", HER_BALANCED_SIZES)
@pytest.mark.parametrize("incx", [1, 2, 3])
@pytest.mark.parametrize("lda_pad", [0, 3])
def test_accuracy_her_balanced(
    name, dtype, alpha_dtype, alpha_value, uplo, n, incx, lda_pad
):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("fp64 is not supported on this device")
    lda = n + lda_pad
    x_len = 1 + (n - 1) * incx
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    x_before = x.clone()
    A = torch.randn((n, lda), dtype=dtype, device=flag_blas.device)
    alpha = torch.tensor(alpha_value, dtype=alpha_dtype, device=flag_blas.device)
    ref = _reference(name, uplo, n, alpha, x, incx, A.clone(), lda)

    getattr(flag_blas, name)(uplo, n, alpha, x, incx, A, lda)

    blas_assert_close(A, ref, dtype, reduce_dim=n)
    torch.testing.assert_close(x, x_before)
