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

pytestmark = pytest.mark.syr

SYR_SIZES = [64, 128, 256]
SYR_UPLOS = [CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER]
SYR_STRIDES = [(2, CUBLAS_FILL_MODE_UPPER, 64), (3, CUBLAS_FILL_MODE_LOWER, 128)]


class _ComplexFloat(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]


class _ComplexDouble(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


def _load_cublas():
    names = ["libcublas.so.13"]
    found = ctypes.util.find_library("cublas")
    if found:
        names.append(found)
    names.extend(["libcublas.so", "libcublas.so.12", "libcublas.so.11"])
    for name in names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError("Unable to find libcublas.so")


_cublas = _load_cublas()
_cublas_handle = None


def _get_cublas_handle():
    global _cublas_handle
    if _cublas_handle is None:
        _cublas_handle = ctypes.c_void_p()
        status = _cublas.cublasCreate_v2(ctypes.byref(_cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status code: {status}")
        status = _cublas.cublasSetPointerMode_v2(_cublas_handle, 0)
        if status != 0:
            raise RuntimeError(
                f"cublasSetPointerMode_v2 failed with status code: {status}"
            )
    return _cublas_handle


def _make_inputs(n, incx, dtype, device):
    lda = n
    x_len = 1 + (n - 1) * incx
    if dtype.is_complex:
        x = torch.randn(x_len, dtype=dtype, device=device) + 1j * torch.randn(
            x_len, dtype=dtype, device=device
        )
        A = torch.randn((lda, n), dtype=dtype, device=device) + 1j * torch.randn(
            (lda, n), dtype=dtype, device=device
        )
    else:
        x = torch.randn(x_len, dtype=dtype, device=device)
        A = torch.randn((lda, n), dtype=dtype, device=device)
    return x, A


def _cpu_ref(name, uplo, n, alpha, x, incx, A, lda):
    x_cpu = x.detach().cpu().contiguous()
    ref = A.detach().cpu().contiguous()
    logical_A = ref[:n, :n].T.numpy().copy(order="F")
    lower = int(uplo == CUBLAS_FILL_MODE_LOWER)
    if name == "ssyr":
        updated = cpu_blas.ssyr(
            alpha, x_cpu.numpy(), a=logical_A, lower=lower, incx=incx, overwrite_a=1
        )
    elif name == "dsyr":
        updated = cpu_blas.dsyr(
            alpha, x_cpu.numpy(), a=logical_A, lower=lower, incx=incx, overwrite_a=1
        )
    elif name == "csyr":
        updated = cpu_blas.csyr(
            alpha, x_cpu.numpy(), a=logical_A, lower=lower, incx=incx, overwrite_a=1
        )
    else:
        updated = cpu_blas.zsyr(
            alpha, x_cpu.numpy(), a=logical_A, lower=lower, incx=incx, overwrite_a=1
        )
    ref[:n, :n] = torch.from_numpy(updated.T.copy())
    return ref


def _cublas_ref(name, uplo, n, alpha, x, incx, A, lda):
    ref = A.clone()
    if name == "ssyr":
        func, scalar = _cublas.cublasSsyr_v2, ctypes.c_float(alpha)
    elif name == "dsyr":
        func, scalar = _cublas.cublasDsyr_v2, ctypes.c_double(alpha)
    elif name == "csyr":
        value = complex(alpha)
        func, scalar = _cublas.cublasCsyr_v2, _ComplexFloat(value.real, value.imag)
    else:
        value = complex(alpha)
        func, scalar = _cublas.cublasZsyr_v2, _ComplexDouble(value.real, value.imag)
    status = func(
        _get_cublas_handle(),
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(scalar),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.c_void_p(ref.data_ptr()),
        ctypes.c_int(lda),
    )
    if status != 0:
        raise RuntimeError(f"cublasXsyr_v2 failed with status code: {status}")
    torch.cuda.synchronize(ref.device)
    return ref


def _run_syr(name, dtype, alpha, uplo, n, incx):
    device = flag_blas.device
    x, A = _make_inputs(n, incx, dtype, device)
    lda = n
    if TO_CPU:
        ref = _cpu_ref(name, uplo, n, alpha, x, incx, A, lda)
    else:
        ref = _cublas_ref(name, uplo, n, alpha, x, incx, A, lda)
    if name == "ssyr":
        flag_blas.ops.ssyr(uplo, n, alpha, x, incx, A, lda)
    elif name == "dsyr":
        flag_blas.ops.dsyr(uplo, n, alpha, x, incx, A, lda)
    elif name == "csyr":
        flag_blas.ops.csyr(uplo, n, alpha, x, incx, A, lda)
    else:
        flag_blas.ops.zsyr(uplo, n, alpha, x, incx, A, lda)
    blas_assert_close(A, ref, dtype, reduce_dim=n)


@pytest.mark.parametrize("n", SYR_SIZES)
@pytest.mark.parametrize("uplo", SYR_UPLOS)
@pytest.mark.ssyr
def test_accuracy_ssyr_sizes(uplo, n):
    _run_syr("ssyr", torch.float32, 0.75, uplo, n, 1)


@pytest.mark.parametrize("incx,uplo,n", SYR_STRIDES)
@pytest.mark.ssyr
def test_accuracy_ssyr_stride(incx, uplo, n):
    _run_syr("ssyr", torch.float32, 0.75, uplo, n, incx)


@pytest.mark.parametrize("n", SYR_SIZES)
@pytest.mark.parametrize("uplo", SYR_UPLOS)
@pytest.mark.dsyr
def test_accuracy_dsyr_sizes(uplo, n):
    _run_syr("dsyr", torch.float64, 0.75, uplo, n, 1)


@pytest.mark.parametrize("incx,uplo,n", SYR_STRIDES)
@pytest.mark.dsyr
def test_accuracy_dsyr_stride(incx, uplo, n):
    _run_syr("dsyr", torch.float64, 0.75, uplo, n, incx)


@pytest.mark.parametrize("n", SYR_SIZES)
@pytest.mark.parametrize("uplo", SYR_UPLOS)
@pytest.mark.csyr
def test_accuracy_csyr_sizes(uplo, n):
    _run_syr("csyr", torch.complex64, np.complex64(0.5 + 0.25j), uplo, n, 1)


@pytest.mark.parametrize("incx,uplo,n", SYR_STRIDES)
@pytest.mark.csyr
def test_accuracy_csyr_stride(incx, uplo, n):
    _run_syr("csyr", torch.complex64, np.complex64(0.5 + 0.25j), uplo, n, incx)


@pytest.mark.parametrize("n", SYR_SIZES)
@pytest.mark.parametrize("uplo", SYR_UPLOS)
@pytest.mark.zsyr
def test_accuracy_zsyr_sizes(uplo, n):
    _run_syr("zsyr", torch.complex128, np.complex128(0.5 + 0.25j), uplo, n, 1)


@pytest.mark.parametrize("incx,uplo,n", SYR_STRIDES)
@pytest.mark.zsyr
def test_accuracy_zsyr_stride(incx, uplo, n):
    _run_syr("zsyr", torch.complex128, np.complex128(0.5 + 0.25j), uplo, n, incx)


def _make_regression_inputs(n, incx, dtype, lda):
    x_len = 1 + (n - 1) * incx if n > 0 else 0
    x = torch.randn(x_len, dtype=dtype, device=flag_blas.device)
    A = torch.randn((n, lda), dtype=dtype, device=flag_blas.device)
    return x, A.contiguous()


def _regression_reference(name, uplo, n, alpha, x, incx, A, lda):
    if TO_CPU:
        return _cpu_ref(name, uplo, n, alpha, x, incx, A, lda)
    return _cublas_ref(name, uplo, n, alpha, x, incx, A, lda)


@pytest.mark.parametrize(
    "name,dtype", [("ssyr", torch.float32), ("dsyr", torch.float64)]
)
def test_syr_accepts_scalar_cuda_tensor_alpha(name, dtype):
    n, lda = 7, 10
    x, A = _make_regression_inputs(n, 1, dtype, lda)
    alpha = torch.tensor(0.75, dtype=dtype, device=flag_blas.device)
    ref = _regression_reference(
        name,
        CUBLAS_FILL_MODE_LOWER,
        n,
        alpha.item(),
        x,
        1,
        A.clone(),
        lda,
    )

    getattr(flag_blas, name)(CUBLAS_FILL_MODE_LOWER, n, alpha, x, 1, A, lda)

    blas_assert_close(A, ref, dtype, reduce_dim=n)


@pytest.mark.parametrize(
    "name,dtype,alpha",
    [
        ("dsyr", torch.float64, 0.12345678901234568),
        (
            "zsyr",
            torch.complex128,
            complex(0.12345678901234568, -0.2345678912345679),
        ),
    ],
)
def test_syr_preserves_double_scalar_precision(name, dtype, alpha):
    n = 31
    x, A = _make_regression_inputs(n, 1, dtype, n)
    ref = _regression_reference(
        name, CUBLAS_FILL_MODE_UPPER, n, alpha, x, 1, A.clone(), n
    ).to(A.device)

    getattr(flag_blas, name)(CUBLAS_FILL_MODE_UPPER, n, alpha, x, 1, A, n)

    torch.testing.assert_close(A, ref, rtol=2e-13, atol=2e-13)


def test_syr_n_zero_is_noop():
    x = torch.empty(0, dtype=torch.float32, device=flag_blas.device)
    A = torch.empty((0, 1), dtype=torch.float32, device=flag_blas.device)

    result = flag_blas.ssyr(CUBLAS_FILL_MODE_LOWER, 0, 0.75, x, 1, A, 1)

    assert result is A


def test_syr_rejects_noncontiguous_matrix():
    n = 8
    x = torch.randn(n, device=flag_blas.device)
    A = torch.randn((n, 2 * n), device=flag_blas.device)[:, ::2]

    with pytest.raises(AssertionError):
        flag_blas.ssyr(CUBLAS_FILL_MODE_LOWER, n, 0.75, x, 1, A, n)


SYR_BALANCED_SIZES = (1, 2, 7, 16, 17, 33, 127)
SYR_VARIANTS = [
    pytest.param("ssyr", torch.float32, 0.75, id="ssyr"),
    pytest.param("dsyr", torch.float64, 0.12345678901234568, id="dsyr"),
    pytest.param("csyr", torch.complex64, complex(0.5, -0.25), id="csyr"),
    pytest.param(
        "zsyr",
        torch.complex128,
        complex(0.12345678901234568, -0.2345678912345679),
        id="zsyr",
    ),
]


@pytest.mark.parametrize("name,dtype,alpha", SYR_VARIANTS)
@pytest.mark.parametrize("uplo", SYR_UPLOS)
@pytest.mark.parametrize("n", SYR_BALANCED_SIZES)
@pytest.mark.parametrize("incx", [1, 2, 3])
@pytest.mark.parametrize("lda_pad", [0, 3])
def test_accuracy_syr_balanced(name, dtype, alpha, uplo, n, incx, lda_pad):
    if (
        dtype in (torch.float64, torch.complex128)
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("fp64 is not supported on this device")
    lda = n + lda_pad
    x, A = _make_regression_inputs(n, incx, dtype, lda)
    x_before = x.clone()
    ref = _regression_reference(name, uplo, n, alpha, x, incx, A.clone(), lda)

    getattr(flag_blas, name)(uplo, n, alpha, x, incx, A, lda)

    blas_assert_close(A, ref, dtype, reduce_dim=n)
    torch.testing.assert_close(x, x_before)
