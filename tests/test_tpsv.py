import ctypes
import ctypes.util

import numpy as np
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

pytestmark = pytest.mark.tpsv


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


def _cublas_tpsv(fn, fn_name, uplo, trans, diag, n, AP, x, incx):
    handle = ctypes.c_void_p()
    status = _cublas.cublasCreate_v2(ctypes.byref(handle))
    if status != 0:
        raise RuntimeError(f"cublasCreate_v2 failed with status code: {status}")
    try:
        status = fn(
            handle,
            ctypes.c_int(uplo),
            ctypes.c_int(trans),
            ctypes.c_int(diag),
            ctypes.c_int(n),
            ctypes.c_void_p(AP.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_int(incx),
        )
        if status != 0:
            raise RuntimeError(f"{fn_name} failed with status code: {status}")
    finally:
        _cublas.cublasDestroy_v2(handle)
    return x


def cublas_stpsv_reference(uplo, trans, diag, n, AP, x, incx):
    return _cublas_tpsv(
        _cublas.cublasStpsv_v2,
        "cublasStpsv_v2",
        uplo,
        trans,
        diag,
        n,
        AP,
        x,
        incx,
    )


def cublas_dtpsv_reference(uplo, trans, diag, n, AP, x, incx):
    return _cublas_tpsv(
        _cublas.cublasDtpsv_v2,
        "cublasDtpsv_v2",
        uplo,
        trans,
        diag,
        n,
        AP,
        x,
        incx,
    )


def cublas_ctpsv_reference(uplo, trans, diag, n, AP, x, incx):
    return _cublas_tpsv(
        _cublas.cublasCtpsv_v2,
        "cublasCtpsv_v2",
        uplo,
        trans,
        diag,
        n,
        AP,
        x,
        incx,
    )


def cublas_ztpsv_reference(uplo, trans, diag, n, AP, x, incx):
    return _cublas_tpsv(
        _cublas.cublasZtpsv_v2,
        "cublasZtpsv_v2",
        uplo,
        trans,
        diag,
        n,
        AP,
        x,
        incx,
    )


def _pack_triangular(A, uplo):
    n = A.shape[0]
    vals = []
    for j in range(n):
        if uplo == CUBLAS_FILL_MODE_UPPER:
            for i in range(j + 1):
                vals.append(A[i, j])
        else:
            for i in range(j, n):
                vals.append(A[i, j])
    return torch.stack(vals).contiguous()


def _make_case(n, dtype, uplo, diag, incx, device):
    torch.manual_seed(n + 17 * int(uplo) + 31 * int(diag) + 43 * int(incx))
    if dtype.is_complex:
        A = (
            torch.randn((n, n), dtype=dtype, device=device) * 0.05
            + 1j * torch.randn((n, n), dtype=dtype, device=device) * 0.05
        )
        b = torch.randn(
            (1 + (n - 1) * incx,), dtype=dtype, device=device
        ) + 1j * torch.randn((1 + (n - 1) * incx,), dtype=dtype, device=device)
    else:
        A = torch.randn((n, n), dtype=dtype, device=device) * 0.05
        b = torch.randn((1 + (n - 1) * incx,), dtype=dtype, device=device)
    diag_idx = torch.arange(n, device=device)
    A[diag_idx, diag_idx] = 1.0 if diag == CUBLAS_DIAG_UNIT else 2.0
    if uplo == CUBLAS_FILL_MODE_UPPER:
        A = torch.triu(A)
    else:
        A = torch.tril(A)
    return _pack_triangular(A, uplo), b.contiguous()


def _scipy_ref(name, n, AP, x, incx, uplo, trans, diag):
    fn = getattr(cpu_blas, name)
    lower = int(uplo == CUBLAS_FILL_MODE_LOWER)
    trans_arg = 0 if trans == CUBLAS_OP_N else 1 if trans == CUBLAS_OP_T else 2
    diag_arg = int(diag == CUBLAS_DIAG_UNIT)
    AP_cpu = to_cpu_blas_tensor(AP).numpy()
    x_cpu = to_cpu_blas_tensor(x).numpy()
    return torch.from_numpy(
        fn(
            n,
            AP_cpu,
            x_cpu,
            incx=incx,
            lower=lower,
            trans=trans_arg,
            diag=diag_arg,
            overwrite_x=1,
        )
    )


def scipy_stpsv_reference(n, AP, x, incx, uplo, trans, diag):
    return _scipy_ref("stpsv", n, AP, x, incx, uplo, trans, diag)


def scipy_dtpsv_reference(n, AP, x, incx, uplo, trans, diag):
    return _scipy_ref("dtpsv", n, AP, x, incx, uplo, trans, diag)


def scipy_ctpsv_reference(n, AP, x, incx, uplo, trans, diag):
    return _scipy_ref("ctpsv", n, AP, x, incx, uplo, trans, diag)


def scipy_ztpsv_reference(n, AP, x, incx, uplo, trans, diag):
    return _scipy_ref("ztpsv", n, AP, x, incx, uplo, trans, diag)


def _run(op, cpu_ref, gpu_ref, dtype, uplo, trans, diag, n, incx=1):
    if (
        dtype in (torch.float64, torch.complex128)
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("fp64 is not supported on this device")
    device = flag_blas.device
    AP, x = _make_case(n, dtype, uplo, diag, incx, device)
    y = x.clone()
    if TO_CPU:
        ref = cpu_ref(n, AP, x, incx, uplo, trans, diag)
    else:
        ref = x.clone()
        gpu_ref(uplo, trans, diag, n, AP, ref, incx)
    op(uplo, trans, diag, n, AP, y, incx)
    blas_assert_close(y, ref, dtype, reduce_dim=n)


TPSV_SIZES = (1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129)
TPSV_STRIDE_SIZES = (3, 17, 33, 127)

REAL_CASES = [
    pytest.param(uplo, trans, diag, n, id=f"{uplo}-{trans}-{diag}-{n}")
    for uplo in (CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER)
    for trans in (CUBLAS_OP_N, CUBLAS_OP_T)
    for diag in (CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT)
    for n in TPSV_SIZES
]
COMPLEX_CASES = [
    pytest.param(uplo, trans, diag, n, id=f"{uplo}-{trans}-{diag}-{n}")
    for uplo in (CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER)
    for trans in (CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C)
    for diag in (CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT)
    for n in TPSV_SIZES
]
REAL_STRIDE_CASES = [
    pytest.param(incx, uplo, trans, diag, n, id=f"{incx}-{uplo}-{trans}-{diag}-{n}")
    for incx in (2, 3)
    for uplo in (CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER)
    for trans in (CUBLAS_OP_N, CUBLAS_OP_T)
    for diag in (CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT)
    for n in TPSV_STRIDE_SIZES
]
COMPLEX_STRIDE_CASES = [
    pytest.param(incx, uplo, trans, diag, n, id=f"{incx}-{uplo}-{trans}-{diag}-{n}")
    for incx in (2, 3)
    for uplo in (CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER)
    for trans in (CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C)
    for diag in (CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT)
    for n in TPSV_STRIDE_SIZES
]


@pytest.mark.parametrize("uplo,trans,diag,n", REAL_CASES)
@pytest.mark.stpsv
def test_accuracy_stpsv(uplo, trans, diag, n):
    _run(
        flag_blas.ops.stpsv,
        scipy_stpsv_reference,
        cublas_stpsv_reference,
        torch.float32,
        uplo,
        trans,
        diag,
        n,
    )


@pytest.mark.parametrize("incx,uplo,trans,diag,n", REAL_STRIDE_CASES)
@pytest.mark.stpsv
def test_accuracy_stpsv_stride(incx, uplo, trans, diag, n):
    _run(
        flag_blas.ops.stpsv,
        scipy_stpsv_reference,
        cublas_stpsv_reference,
        torch.float32,
        uplo,
        trans,
        diag,
        n,
        incx,
    )


@pytest.mark.parametrize("uplo,trans,diag,n", REAL_CASES)
@pytest.mark.dtpsv
def test_accuracy_dtpsv(uplo, trans, diag, n):
    _run(
        flag_blas.ops.dtpsv,
        scipy_dtpsv_reference,
        cublas_dtpsv_reference,
        torch.float64,
        uplo,
        trans,
        diag,
        n,
    )


@pytest.mark.parametrize("incx,uplo,trans,diag,n", REAL_STRIDE_CASES)
@pytest.mark.dtpsv
def test_accuracy_dtpsv_stride(incx, uplo, trans, diag, n):
    _run(
        flag_blas.ops.dtpsv,
        scipy_dtpsv_reference,
        cublas_dtpsv_reference,
        torch.float64,
        uplo,
        trans,
        diag,
        n,
        incx,
    )


@pytest.mark.parametrize("uplo,trans,diag,n", COMPLEX_CASES)
@pytest.mark.ctpsv
def test_accuracy_ctpsv(uplo, trans, diag, n):
    _run(
        flag_blas.ops.ctpsv,
        scipy_ctpsv_reference,
        cublas_ctpsv_reference,
        torch.complex64,
        uplo,
        trans,
        diag,
        n,
    )


@pytest.mark.parametrize("incx,uplo,trans,diag,n", COMPLEX_STRIDE_CASES)
@pytest.mark.ctpsv
def test_accuracy_ctpsv_stride(incx, uplo, trans, diag, n):
    _run(
        flag_blas.ops.ctpsv,
        scipy_ctpsv_reference,
        cublas_ctpsv_reference,
        torch.complex64,
        uplo,
        trans,
        diag,
        n,
        incx,
    )


@pytest.mark.parametrize("uplo,trans,diag,n", COMPLEX_CASES)
@pytest.mark.ztpsv
def test_accuracy_ztpsv(uplo, trans, diag, n):
    _run(
        flag_blas.ops.ztpsv,
        scipy_ztpsv_reference,
        cublas_ztpsv_reference,
        torch.complex128,
        uplo,
        trans,
        diag,
        n,
    )


@pytest.mark.parametrize("incx,uplo,trans,diag,n", COMPLEX_STRIDE_CASES)
@pytest.mark.ztpsv
def test_accuracy_ztpsv_stride(incx, uplo, trans, diag, n):
    _run(
        flag_blas.ops.ztpsv,
        scipy_ztpsv_reference,
        cublas_ztpsv_reference,
        torch.complex128,
        uplo,
        trans,
        diag,
        n,
        incx,
    )


TPSV_VARIANTS = [
    pytest.param(flag_blas.stpsv, torch.float32, id="stpsv"),
    pytest.param(flag_blas.dtpsv, torch.float64, id="dtpsv"),
    pytest.param(flag_blas.ctpsv, torch.complex64, id="ctpsv"),
    pytest.param(flag_blas.ztpsv, torch.complex128, id="ztpsv"),
]


@pytest.mark.parametrize("op,dtype", TPSV_VARIANTS)
def test_tpsv_n_zero_is_noop(op, dtype):
    AP = torch.empty(0, dtype=dtype, device=flag_blas.device)
    x = torch.empty(0, dtype=dtype, device=flag_blas.device)

    result = op(
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
        0,
        AP,
        x,
        1,
    )

    assert result is x


@pytest.mark.parametrize("op,dtype", TPSV_VARIANTS)
@pytest.mark.parametrize("uplo", [CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER])
def test_tpsv_unit_diag_ignores_stored_diagonal(op, dtype, uplo):
    n = 9
    AP, x = _make_case(n, dtype, uplo, CUBLAS_DIAG_UNIT, 1, flag_blas.device)
    dirty = AP.clone()
    for j in range(n):
        if uplo == CUBLAS_FILL_MODE_UPPER:
            offset = j * (j + 1) // 2 + j
        else:
            offset = j * n - j * (j - 1) // 2
        dirty[offset] = (
            complex(float("nan"), float("nan")) if dtype.is_complex else float("nan")
        )
    clean_x = x.clone()
    dirty_x = x.clone()

    op(uplo, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, AP, clean_x, 1)
    op(uplo, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, dirty, dirty_x, 1)

    torch.testing.assert_close(dirty_x, clean_x)


@pytest.mark.parametrize("op,dtype", TPSV_VARIANTS)
def test_tpsv_rejects_noncontiguous_packed_storage(op, dtype):
    n = 8
    packed_len = n * (n + 1) // 2
    AP = torch.randn(2 * packed_len, dtype=dtype, device=flag_blas.device)[::2]
    x = torch.randn(n, dtype=dtype, device=flag_blas.device)

    with pytest.raises(AssertionError):
        op(
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT,
            n,
            AP,
            x,
            1,
        )
