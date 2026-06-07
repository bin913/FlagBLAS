import pytest
import torch

import flag_blas
from .accuracy_utils import L1_PAIR_STRIDES, L1_STRIDE_SHAPES

ROTM_FLAGS = [-2.0, -1.5, -1.0, -0.5, 0.0, 1.0]
ROTM_OPS = ["srotm", "drotm"]


def _build_param(dtype, device, flag):
    return torch.tensor([flag, 0.75, -0.5, 0.25, 1.5], dtype=dtype, device=device)


def _rotm_reference(n, x, incx, y, incy, param):
    if n <= 0:
        return

    flag = float(param[0].item())
    h11, h21, h12, h22 = param[1], param[2], param[3], param[4]
    x_view = x[0 : n * incx : incx]
    y_view = y[0 : n * incy : incy]
    old_x = x_view.clone()
    old_y = y_view.clone()

    if flag == -2.0:
        return
    if flag < 0.0:
        x_view.copy_(h11 * old_x + h12 * old_y)
        y_view.copy_(h21 * old_x + h22 * old_y)
        return
    if flag == 0.0:
        x_view.copy_(old_x + h12 * old_y)
        y_view.copy_(h21 * old_x + old_y)
        return
    if flag == 1.0:
        x_view.copy_(h11 * old_x + old_y)
        y_view.copy_(-old_x + h22 * old_y)
        return
    raise ValueError(f"Invalid rotm flag: {flag}")


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
    ref_x = x.cpu()
    ref_y = y.cpu()
    _rotm_reference(n, ref_x, incx, ref_y, incy, param.cpu())

    _call_rotm(dtype, n, x, incx, y, incy, param)

    tol = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(x.cpu(), ref_x, rtol=tol, atol=tol)
    torch.testing.assert_close(y.cpu(), ref_y, rtol=tol, atol=tol)


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
    ref_x = x.cpu()
    ref_y = y.cpu()
    _rotm_reference(n, ref_x, 1, ref_y, 1, param.cpu())

    _call_rotm(dtype, n, x, 1, y, 1, param)

    tol = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(x.cpu(), ref_x, rtol=tol, atol=tol)
    torch.testing.assert_close(y.cpu(), ref_y, rtol=tol, atol=tol)


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

    expected_x = torch.tensor([1.75, 2.75, 3.75], dtype=dtype)
    expected_y = torch.tensor([5.5, 6.5, 7.5], dtype=dtype)
    tol = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(x.cpu(), expected_x, rtol=tol, atol=tol)
    torch.testing.assert_close(y.cpu(), expected_y, rtol=tol, atol=tol)


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
