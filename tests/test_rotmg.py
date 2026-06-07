import pytest
import torch

import flag_blas

ROTMG_CASES = [
    (-1.0, 2.0, 3.0, 4.0),
    (1.0, 0.0, 2.0, 3.0),
    (1.0, 2.0, 3.0, 0.0),
    (2.0, 1.0, 3.0, 1.0),
    (1.0, -2.0, 3.0, 4.0),
    (1.0, 2.0, 1.0, 1.0),
    (1e-20, 2.0, 3.0, 4.0),
    (1e20, 2.0, 3.0, 4.0),
    (2.0, 1e-20, 3.0, 4.0),
    (2.0, 1e20, 3.0, 4.0),
]
ROTMG_OPS = ["srotmg", "drotmg"]


def _rotmg_reference(d1, d2, x1, y1):
    gam = 4096.0
    gamsq = gam * gam
    rgamsq = 1.0 / gamsq
    flag = -2.0
    h11 = h12 = h21 = h22 = 0.0

    if d1 < 0.0:
        flag = -1.0
        return 0.0, 0.0, 0.0, [flag, h11, h21, h12, h22]

    p2 = d2 * y1
    if p2 == 0.0:
        return d1, d2, x1, [flag, h11, h21, h12, h22]

    p1 = d1 * x1
    q2 = p2 * y1
    q1 = p1 * x1

    if abs(q1) > abs(q2):
        h21 = -y1 / x1
        h12 = p2 / p1
        u = 1.0 - h12 * h21
        if u > 0.0:
            flag = 0.0
            d1 = d1 / u
            d2 = d2 / u
            x1 = x1 * u
        else:
            flag = -1.0
            return 0.0, 0.0, 0.0, [flag, 0.0, 0.0, 0.0, 0.0]
    else:
        if q2 < 0.0:
            flag = -1.0
            return 0.0, 0.0, 0.0, [flag, 0.0, 0.0, 0.0, 0.0]
        flag = 1.0
        h11 = p1 / p2
        h22 = x1 / y1
        u = 1.0 + h11 * h22
        temp = d2 / u
        d2 = d1 / u
        d1 = temp
        x1 = y1 * u

    if d1 != 0.0:
        while d1 <= rgamsq or d1 >= gamsq:
            if flag == 0.0:
                h11 = 1.0
                h22 = 1.0
            else:
                h21 = -1.0
                h12 = 1.0
            flag = -1.0

            if d1 <= rgamsq:
                d1 = d1 * gamsq
                x1 = x1 / gam
                h11 = h11 / gam
                h12 = h12 / gam
            else:
                d1 = d1 / gamsq
                x1 = x1 * gam
                h11 = h11 * gam
                h12 = h12 * gam

    if d2 != 0.0:
        while abs(d2) <= rgamsq or abs(d2) >= gamsq:
            if flag == 0.0:
                h11 = 1.0
                h22 = 1.0
            else:
                h21 = -1.0
                h12 = 1.0
            flag = -1.0

            if abs(d2) <= rgamsq:
                d2 = d2 * gamsq
                h21 = h21 / gam
                h22 = h22 / gam
            else:
                d2 = d2 / gamsq
                h21 = h21 * gam
                h22 = h22 * gam

    if flag < 0.0:
        param = [flag, h11, h21, h12, h22]
    elif flag == 0.0:
        param = [flag, 0.0, h21, h12, 0.0]
    else:
        param = [flag, h11, 0.0, 0.0, h22]
    return d1, d2, x1, param


@pytest.mark.rotmg
@pytest.mark.parametrize("op_name", ROTMG_OPS)
def test_accuracy_rotmg_flag_blas_api_symbols(op_name):
    assert callable(getattr(flag_blas.ops, op_name))
    assert callable(getattr(flag_blas, op_name))
    assert getattr(flag_blas.ops, op_name) is getattr(flag_blas, op_name)


@pytest.mark.rotmg
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("d1_val,d2_val,x1_val,y1_val", ROTMG_CASES)
def test_accuracy_rotmg(dtype, d1_val, d2_val, x1_val, y1_val):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    d1 = torch.tensor([d1_val], dtype=dtype, device=flag_blas.device)
    d2 = torch.tensor([d2_val], dtype=dtype, device=flag_blas.device)
    x1 = torch.tensor([x1_val], dtype=dtype, device=flag_blas.device)
    y1 = torch.tensor([y1_val], dtype=dtype, device=flag_blas.device)
    param = torch.empty(5, dtype=dtype, device=flag_blas.device)

    ref_d1, ref_d2, ref_x1, ref_param = _rotmg_reference(
        float(torch.tensor(d1_val, dtype=dtype).item()),
        float(torch.tensor(d2_val, dtype=dtype).item()),
        float(torch.tensor(x1_val, dtype=dtype).item()),
        float(torch.tensor(y1_val, dtype=dtype).item()),
    )

    if dtype == torch.float32:
        flag_blas.ops.srotmg(d1, d2, x1, y1, param)
        rtol, atol = 1e-4, 1e-4
    else:
        flag_blas.ops.drotmg(d1, d2, x1, y1, param)
        rtol, atol = 1e-12, 1e-12

    torch.testing.assert_close(
        d1.cpu(), torch.tensor([ref_d1], dtype=dtype), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        d2.cpu(), torch.tensor([ref_d2], dtype=dtype), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        x1.cpu(), torch.tensor([ref_x1], dtype=dtype), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        y1.cpu(), torch.tensor([y1_val], dtype=dtype), rtol=0, atol=0
    )
    torch.testing.assert_close(
        param.cpu(), torch.tensor(ref_param, dtype=dtype), rtol=rtol, atol=atol
    )


@pytest.mark.rotmg
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotmg_rejects_bad_shapes(dtype):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    d1 = torch.randn(1, dtype=dtype, device=flag_blas.device)
    d2 = torch.randn(1, dtype=dtype, device=flag_blas.device)
    x1 = torch.randn(2, dtype=dtype, device=flag_blas.device)
    y1 = torch.randn(1, dtype=dtype, device=flag_blas.device)
    param = torch.empty(5, dtype=dtype, device=flag_blas.device)

    with pytest.raises(AssertionError):
        if dtype == torch.float32:
            flag_blas.ops.srotmg(d1, d2, x1, y1, param)
        else:
            flag_blas.ops.drotmg(d1, d2, x1, y1, param)


@pytest.mark.rotmg
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accuracy_rotmg_flag_blas_api(dtype):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    d1 = torch.tensor([1.0], dtype=dtype, device=flag_blas.device)
    d2 = torch.tensor([0.0], dtype=dtype, device=flag_blas.device)
    x1 = torch.tensor([2.0], dtype=dtype, device=flag_blas.device)
    y1 = torch.tensor([3.0], dtype=dtype, device=flag_blas.device)
    param = torch.empty(5, dtype=dtype, device=flag_blas.device)

    if dtype == torch.float32:
        flag_blas.srotmg(d1, d2, x1, y1, param)
    else:
        flag_blas.drotmg(d1, d2, x1, y1, param)

    expected_param = torch.tensor([-2.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype)
    tol = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(
        d1.cpu(), torch.tensor([1.0], dtype=dtype), rtol=tol, atol=tol
    )
    torch.testing.assert_close(
        d2.cpu(), torch.tensor([0.0], dtype=dtype), rtol=tol, atol=tol
    )
    torch.testing.assert_close(
        x1.cpu(), torch.tensor([2.0], dtype=dtype), rtol=tol, atol=tol
    )
    torch.testing.assert_close(param.cpu(), expected_param, rtol=tol, atol=tol)


@pytest.mark.rotmg
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotmg_rejects_bad_param_shape(dtype):
    if dtype == torch.float64 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    d1 = torch.randn(1, dtype=dtype, device=flag_blas.device)
    d2 = torch.randn(1, dtype=dtype, device=flag_blas.device)
    x1 = torch.randn(1, dtype=dtype, device=flag_blas.device)
    y1 = torch.randn(1, dtype=dtype, device=flag_blas.device)
    param = torch.empty(4, dtype=dtype, device=flag_blas.device)

    with pytest.raises(AssertionError, match="param must have exactly five elements"):
        if dtype == torch.float32:
            flag_blas.ops.srotmg(d1, d2, x1, y1, param)
        else:
            flag_blas.ops.drotmg(d1, d2, x1, y1, param)


@pytest.mark.rotmg
def test_srotmg_rejects_mismatched_param_dtype():
    d1 = torch.randn(1, dtype=torch.float32, device=flag_blas.device)
    d2 = torch.randn(1, dtype=torch.float32, device=flag_blas.device)
    x1 = torch.randn(1, dtype=torch.float32, device=flag_blas.device)
    y1 = torch.randn(1, dtype=torch.float32, device=flag_blas.device)
    param = torch.empty(5, dtype=torch.float64, device=flag_blas.device)

    with pytest.raises(AssertionError, match="param must be float32"):
        flag_blas.ops.srotmg(d1, d2, x1, y1, param)


@pytest.mark.rotmg
def test_srotmg_rejects_non_scalar_d1():
    d1 = torch.randn(2, dtype=torch.float32, device=flag_blas.device)
    d2 = torch.randn(1, dtype=torch.float32, device=flag_blas.device)
    x1 = torch.randn(1, dtype=torch.float32, device=flag_blas.device)
    y1 = torch.randn(1, dtype=torch.float32, device=flag_blas.device)
    param = torch.empty(5, dtype=torch.float32, device=flag_blas.device)

    with pytest.raises(AssertionError, match="d1 must have exactly one element"):
        flag_blas.ops.srotmg(d1, d2, x1, y1, param)
