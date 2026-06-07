import pytest
import torch

import flag_blas


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
    return (
        torch.float32
        if dtype in (torch.float32, torch.complex64)
        else torch.float64
    )


def _real_rotg_reference(a, b):
    abs_a = abs(a)
    abs_b = abs(b)
    scale = abs_a + abs_b
    if scale == 0.0:
        return 0.0, 0.0, 1.0, 0.0

    roe = a if abs_a > abs_b else b
    r = scale * ((a / scale) ** 2 + (b / scale) ** 2) ** 0.5
    if roe < 0.0:
        r = -r
    c = a / r
    s = b / r
    z = 1.0
    if abs_a > abs_b:
        z = s
    elif c != 0.0:
        z = 1.0 / c
    return r, z, c, s


def _complex_rotg_reference(a, b):
    abs_a = abs(a)
    if abs_a == 0.0:
        return b, 0.0, 1.0 + 0.0j

    abs_b = abs(b)
    scale = abs_a + abs_b
    norm = scale * (
        abs(a / scale) * abs(a / scale) + abs(b / scale) * abs(b / scale)
    ) ** 0.5
    alpha = a / abs_a
    c = abs_a / norm
    s = alpha * b.conjugate() / norm
    return alpha * norm, c, s


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

    ref_a, ref_b, ref_c, ref_s = _real_rotg_reference(a_val, b_val)
    if dtype == torch.float32:
        flag_blas.ops.srotg(a, b, c, s)
        rtol, atol = 1e-5, 1e-5
    else:
        flag_blas.ops.drotg(a, b, c, s)
        rtol, atol = 1e-12, 1e-12

    expected = torch.tensor([ref_a, ref_b, ref_c, ref_s], dtype=dtype)
    actual = torch.stack((a.cpu()[0], b.cpu()[0], c.cpu()[0], s.cpu()[0]))
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.rotg
def test_accuracy_rotg_flag_blas_api():
    a = torch.tensor([3.0], dtype=torch.float32, device=flag_blas.device)
    b = torch.tensor([4.0], dtype=torch.float32, device=flag_blas.device)
    c = torch.zeros(1, dtype=torch.float32, device=flag_blas.device)
    s = torch.zeros(1, dtype=torch.float32, device=flag_blas.device)

    flag_blas.srotg(a, b, c, s)

    torch.testing.assert_close(a.cpu(), torch.tensor([5.0]), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        b.cpu(), torch.tensor([5.0 / 3.0]), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(c.cpu(), torch.tensor([0.6]), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(s.cpu(), torch.tensor([0.8]), rtol=1e-5, atol=1e-5)


@pytest.mark.rotg
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("a_val,b_val", ROTG_COMPLEX_CASES)
def test_accuracy_rotg_complex(dtype, a_val, b_val):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    a = torch.tensor([a_val], dtype=dtype, device=flag_blas.device)
    b = torch.tensor([b_val], dtype=dtype, device=flag_blas.device)
    c = torch.zeros(1, dtype=real_dtype, device=flag_blas.device)
    s = torch.zeros(1, dtype=dtype, device=flag_blas.device)

    ref_a, ref_c, ref_s = _complex_rotg_reference(a_val, b_val)
    if dtype == torch.complex64:
        flag_blas.ops.crotg(a, b, c, s)
        rtol, atol = 1e-5, 1e-5
    else:
        flag_blas.ops.zrotg(a, b, c, s)
        rtol, atol = 1e-12, 1e-12

    torch.testing.assert_close(
        a.cpu(), torch.tensor([ref_a], dtype=dtype), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        b.cpu(), torch.tensor([b_val], dtype=dtype), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        c.cpu(), torch.tensor([ref_c], dtype=real_dtype), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        s.cpu(), torch.tensor([ref_s], dtype=dtype), rtol=rtol, atol=atol
    )


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
