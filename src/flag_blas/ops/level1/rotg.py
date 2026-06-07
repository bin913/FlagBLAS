import logging

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _rotg_real_kernel(a_ptr, b_ptr, c_ptr, s_ptr):
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)

    abs_a = tl.abs(a)
    abs_b = tl.abs(b)
    scale = abs_a + abs_b
    zero = a * 0.0
    one = zero + 1.0

    if scale == 0.0:
        tl.store(a_ptr, zero)
        tl.store(b_ptr, zero)
        tl.store(c_ptr, one)
        tl.store(s_ptr, zero)
        return

    roe = tl.where(abs_a > abs_b, a, b)
    a_scaled = a / scale
    b_scaled = b / scale
    r = scale * tl.sqrt(a_scaled * a_scaled + b_scaled * b_scaled)
    r = tl.where(roe < 0.0, -r, r)
    c = a / r
    s = b / r

    z = one
    if abs_a > abs_b:
        z = s
    elif c != 0.0:
        z = one / c

    tl.store(a_ptr, r)
    tl.store(b_ptr, z)
    tl.store(c_ptr, c)
    tl.store(s_ptr, s)


@libentry()
@triton.jit
def _rotg_complex_kernel(a_ptr, b_ptr, c_ptr, s_ptr):
    a_re = tl.load(a_ptr)
    a_im = tl.load(a_ptr + 1)
    b_re = tl.load(b_ptr)
    b_im = tl.load(b_ptr + 1)

    zero = a_re * 0.0
    one = zero + 1.0
    a_scale = tl.maximum(tl.abs(a_re), tl.abs(a_im))
    b_scale = tl.maximum(tl.abs(b_re), tl.abs(b_im))
    a_safe_scale = tl.where(a_scale == 0.0, one, a_scale)
    b_safe_scale = tl.where(b_scale == 0.0, one, b_scale)
    a_re_norm = a_re / a_safe_scale
    a_im_norm = a_im / a_safe_scale
    b_re_norm = b_re / b_safe_scale
    b_im_norm = b_im / b_safe_scale
    abs_a = tl.where(
        a_scale == 0.0,
        zero,
        a_scale * tl.sqrt(a_re_norm * a_re_norm + a_im_norm * a_im_norm),
    )
    abs_b = tl.where(
        b_scale == 0.0,
        zero,
        b_scale * tl.sqrt(b_re_norm * b_re_norm + b_im_norm * b_im_norm),
    )

    if abs_a == 0.0:
        tl.store(c_ptr, zero)
        tl.store(s_ptr, one)
        tl.store(s_ptr + 1, zero)
        tl.store(a_ptr, b_re)
        tl.store(a_ptr + 1, b_im)
        return

    scale = abs_a + abs_b
    a_re_scaled = a_re / scale
    a_im_scaled = a_im / scale
    b_re_scaled = b_re / scale
    b_im_scaled = b_im / scale
    norm = scale * tl.sqrt(
        a_re_scaled * a_re_scaled
        + a_im_scaled * a_im_scaled
        + b_re_scaled * b_re_scaled
        + b_im_scaled * b_im_scaled
    )

    alpha_re = a_re / abs_a
    alpha_im = a_im / abs_a
    c = abs_a / norm

    # s = alpha * conj(b) / norm
    s_re = (alpha_re * b_re + alpha_im * b_im) / norm
    s_im = (alpha_im * b_re - alpha_re * b_im) / norm
    a_new_re = alpha_re * norm
    a_new_im = alpha_im * norm

    tl.store(c_ptr, c)
    tl.store(s_ptr, s_re)
    tl.store(s_ptr + 1, s_im)
    tl.store(a_ptr, a_new_re)
    tl.store(a_ptr + 1, a_new_im)


def _validate_rotg_inputs(a, b, c, s):
    for name, tensor in (("a", a), ("b", b), ("c", c), ("s", s)):
        assert tensor.dim() == 1, f"{name} must be 1-dimensional"
        assert tensor.numel() == 1, f"{name} must have exactly one element"
    assert a.device == b.device == c.device == s.device, (
        "a, b, c, and s must be on the same device"
    )


def _rotg_real_impl(a, b, c, s):
    _validate_rotg_inputs(a, b, c, s)
    with torch_device_fn.device(a.device):
        _rotg_real_kernel[(1,)](a, b, c, s)


def _rotg_complex_impl(a, b, c, s):
    _validate_rotg_inputs(a, b, c, s)
    a_real = torch.view_as_real(a).reshape(-1)
    b_real = torch.view_as_real(b).reshape(-1)
    s_real = torch.view_as_real(s).reshape(-1)
    with torch_device_fn.device(a.device):
        _rotg_complex_kernel[(1,)](a_real, b_real, c, s_real)


def srotg(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS SROTG")
    assert a.dtype == torch.float32, "a must be float32"
    assert b.dtype == torch.float32, "b must be float32"
    assert c.dtype == torch.float32, "c must be float32"
    assert s.dtype == torch.float32, "s must be float32"
    _rotg_real_impl(a, b, c, s)


def drotg(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS DROTG")
    assert a.dtype == torch.float64, "a must be float64"
    assert b.dtype == torch.float64, "b must be float64"
    assert c.dtype == torch.float64, "c must be float64"
    assert s.dtype == torch.float64, "s must be float64"
    _rotg_real_impl(a, b, c, s)


def crotg(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS CROTG")
    assert a.dtype == torch.complex64, "a must be complex64"
    assert b.dtype == torch.complex64, "b must be complex64"
    assert c.dtype == torch.float32, "c must be float32"
    assert s.dtype == torch.complex64, "s must be complex64"
    _rotg_complex_impl(a, b, c, s)


def zrotg(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS ZROTG")
    assert a.dtype == torch.complex128, "a must be complex128"
    assert b.dtype == torch.complex128, "b must be complex128"
    assert c.dtype == torch.float64, "c must be float64"
    assert s.dtype == torch.complex128, "s must be complex128"
    _rotg_complex_impl(a, b, c, s)
