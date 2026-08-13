import logging

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _rotmg_kernel(d1_ptr, d2_ptr, x1_ptr, y1_ptr, param_ptr):
    d1 = tl.load(d1_ptr)
    d2 = tl.load(d2_ptr)
    x1 = tl.load(x1_ptr)
    y1 = tl.load(y1_ptr)

    zero = d1 * 0.0
    one = zero + 1.0
    flag = -one - one
    gam = zero + 4096.0
    gamsq = gam * gam
    rgamsq = one / gamsq

    h11 = zero
    h12 = zero
    h21 = zero
    h22 = zero

    if d1 < 0.0:
        flag = -one
        d1 = zero
        d2 = zero
        x1 = zero
    else:
        p2 = d2 * y1
        if p2 == 0.0:
            flag = -one - one
        else:
            p1 = d1 * x1
            q2 = p2 * y1
            q1 = p1 * x1

            if tl.abs(q1) > tl.abs(q2):
                h21 = -y1 / x1
                h12 = p2 / p1
                u = one - h12 * h21
                if u > 0.0:
                    flag = zero
                    d1 = d1 / u
                    d2 = d2 / u
                    x1 = x1 * u
                else:
                    flag = -one
                    d1 = zero
                    d2 = zero
                    x1 = zero
            else:
                if q2 < 0.0:
                    flag = -one
                    d1 = zero
                    d2 = zero
                    x1 = zero
                else:
                    flag = one
                    h11 = p1 / p2
                    h22 = x1 / y1
                    u = one + h11 * h22
                    temp = d2 / u
                    d2 = d1 / u
                    d1 = temp
                    x1 = y1 * u

            if flag >= 0.0:
                if d1 != 0.0:
                    while d1 <= rgamsq or d1 >= gamsq:
                        if flag == 0.0:
                            h11 = one
                            h22 = one
                        else:
                            h21 = -one
                            h12 = one
                        flag = -one

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
                    while tl.abs(d2) <= rgamsq or tl.abs(d2) >= gamsq:
                        if flag == 0.0:
                            h11 = one
                            h22 = one
                        else:
                            h21 = -one
                            h12 = one
                        flag = -one

                        if tl.abs(d2) <= rgamsq:
                            d2 = d2 * gamsq
                            h21 = h21 / gam
                            h22 = h22 / gam
                        else:
                            d2 = d2 / gamsq
                            h21 = h21 * gam
                            h22 = h22 * gam

    tl.store(d1_ptr, d1)
    tl.store(d2_ptr, d2)
    tl.store(x1_ptr, x1)
    tl.store(param_ptr, flag)

    if flag < 0.0:
        tl.store(param_ptr + 1, h11)
        tl.store(param_ptr + 2, h21)
        tl.store(param_ptr + 3, h12)
        tl.store(param_ptr + 4, h22)
    elif flag == 0.0:
        tl.store(param_ptr + 1, zero)
        tl.store(param_ptr + 2, h21)
        tl.store(param_ptr + 3, h12)
        tl.store(param_ptr + 4, zero)
    else:
        tl.store(param_ptr + 1, h11)
        tl.store(param_ptr + 2, zero)
        tl.store(param_ptr + 3, zero)
        tl.store(param_ptr + 4, h22)


def _validate_rotmg_inputs(d1, d2, x1, y1, param):
    for name, tensor in (("d1", d1), ("d2", d2), ("x1", x1), ("y1", y1)):
        assert tensor.dim() == 1, f"{name} must be 1-dimensional"
        assert tensor.numel() == 1, f"{name} must have exactly one element"
    assert param.dim() == 1, "param must be 1-dimensional"
    assert param.numel() == 5, "param must have exactly five elements"
    assert (
        d1.device == d2.device == x1.device == y1.device == param.device
    ), "d1, d2, x1, y1, and param must be on the same device"
    assert (
        d1.dtype == d2.dtype == x1.dtype == y1.dtype == param.dtype
    ), "d1, d2, x1, y1, and param must have the same dtype"


def _rotmg_impl(d1, d2, x1, y1, param):
    _validate_rotmg_inputs(d1, d2, x1, y1, param)
    with torch_device_fn.device(d1.device):
        _rotmg_kernel[(1,)](d1, d2, x1, y1, param)


def srotmg(
    d1: torch.Tensor,
    d2: torch.Tensor,
    x1: torch.Tensor,
    y1: torch.Tensor,
    param: torch.Tensor,
) -> None:
    logger.debug("FLAG_BLAS SROTMG")
    assert d1.dtype == torch.float32, "d1 must be float32"
    assert d2.dtype == torch.float32, "d2 must be float32"
    assert x1.dtype == torch.float32, "x1 must be float32"
    assert y1.dtype == torch.float32, "y1 must be float32"
    assert param.dtype == torch.float32, "param must be float32"
    _rotmg_impl(d1, d2, x1, y1, param)


def drotmg(
    d1: torch.Tensor,
    d2: torch.Tensor,
    x1: torch.Tensor,
    y1: torch.Tensor,
    param: torch.Tensor,
) -> None:
    logger.debug("FLAG_BLAS DROTMG")
    assert d1.dtype == torch.float64, "d1 must be float64"
    assert d2.dtype == torch.float64, "d2 must be float64"
    assert x1.dtype == torch.float64, "x1 must be float64"
    assert y1.dtype == torch.float64, "y1 must be float64"
    assert param.dtype == torch.float64, "param must be float64"
    _rotmg_impl(d1, d2, x1, y1, param)
