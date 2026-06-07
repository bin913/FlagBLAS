import logging

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry
from flag_blas.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n", "incx", "incy"],
    restore_value=["x_ptr", "y_ptr"],
)
@triton.jit
def _rotm_kernel(
    x_ptr,
    y_ptr,
    n,
    incx,
    incy,
    param_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x_offs = idx * incx
    y_offs = idx * incy

    x = tl.load(x_ptr + x_offs, mask=mask)
    y = tl.load(y_ptr + y_offs, mask=mask)

    flag = tl.load(param_ptr)
    h11 = tl.load(param_ptr + 1)
    h21 = tl.load(param_ptr + 2)
    h12 = tl.load(param_ptr + 3)
    h22 = tl.load(param_ptr + 4)

    new_x_m1 = h11 * x + h12 * y
    new_y_m1 = h21 * x + h22 * y
    new_x_0 = x + h12 * y
    new_y_0 = h21 * x + y
    new_x_1 = h11 * x + y
    new_y_1 = -x + h22 * y

    new_x = tl.where(
        flag == -2.0,
        x,
        tl.where(flag < 0.0, new_x_m1, tl.where(flag == 0.0, new_x_0, new_x_1)),
    )
    new_y = tl.where(
        flag == -2.0,
        y,
        tl.where(flag < 0.0, new_y_m1, tl.where(flag == 0.0, new_y_0, new_y_1)),
    )

    tl.store(x_ptr + x_offs, new_x, mask=mask)
    tl.store(y_ptr + y_offs, new_y, mask=mask)


def _validate_rotm_inputs(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    param: torch.Tensor,
) -> None:
    if n <= 0:
        return

    assert incx > 0, "incx must be positive"
    assert incy > 0, "incy must be positive"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert param.dim() == 1, "param must be 1-dimensional"
    assert param.numel() == 5, "param must have exactly five elements"
    assert x.dtype == y.dtype == param.dtype, "x, y, and param must have the same dtype"
    assert x.device == y.device == param.device, (
        "x, y, and param must be on the same device"
    )

    required_x = 1 + (n - 1) * incx
    required_y = 1 + (n - 1) * incy
    assert x.numel() >= required_x, (
        f"x is too short: need at least {required_x} elements "
        f"for n={n}, incx={incx}, got {x.numel()}"
    )
    assert y.numel() >= required_y, (
        f"y is too short: need at least {required_y} elements "
        f"for n={n}, incy={incy}, got {y.numel()}"
    )


def _rotm_impl(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    param: torch.Tensor,
) -> None:
    if n <= 0:
        return

    with torch_device_fn.device(x.device):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _rotm_kernel[grid](x, y, n, incx, incy, param)


def srotm(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    param: torch.Tensor,
) -> None:
    logger.debug("FLAG_BLAS SROTM")
    assert x.dtype == torch.float32, "x must be float32"
    assert y.dtype == torch.float32, "y must be float32"
    assert param.dtype == torch.float32, "param must be float32"
    _validate_rotm_inputs(n, x, incx, y, incy, param)
    _rotm_impl(n, x, incx, y, incy, param)


def drotm(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    param: torch.Tensor,
) -> None:
    logger.debug("FLAG_BLAS DROTM")
    assert x.dtype == torch.float64, "x must be float64"
    assert y.dtype == torch.float64, "y must be float64"
    assert param.dtype == torch.float64, "param must be float64"
    _validate_rotm_inputs(n, x, incx, y, incy, param)
    _rotm_impl(n, x, incx, y, incy, param)
