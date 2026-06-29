import logging
import struct
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2._constants import (
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]

_HPR2_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 8}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 32}, num_warps=4, num_stages=2),
]
_HPR2_KEY = ["n", "INCX", "INCY", "UPLO"]
_HPR2_RESTORE = ["ap_ptr"]


def _f64_to_i64(v: float) -> int:
    return struct.unpack("<q", struct.pack("<d", v))[0]


@libentry()
@triton.autotune(configs=_HPR2_CONFIGS, key=_HPR2_KEY, restore_value=_HPR2_RESTORE)
@triton.jit
def chpr2_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    n,
    INCX,
    INCY,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if UPLO == 0:
        if pid_m < pid_n:
            return
    else:
        if pid_m > pid_n:
            return
    rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < n
    col_mask = cols < n
    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    cols64 = cols.to(tl.int64)
    if UPLO == 0:
        tri_mask = rows[:, None] >= cols[None, :]
        off = rows64[:, None] + cols64[None, :] * (2 * n64 - cols64[None, :] - 1) // 2
    else:
        tri_mask = rows[:, None] <= cols[None, :]
        off = cols64[None, :] * (cols64[None, :] + 1) // 2 + rows64[:, None]
    mask = row_mask[:, None] & col_mask[None, :] & tri_mask

    xrr = tl.load(x_ptr + rows * INCX * 2, mask=row_mask, other=0.0)
    xri = tl.load(x_ptr + rows * INCX * 2 + 1, mask=row_mask, other=0.0)
    yrr = tl.load(y_ptr + rows * INCY * 2, mask=row_mask, other=0.0)
    yri = tl.load(y_ptr + rows * INCY * 2 + 1, mask=row_mask, other=0.0)
    xcr = tl.load(x_ptr + cols * INCX * 2, mask=col_mask, other=0.0)
    xci = tl.load(x_ptr + cols * INCX * 2 + 1, mask=col_mask, other=0.0)
    ycr = tl.load(y_ptr + cols * INCY * 2, mask=col_mask, other=0.0)
    yci = tl.load(y_ptr + cols * INCY * 2 + 1, mask=col_mask, other=0.0)

    p1r = xrr[:, None] * ycr[None, :] + xri[:, None] * yci[None, :]
    p1i = xri[:, None] * ycr[None, :] - xrr[:, None] * yci[None, :]
    p2r = yrr[:, None] * xcr[None, :] + yri[:, None] * xci[None, :]
    p2i = yri[:, None] * xcr[None, :] - yrr[:, None] * xci[None, :]
    update_r = alpha_r * p1r - alpha_i * p1i + alpha_r * p2r + alpha_i * p2i
    update_i = alpha_r * p1i + alpha_i * p1r + alpha_r * p2i - alpha_i * p2r

    ap_off = off * 2
    ar = tl.load(ap_ptr + ap_off, mask=mask, other=0.0)
    ai = tl.load(ap_ptr + ap_off + 1, mask=mask, other=0.0)
    diag = rows[:, None] == cols[None, :]
    out_i = tl.where(diag, 0.0, ai + update_i)
    tl.store(ap_ptr + ap_off, ar + update_r, mask=mask)
    tl.store(ap_ptr + ap_off + 1, out_i, mask=mask)


@libentry()
@triton.autotune(configs=_HPR2_CONFIGS, key=_HPR2_KEY, restore_value=_HPR2_RESTORE)
@triton.jit
def zhpr2_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha_r_int: tl.int64,
    alpha_i_int: tl.int64,
    n,
    INCX,
    INCY,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    alpha_r = alpha_r_int.to(tl.float64, bitcast=True)
    alpha_i = alpha_i_int.to(tl.float64, bitcast=True)
    if UPLO == 0:
        if pid_m < pid_n:
            return
    else:
        if pid_m > pid_n:
            return
    rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < n
    col_mask = cols < n
    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    cols64 = cols.to(tl.int64)
    if UPLO == 0:
        tri_mask = rows[:, None] >= cols[None, :]
        off = rows64[:, None] + cols64[None, :] * (2 * n64 - cols64[None, :] - 1) // 2
    else:
        tri_mask = rows[:, None] <= cols[None, :]
        off = cols64[None, :] * (cols64[None, :] + 1) // 2 + rows64[:, None]
    mask = row_mask[:, None] & col_mask[None, :] & tri_mask

    xrr = tl.load(x_ptr + rows * INCX * 2, mask=row_mask, other=0.0)
    xri = tl.load(x_ptr + rows * INCX * 2 + 1, mask=row_mask, other=0.0)
    yrr = tl.load(y_ptr + rows * INCY * 2, mask=row_mask, other=0.0)
    yri = tl.load(y_ptr + rows * INCY * 2 + 1, mask=row_mask, other=0.0)
    xcr = tl.load(x_ptr + cols * INCX * 2, mask=col_mask, other=0.0)
    xci = tl.load(x_ptr + cols * INCX * 2 + 1, mask=col_mask, other=0.0)
    ycr = tl.load(y_ptr + cols * INCY * 2, mask=col_mask, other=0.0)
    yci = tl.load(y_ptr + cols * INCY * 2 + 1, mask=col_mask, other=0.0)

    p1r = xrr[:, None] * ycr[None, :] + xri[:, None] * yci[None, :]
    p1i = xri[:, None] * ycr[None, :] - xrr[:, None] * yci[None, :]
    p2r = yrr[:, None] * xcr[None, :] + yri[:, None] * xci[None, :]
    p2i = yri[:, None] * xcr[None, :] - yrr[:, None] * xci[None, :]
    update_r = alpha_r * p1r - alpha_i * p1i + alpha_r * p2r + alpha_i * p2i
    update_i = alpha_r * p1i + alpha_i * p1r + alpha_r * p2i - alpha_i * p2r

    ap_off = off * 2
    ar = tl.load(ap_ptr + ap_off, mask=mask, other=0.0)
    ai = tl.load(ap_ptr + ap_off + 1, mask=mask, other=0.0)
    diag = rows[:, None] == cols[None, :]
    out_i = tl.where(diag, 0.0, ai + update_i)
    tl.store(ap_ptr + ap_off, ar + update_r, mask=mask)
    tl.store(ap_ptr + ap_off + 1, out_i, mask=mask)


def _check_hpr2_args(dtype, uplo, n, x, incx, y, incy, AP) -> None:
    assert AP.dtype == dtype == x.dtype == y.dtype
    assert AP.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert AP.device == x.device == y.device
    assert uplo in (CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER)
    assert n >= 0
    assert incx > 0 and incy > 0
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert y.numel() >= 1 + (n - 1) * incy
        assert AP.numel() >= n * (n + 1) // 2


def _complex_scalar(alpha: ScalarType):
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    ar = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    ai = float(alpha.imag) if isinstance(alpha, complex) else 0.0
    return ar, ai


def chpr2(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    AP: torch.Tensor,
) -> None:
    _check_hpr2_args(torch.complex64, uplo, n, x, incx, y, incy, AP)
    if n == 0:
        return
    ar, ai = _complex_scalar(alpha)
    if ar == 0.0 and ai == 0.0:
        return

    def grid(meta):
        blocks = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (blocks, blocks)

    with torch_device_fn.device(AP.device):
        chpr2_kernel[grid](
            torch.view_as_real(AP),
            torch.view_as_real(x),
            torch.view_as_real(y),
            ar,
            ai,
            n,
            incx,
            incy,
            UPLO=uplo,
        )


def zhpr2(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    AP: torch.Tensor,
) -> None:
    _check_hpr2_args(torch.complex128, uplo, n, x, incx, y, incy, AP)
    if n == 0:
        return
    ar, ai = _complex_scalar(alpha)
    if ar == 0.0 and ai == 0.0:
        return

    def grid(meta):
        blocks = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (blocks, blocks)

    with torch_device_fn.device(AP.device):
        zhpr2_kernel[grid](
            torch.view_as_real(AP),
            torch.view_as_real(x),
            torch.view_as_real(y),
            _f64_to_i64(ar),
            _f64_to_i64(ai),
            n,
            incx,
            incy,
            UPLO=uplo,
        )
