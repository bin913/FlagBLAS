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

_HPR_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 8}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 32}, num_warps=4, num_stages=2),
]
_HPR_KEY = ["n", "INCX", "UPLO"]
_HPR_RESTORE = ["ap_ptr"]


def _f64_to_i64(v: float) -> int:
    return struct.unpack("<q", struct.pack("<d", v))[0]


@libentry()
@triton.autotune(configs=_HPR_CONFIGS, key=_HPR_KEY, restore_value=_HPR_RESTORE)
@triton.jit
def chpr_kernel(
    ap_ptr,
    x_ptr,
    alpha: tl.float32,
    n,
    INCX,
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

    xr = tl.load(x_ptr + rows * INCX * 2, mask=row_mask, other=0.0)
    xi = tl.load(x_ptr + rows * INCX * 2 + 1, mask=row_mask, other=0.0)
    xcr = tl.load(x_ptr + cols * INCX * 2, mask=col_mask, other=0.0)
    xci = tl.load(x_ptr + cols * INCX * 2 + 1, mask=col_mask, other=0.0)
    update_r = alpha * (xr[:, None] * xcr[None, :] + xi[:, None] * xci[None, :])
    update_i = alpha * (xi[:, None] * xcr[None, :] - xr[:, None] * xci[None, :])

    ap_off = off * 2
    ar = tl.load(ap_ptr + ap_off, mask=mask, other=0.0)
    ai = tl.load(ap_ptr + ap_off + 1, mask=mask, other=0.0)
    diag = rows[:, None] == cols[None, :]
    out_i = tl.where(diag, 0.0, ai + update_i)
    tl.store(ap_ptr + ap_off, ar + update_r, mask=mask)
    tl.store(ap_ptr + ap_off + 1, out_i, mask=mask)


@libentry()
@triton.autotune(configs=_HPR_CONFIGS, key=_HPR_KEY, restore_value=_HPR_RESTORE)
@triton.jit
def zhpr_kernel(
    ap_ptr,
    x_ptr,
    alpha_int: tl.int64,
    n,
    INCX,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    alpha = alpha_int.to(tl.float64, bitcast=True)
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

    xr = tl.load(x_ptr + rows * INCX * 2, mask=row_mask, other=0.0)
    xi = tl.load(x_ptr + rows * INCX * 2 + 1, mask=row_mask, other=0.0)
    xcr = tl.load(x_ptr + cols * INCX * 2, mask=col_mask, other=0.0)
    xci = tl.load(x_ptr + cols * INCX * 2 + 1, mask=col_mask, other=0.0)
    update_r = alpha * (xr[:, None] * xcr[None, :] + xi[:, None] * xci[None, :])
    update_i = alpha * (xi[:, None] * xcr[None, :] - xr[:, None] * xci[None, :])

    ap_off = off * 2
    ar = tl.load(ap_ptr + ap_off, mask=mask, other=0.0)
    ai = tl.load(ap_ptr + ap_off + 1, mask=mask, other=0.0)
    diag = rows[:, None] == cols[None, :]
    out_i = tl.where(diag, 0.0, ai + update_i)
    tl.store(ap_ptr + ap_off, ar + update_r, mask=mask)
    tl.store(ap_ptr + ap_off + 1, out_i, mask=mask)


def _check_hpr_args(dtype, uplo, n, x, incx, AP) -> None:
    assert AP.dtype == dtype == x.dtype
    assert AP.is_contiguous() and x.is_contiguous()
    assert AP.device == x.device
    assert uplo in (CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER)
    assert n >= 0
    assert incx > 0
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert AP.numel() >= n * (n + 1) // 2


def chpr(
    uplo: int, n: int, alpha: ScalarType, x: torch.Tensor, incx: int, AP: torch.Tensor
) -> None:
    _check_hpr_args(torch.complex64, uplo, n, x, incx, AP)
    if n == 0:
        return
    alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha == 0.0:
        return

    def grid(meta):
        blocks = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (blocks, blocks)

    with torch_device_fn.device(AP.device):
        chpr_kernel[grid](
            torch.view_as_real(AP),
            torch.view_as_real(x),
            alpha,
            n,
            incx,
            UPLO=uplo,
        )


def zhpr(
    uplo: int, n: int, alpha: ScalarType, x: torch.Tensor, incx: int, AP: torch.Tensor
) -> None:
    _check_hpr_args(torch.complex128, uplo, n, x, incx, AP)
    if n == 0:
        return
    alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_val == 0.0:
        return

    def grid(meta):
        blocks = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (blocks, blocks)

    with torch_device_fn.device(AP.device):
        zhpr_kernel[grid](
            torch.view_as_real(AP),
            torch.view_as_real(x),
            _f64_to_i64(alpha_val),
            n,
            incx,
            UPLO=uplo,
        )
