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

_SPR_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 8}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 32}, num_warps=4, num_stages=2),
]
_SPR_KEY = ["n", "INCX", "UPLO"]
_SPR_RESTORE = ["ap_ptr"]


def _f64_to_i64(v: float) -> int:
    return struct.unpack("<q", struct.pack("<d", v))[0]


@libentry()
@triton.autotune(configs=_SPR_CONFIGS, key=_SPR_KEY, restore_value=_SPR_RESTORE)
@triton.jit
def sspr_kernel(
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
    x_rows = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
    x_cols = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)
    ap_vals = tl.load(ap_ptr + off, mask=mask, other=0.0)
    tl.store(
        ap_ptr + off, ap_vals + alpha * x_rows[:, None] * x_cols[None, :], mask=mask
    )


@libentry()
@triton.autotune(configs=_SPR_CONFIGS, key=_SPR_KEY, restore_value=_SPR_RESTORE)
@triton.jit
def dspr_kernel(
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
    x_rows = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
    x_cols = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)
    ap_vals = tl.load(ap_ptr + off, mask=mask, other=0.0)
    tl.store(
        ap_ptr + off, ap_vals + alpha * x_rows[:, None] * x_cols[None, :], mask=mask
    )


def _check_spr_args(dtype, uplo, n, alpha, x, incx, AP) -> None:
    assert AP.dtype == dtype == x.dtype
    assert AP.is_contiguous() and x.is_contiguous()
    assert AP.device == x.device
    assert uplo in (CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER)
    assert n >= 0
    assert incx > 0
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert AP.numel() >= n * (n + 1) // 2


def sspr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    AP: torch.Tensor,
) -> None:
    _check_spr_args(torch.float32, uplo, n, alpha, x, incx, AP)
    if n == 0:
        return

    alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha == 0.0:
        return

    def grid(meta):
        blocks = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (blocks, blocks)

    with torch_device_fn.device(AP.device):
        sspr_kernel[grid](AP, x, alpha, n, incx, UPLO=uplo)


def dspr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    AP: torch.Tensor,
) -> None:
    _check_spr_args(torch.float64, uplo, n, alpha, x, incx, AP)
    if n == 0:
        return

    alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_val == 0.0:
        return

    def grid(meta):
        blocks = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (blocks, blocks)

    with torch_device_fn.device(AP.device):
        dspr_kernel[grid](AP, x, _f64_to_i64(alpha_val), n, incx, UPLO=uplo)
