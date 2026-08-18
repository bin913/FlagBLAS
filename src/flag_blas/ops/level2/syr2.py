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

ScalarType = Union[float, int, torch.Tensor]

_SYR2_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 8}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 32}, num_warps=4, num_stages=2),
]
_SYR2_KEY = ["n", "LDA", "INCX", "INCY", "UPLO"]
_SYR2_RESTORE = ["a_ptr"]


def _f64_to_i64(v: float) -> int:
    return struct.unpack("<q", struct.pack("<d", v))[0]


@libentry()
@triton.autotune(configs=_SYR2_CONFIGS, key=_SYR2_KEY, restore_value=_SYR2_RESTORE)
@triton.jit
def ssyr2_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    n,
    LDA,
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

    if UPLO == 0:
        tri_mask = rows[:, None] >= cols[None, :]
    else:
        tri_mask = rows[:, None] <= cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :] & tri_mask

    x_rows = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
    y_rows = tl.load(y_ptr + rows * INCY, mask=row_mask, other=0.0)
    x_cols = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)
    y_cols = tl.load(y_ptr + cols * INCY, mask=col_mask, other=0.0)

    a_off = rows[:, None] + cols[None, :] * LDA
    a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
    update = alpha * (
        x_rows[:, None] * y_cols[None, :] + y_rows[:, None] * x_cols[None, :]
    )
    tl.store(a_ptr + a_off, a_vals + update, mask=mask)


@libentry()
@triton.autotune(configs=_SYR2_CONFIGS, key=_SYR2_KEY, restore_value=_SYR2_RESTORE)
@triton.jit
def dsyr2_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    n,
    LDA,
    INCX,
    INCY,
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

    if UPLO == 0:
        tri_mask = rows[:, None] >= cols[None, :]
    else:
        tri_mask = rows[:, None] <= cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :] & tri_mask

    x_rows = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
    y_rows = tl.load(y_ptr + rows * INCY, mask=row_mask, other=0.0)
    x_cols = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)
    y_cols = tl.load(y_ptr + cols * INCY, mask=col_mask, other=0.0)

    a_off = rows[:, None] + cols[None, :] * LDA
    a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
    update = alpha * (
        x_rows[:, None] * y_cols[None, :] + y_rows[:, None] * x_cols[None, :]
    )
    tl.store(a_ptr + a_off, a_vals + update, mask=mask)


# csyr2/zsyr2 are intentionally disabled for the strict workflow. SciPy/CPU
# BLAS does not expose direct csyr2/zsyr2 references, so accepting these variants
# would require hand-written or algebraically composed CPU reference math.


def _check_ssyr2_args(
    uplo: int,
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
) -> None:
    assert A.dtype == torch.float32 == x.dtype == y.dtype
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.device == x.device == y.device
    assert uplo in (CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER)
    assert n >= 0
    assert incx > 0 and incy > 0
    assert lda >= max(1, n)
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert y.numel() >= 1 + (n - 1) * incy
        assert A.numel() >= n * lda


def _check_syr2_args(dtype, n, x, incx, y, incy, A, lda, uplo) -> None:
    assert A.dtype == dtype == x.dtype == y.dtype
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.device == x.device == y.device
    assert uplo in (CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER)
    assert n >= 0
    assert incx > 0 and incy > 0
    assert lda >= max(1, n)
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert y.numel() >= 1 + (n - 1) * incy
        assert A.numel() >= n * lda


def ssyr2(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
) -> None:
    _check_ssyr2_args(uplo, n, x, incx, y, incy, A, lda)
    if n == 0:
        return

    alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha == 0.0:
        return

    def grid(meta):
        blocks = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (blocks, blocks)

    with torch_device_fn.device(A.device):
        ssyr2_kernel[grid](
            A,
            x,
            y,
            alpha,
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
        )


def dsyr2(uplo, n, alpha, x, incx, y, incy, A, lda) -> None:
    _check_syr2_args(torch.float64, n, x, incx, y, incy, A, lda, uplo)
    if n == 0:
        return
    alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_val == 0.0:
        return
    alpha_int = _f64_to_i64(alpha_val)

    def grid(meta):
        blocks = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (blocks, blocks)

    with torch_device_fn.device(A.device):
        dsyr2_kernel[grid](A, x, y, alpha_int, n, lda, incx, incy, UPLO=uplo)


# Public csyr2/zsyr2 wrappers are intentionally not defined. See the strict
# workflow unsupported-variant rule before re-enabling them.
