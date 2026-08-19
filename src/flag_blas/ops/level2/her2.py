# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

_HER2_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 8}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 32}, num_warps=4, num_stages=2),
]
_ZHER2_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 32}, num_warps=8, num_stages=1),
]
_HER2_KEY = ["n", "LDA", "INCX", "INCY", "UPLO"]
_HER2_RESTORE = ["a_ptr"]


def _f64_to_i64(v: float) -> int:
    return struct.unpack("<q", struct.pack("<d", v))[0]


def _row_major_uplo(uplo: int) -> int:
    return (
        CUBLAS_FILL_MODE_LOWER
        if uplo == CUBLAS_FILL_MODE_UPPER
        else CUBLAS_FILL_MODE_UPPER
    )


@libentry()
@triton.autotune(configs=_HER2_CONFIGS, key=_HER2_KEY, restore_value=_HER2_RESTORE)
@triton.jit
def cher2_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
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

    x_rows_off = rows * INCX * 2
    y_rows_off = rows * INCY * 2
    x_cols_off = cols * INCX * 2
    y_cols_off = cols * INCY * 2

    xrr = tl.load(x_ptr + x_rows_off, mask=row_mask, other=0.0)
    xri = tl.load(x_ptr + x_rows_off + 1, mask=row_mask, other=0.0)
    yrr = tl.load(y_ptr + y_rows_off, mask=row_mask, other=0.0)
    yri = tl.load(y_ptr + y_rows_off + 1, mask=row_mask, other=0.0)
    xcr = tl.load(x_ptr + x_cols_off, mask=col_mask, other=0.0)
    xci = tl.load(x_ptr + x_cols_off + 1, mask=col_mask, other=0.0)
    ycr = tl.load(y_ptr + y_cols_off, mask=col_mask, other=0.0)
    yci = tl.load(y_ptr + y_cols_off + 1, mask=col_mask, other=0.0)

    p1r = yrr[:, None] * xcr[None, :] + yri[:, None] * xci[None, :]
    p1i = yrr[:, None] * xci[None, :] - yri[:, None] * xcr[None, :]
    p2r = xrr[:, None] * ycr[None, :] + xri[:, None] * yci[None, :]
    p2i = xrr[:, None] * yci[None, :] - xri[:, None] * ycr[None, :]

    update_r = alpha_r * p1r - alpha_i * p1i + alpha_r * p2r + alpha_i * p2i
    update_i = alpha_r * p1i + alpha_i * p1r + alpha_r * p2i - alpha_i * p2r

    a_off = (rows[:, None] + cols[None, :] * LDA) * 2
    ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
    ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
    diag = rows[:, None] == cols[None, :]
    out_i = tl.where(diag, 0.0, ai + update_i)
    tl.store(a_ptr + a_off, ar + update_r, mask=mask)
    tl.store(a_ptr + a_off + 1, out_i, mask=mask)


@libentry()
@triton.autotune(configs=_ZHER2_CONFIGS, key=_HER2_KEY, restore_value=_HER2_RESTORE)
@triton.jit
def zher2_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r_int: tl.int64,
    alpha_i_int: tl.int64,
    n,
    LDA,
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
    if UPLO == 0:
        tri_mask = rows[:, None] >= cols[None, :]
    else:
        tri_mask = rows[:, None] <= cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :] & tri_mask

    x_rows_off = rows * INCX * 2
    y_rows_off = rows * INCY * 2
    x_cols_off = cols * INCX * 2
    y_cols_off = cols * INCY * 2

    xrr = tl.load(x_ptr + x_rows_off, mask=row_mask, other=0.0)
    xri = tl.load(x_ptr + x_rows_off + 1, mask=row_mask, other=0.0)
    yrr = tl.load(y_ptr + y_rows_off, mask=row_mask, other=0.0)
    yri = tl.load(y_ptr + y_rows_off + 1, mask=row_mask, other=0.0)
    xcr = tl.load(x_ptr + x_cols_off, mask=col_mask, other=0.0)
    xci = tl.load(x_ptr + x_cols_off + 1, mask=col_mask, other=0.0)
    ycr = tl.load(y_ptr + y_cols_off, mask=col_mask, other=0.0)
    yci = tl.load(y_ptr + y_cols_off + 1, mask=col_mask, other=0.0)

    p1r = yrr[:, None] * xcr[None, :] + yri[:, None] * xci[None, :]
    p1i = yrr[:, None] * xci[None, :] - yri[:, None] * xcr[None, :]
    p2r = xrr[:, None] * ycr[None, :] + xri[:, None] * yci[None, :]
    p2i = xrr[:, None] * yci[None, :] - xri[:, None] * ycr[None, :]

    update_r = alpha_r * p1r - alpha_i * p1i + alpha_r * p2r + alpha_i * p2i
    update_i = alpha_r * p1i + alpha_i * p1r + alpha_r * p2i - alpha_i * p2r

    a_off = (rows[:, None] + cols[None, :] * LDA) * 2
    ar = tl.load(a_ptr + a_off, mask=mask, other=0.0)
    ai = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
    diag = rows[:, None] == cols[None, :]
    out_i = tl.where(diag, 0.0, ai + update_i)
    tl.store(a_ptr + a_off, ar + update_r, mask=mask)
    tl.store(a_ptr + a_off + 1, out_i, mask=mask)


def _check_her2_args(dtype, uplo, n, x, incx, y, incy, A, lda) -> None:
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


def _complex_scalar(alpha: ScalarType):
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    ar = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    ai = float(alpha.imag) if isinstance(alpha, complex) else 0.0
    return ar, ai


def cher2(
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
    _check_her2_args(torch.complex64, uplo, n, x, incx, y, incy, A, lda)
    if n == 0:
        return
    uplo = _row_major_uplo(uplo)
    ar, ai = _complex_scalar(alpha)
    if ar == 0.0 and ai == 0.0:
        return

    def grid(meta):
        blocks = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (blocks, blocks)

    with torch_device_fn.device(A.device):
        cher2_kernel[grid](
            torch.view_as_real(A),
            torch.view_as_real(x),
            torch.view_as_real(y),
            ar,
            ai,
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
        )


def zher2(
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
    _check_her2_args(torch.complex128, uplo, n, x, incx, y, incy, A, lda)
    if n == 0:
        return
    uplo = _row_major_uplo(uplo)
    ar, ai = _complex_scalar(alpha)
    if ar == 0.0 and ai == 0.0:
        return

    def grid(meta):
        blocks = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (blocks, blocks)

    with torch_device_fn.device(A.device):
        zher2_kernel[grid](
            torch.view_as_real(A),
            torch.view_as_real(x),
            torch.view_as_real(y),
            _f64_to_i64(ar),
            _f64_to_i64(ai),
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
        )
