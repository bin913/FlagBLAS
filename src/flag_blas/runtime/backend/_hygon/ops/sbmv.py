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

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2.sbmv import (
    ScalarType,
    _check_common,
    _f64_to_i64,
    _row_major_sbmv_uplo,
    _strided_y,
)
from flag_blas.ops.level2.sbmv import dsbmv as common_dsbmv
from flag_blas.ops.level2.sbmv import ssbmv as common_ssbmv
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry


@libentry()
@triton.jit
def ssbmv_hygon_row_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    n,
    k,
    LDA,
    INCX,
    INCY,
    UPLO: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    delta = cols - row
    abs_delta = tl.where(delta >= 0, delta, -delta)
    mask = (cols < n) & (abs_delta <= k)
    if UPLO == 1:
        packed_row = k - abs_delta
        packed_col = tl.where(delta >= 0, cols, row)
    else:
        packed_row = abs_delta
        packed_col = tl.where(delta >= 0, row, cols)
    a = tl.load(
        a_ptr + packed_row + packed_col * LDA,
        mask=mask,
        other=0.0,
    )
    x = tl.load(x_ptr + cols * INCX, mask=mask, other=0.0)
    acc = tl.sum(a * x, axis=0)
    if BETA_IS_ZERO:
        out = alpha * acc
    else:
        out = alpha * acc + beta * tl.load(y_ptr + row * INCY)
    tl.store(y_ptr + row * INCY, out)


@libentry()
@triton.jit
def dsbmv_hygon_row_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    beta_int: tl.int64,
    n,
    k,
    LDA,
    INCX,
    INCY,
    UPLO: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    delta = cols - row
    abs_delta = tl.where(delta >= 0, delta, -delta)
    mask = (cols < n) & (abs_delta <= k)
    if UPLO == 1:
        packed_row = k - abs_delta
        packed_col = tl.where(delta >= 0, cols, row)
    else:
        packed_row = abs_delta
        packed_col = tl.where(delta >= 0, row, cols)
    a = tl.load(
        a_ptr + packed_row + packed_col * LDA,
        mask=mask,
        other=0.0,
    )
    x = tl.load(x_ptr + cols * INCX, mask=mask, other=0.0)
    acc = tl.sum(a * x, axis=0)
    alpha = alpha_int.to(tl.float64, bitcast=True)
    if BETA_IS_ZERO:
        out = alpha * acc
    else:
        beta = beta_int.to(tl.float64, bitcast=True)
        out = alpha * acc + beta * tl.load(y_ptr + row * INCY)
    tl.store(y_ptr + row * INCY, out)


def _use_hygon_row_kernel(n: int, k: int) -> bool:
    return 0 < n <= 512 and k >= 128


def ssbmv(
    uplo: int,
    n: int,
    k: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    if not _use_hygon_row_kernel(n, k):
        common_ssbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
        return

    assert A.dtype == torch.float32 == x.dtype == y.dtype
    _check_common(A, x, y, uplo, n, k, lda, incx, incy)
    alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta_val = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    if alpha_val == 0.0:
        y_view = _strided_y(y, n, incy)
        if beta_val == 0.0:
            y_view.zero_()
        elif beta_val != 1.0:
            y_view.mul_(beta_val)
        return

    uplo = _row_major_sbmv_uplo(uplo)
    with torch_device_fn.device(A.device):
        ssbmv_hygon_row_kernel[(n,)](
            A,
            x,
            y,
            alpha_val,
            beta_val,
            n,
            k,
            lda,
            incx,
            incy,
            UPLO=uplo,
            BETA_IS_ZERO=beta_val == 0.0,
            BLOCK_N=triton.next_power_of_2(n),
            num_warps=8 if n > 256 else 4,
            num_stages=1,
        )


def dsbmv(
    uplo: int,
    n: int,
    k: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    if not _use_hygon_row_kernel(n, k):
        common_dsbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
        return

    assert A.dtype == torch.float64 == x.dtype == y.dtype
    _check_common(A, x, y, uplo, n, k, lda, incx, incy)
    alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta_val = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    if alpha_val == 0.0:
        y_view = _strided_y(y, n, incy)
        if beta_val == 0.0:
            y_view.zero_()
        elif beta_val != 1.0:
            y_view.mul_(beta_val)
        return

    uplo = _row_major_sbmv_uplo(uplo)
    with torch_device_fn.device(A.device):
        dsbmv_hygon_row_kernel[(n,)](
            A,
            x,
            y,
            _f64_to_i64(alpha_val),
            _f64_to_i64(beta_val),
            n,
            k,
            lda,
            incx,
            incy,
            UPLO=uplo,
            BETA_IS_ZERO=beta_val == 0.0,
            BLOCK_N=triton.next_power_of_2(n),
            num_warps=4,
            num_stages=1,
        )


__all__ = ["dsbmv", "ssbmv"]
