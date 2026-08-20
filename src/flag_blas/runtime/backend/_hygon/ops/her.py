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

from flag_blas import runtime
from flag_blas.ops.level2.her import ScalarType, _check_her_args, _f64_to_i64
from flag_blas.ops.level2.her import _her_kernel as common_her_kernel
from flag_blas.ops.level2.her import _row_major_uplo
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

_HER_KEY = ["n", "lda", "incx", "uplo"]
_HER_RESTORE = ["A"]

cher_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cher_hygon"),
        key=_HER_KEY,
        restore_value=_HER_RESTORE,
    )(common_her_kernel.fn)
)

zher_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("zher_hygon"),
        key=_HER_KEY,
        restore_value=_HER_RESTORE,
    )(common_her_kernel.fn)
)


@triton.jit
def zher_rect_hygon_kernel(
    x,
    A,
    n,
    alpha,
    incx,
    lda,
    uplo: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N
    if uplo == 0:
        if row_start + BLOCK_M <= col_start:
            return
    else:
        if row_start >= col_start + BLOCK_N:
            return

    alpha_value = alpha.to(tl.float64, bitcast=True)
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < n) & (cols[None, :] < n)
    if uplo == 0:
        mask = mask & (rows[:, None] >= cols[None, :])
    else:
        mask = mask & (rows[:, None] <= cols[None, :])

    xr = tl.load(x + rows * incx * 2, mask=rows < n, other=0.0)
    xi = tl.load(x + rows * incx * 2 + 1, mask=rows < n, other=0.0)
    yr = tl.load(x + cols * incx * 2, mask=cols < n, other=0.0)
    yi = tl.load(x + cols * incx * 2 + 1, mask=cols < n, other=0.0)
    prod_r = xr[:, None] * yr[None, :] + xi[:, None] * yi[None, :]
    prod_i = xr[:, None] * yi[None, :] - xi[:, None] * yr[None, :]

    a_off = (rows[:, None] + cols[None, :] * lda) * 2
    old_r = tl.load(A + a_off, mask=mask, other=0.0)
    old_i = tl.load(A + a_off + 1, mask=mask, other=0.0)
    out_i = old_i + alpha_value * prod_i
    out_i = tl.where(rows[:, None] == cols[None, :], 0.0, out_i)
    tl.store(A + a_off, old_r + alpha_value * prod_r, mask=mask)
    tl.store(A + a_off + 1, out_i, mask=mask)


def _her_hygon_impl(
    name,
    uplo,
    n,
    alpha,
    x,
    incx,
    A,
    lda,
    dtype,
    alpha_dtype,
    kernel,
):
    _check_her_args(name, uplo, n, alpha, x, incx, A, lda, dtype, alpha_dtype)
    if n == 0:
        return A
    uplo = _row_major_uplo(uplo)

    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    is_double = dtype == torch.complex128
    kernel_alpha = _f64_to_i64(alpha_value) if is_double else alpha_value
    x_real = torch.view_as_real(x).reshape(-1)
    A_real = torch.view_as_real(A).reshape(-1)

    if is_double and uplo == 0 and n >= 3072:
        grid = (triton.cdiv(n, 64), triton.cdiv(n, 8))
        with torch_device_fn.device(A.device):
            zher_rect_hygon_kernel[grid](
                x_real,
                A_real,
                n,
                kernel_alpha,
                incx,
                lda,
                uplo,
                BLOCK_M=64,
                BLOCK_N=8,
                num_warps=4,
                num_stages=1,
            )
        return A

    def grid(meta):
        tile_count = triton.cdiv(n, meta["BLOCK_M"])
        return (tile_count * (tile_count + 1) // 2,)

    with torch_device_fn.device(A.device):
        kernel[grid](
            x_real,
            A_real,
            n,
            kernel_alpha,
            incx,
            lda,
            uplo,
            IS_DOUBLE=is_double,
        )
    return A


def cher(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    return _her_hygon_impl(
        "cher",
        uplo,
        n,
        alpha,
        x,
        incx,
        A,
        lda,
        torch.complex64,
        torch.float32,
        cher_hygon_kernel,
    )


def zher(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    return _her_hygon_impl(
        "zher",
        uplo,
        n,
        alpha,
        x,
        incx,
        A,
        lda,
        torch.complex128,
        torch.float64,
        zher_hygon_kernel,
    )


__all__ = ["cher", "zher"]
