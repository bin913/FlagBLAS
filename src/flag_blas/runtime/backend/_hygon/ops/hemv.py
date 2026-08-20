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

import struct

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2.hemv import (
    ScalarType,
    _check_common,
    _complex_scalars,
    _strided_y,
)
from flag_blas.ops.level2.hemv import chemv as _common_chemv
from flag_blas.ops.level2.hemv import zhemv as _common_zhemv
from flag_blas.ops.level2.hemv import zhemv_kernel as _common_zhemv_kernel
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner


def _f64_to_i64(value: float) -> int:
    return struct.unpack("<q", struct.pack("<d", value))[0]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("chemv_hygon"),
    key=["n", "UPLO"],
    restore_value=["y_ptr"],
)
@triton.jit
def chemv_hygon_packed_kernel(
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
    pid = tl.program_id(0)
    tiles = tl.cdiv(n, BLOCK_SIZE)
    tri_row = ((tl.sqrt(8.0 * pid + 1.0) - 1.0) * 0.5).to(tl.int32)
    tri_col = pid - tri_row * (tri_row + 1) // 2
    if UPLO == 0:
        pid_m = tiles - 1 - tri_col
        pid_n = tiles - 1 - tri_row
    else:
        pid_m = tri_col
        pid_n = tri_row

    rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < n
    col_mask = cols < n
    mask2d = row_mask[:, None] & col_mask[None, :]
    y_rows_off = rows * INCY * 2
    y_cols_off = cols * INCY * 2

    x_rows_bits = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0)
    x_cols_bits = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0)
    xrr = (x_rows_bits & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    xri = ((x_rows_bits >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    xcr = (x_cols_bits & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    xci = ((x_cols_bits >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)

    if pid_m == pid_n:
        i = rows[:, None]
        j = cols[None, :]
        if UPLO == 0:
            use_direct = j <= i
        else:
            use_direct = j >= i
        elem_off = tl.where(use_direct, i * LDA + j, j * LDA + i)
        a_bits = tl.load(a_ptr + elem_off, mask=mask2d, other=0)
        ar = (a_bits & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
        ai = ((a_bits >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
        ai = tl.where(use_direct, ai, -ai)
        ai = tl.where(i == j, 0.0, ai)
        acc_r = tl.sum(ar * xcr[None, :] - ai * xci[None, :], axis=1)
        acc_i = tl.sum(ar * xci[None, :] + ai * xcr[None, :], axis=1)
        out_r = alpha_r * acc_r - alpha_i * acc_i
        out_i = alpha_r * acc_i + alpha_i * acc_r
        tl.atomic_add(y_ptr + y_rows_off, out_r, mask=row_mask, sem="relaxed")
        tl.atomic_add(y_ptr + y_rows_off + 1, out_i, mask=row_mask, sem="relaxed")
        return

    elem_off = rows[:, None] * LDA + cols[None, :]
    a_bits = tl.load(a_ptr + elem_off, mask=mask2d, other=0)
    ar = (a_bits & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    ai = ((a_bits >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    row_r = tl.sum(ar * xcr[None, :] - ai * xci[None, :], axis=1)
    row_i = tl.sum(ar * xci[None, :] + ai * xcr[None, :], axis=1)
    col_r = tl.sum(ar * xrr[:, None] + ai * xri[:, None], axis=0)
    col_i = tl.sum(ar * xri[:, None] - ai * xrr[:, None], axis=0)
    out_row_r = alpha_r * row_r - alpha_i * row_i
    out_row_i = alpha_r * row_i + alpha_i * row_r
    out_col_r = alpha_r * col_r - alpha_i * col_i
    out_col_i = alpha_r * col_i + alpha_i * col_r
    tl.atomic_add(y_ptr + y_rows_off, out_row_r, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_rows_off + 1, out_row_i, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_cols_off, out_col_r, mask=col_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_cols_off + 1, out_col_i, mask=col_mask, sem="relaxed")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("zhemv_hygon"),
    key=["n", "UPLO"],
    restore_value=["y_ptr"],
)
@triton.jit
def zhemv_hygon_triangular_kernel(
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
    pid = tl.program_id(0)
    tiles = tl.cdiv(n, BLOCK_SIZE)
    tri_row = ((tl.sqrt(8.0 * pid + 1.0) - 1.0) * 0.5).to(tl.int32)
    tri_col = pid - tri_row * (tri_row + 1) // 2
    if UPLO == 0:
        pid_m = tiles - 1 - tri_col
        pid_n = tiles - 1 - tri_row
    else:
        pid_m = tri_col
        pid_n = tri_row

    alpha_r = alpha_r_int.to(tl.float64, bitcast=True)
    alpha_i = alpha_i_int.to(tl.float64, bitcast=True)
    rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < n
    col_mask = cols < n
    mask2d = row_mask[:, None] & col_mask[None, :]
    y_rows_off = rows * INCY * 2
    y_cols_off = cols * INCY * 2
    x_rows_off = rows * INCX * 2
    x_cols_off = cols * INCX * 2
    xrr = tl.load(x_ptr + x_rows_off, mask=row_mask, other=0.0)
    xri = tl.load(x_ptr + x_rows_off + 1, mask=row_mask, other=0.0)
    xcr = tl.load(x_ptr + x_cols_off, mask=col_mask, other=0.0)
    xci = tl.load(x_ptr + x_cols_off + 1, mask=col_mask, other=0.0)

    if pid_m == pid_n:
        i = rows[:, None]
        j = cols[None, :]
        if UPLO == 0:
            use_direct = j <= i
        else:
            use_direct = j >= i
        elem_off = tl.where(use_direct, i * LDA + j, j * LDA + i)
        a_off = elem_off * 2
        ar = tl.load(a_ptr + a_off, mask=mask2d, other=0.0)
        ai = tl.load(a_ptr + a_off + 1, mask=mask2d, other=0.0)
        ai = tl.where(use_direct, ai, -ai)
        ai = tl.where(i == j, 0.0, ai)
        acc_r = tl.sum(ar * xcr[None, :] - ai * xci[None, :], axis=1)
        acc_i = tl.sum(ar * xci[None, :] + ai * xcr[None, :], axis=1)
        out_r = alpha_r * acc_r - alpha_i * acc_i
        out_i = alpha_r * acc_i + alpha_i * acc_r
        tl.atomic_add(y_ptr + y_rows_off, out_r, mask=row_mask, sem="relaxed")
        tl.atomic_add(y_ptr + y_rows_off + 1, out_i, mask=row_mask, sem="relaxed")
        return

    elem_off = rows[:, None] * LDA + cols[None, :]
    a_off = elem_off * 2
    ar = tl.load(a_ptr + a_off, mask=mask2d, other=0.0)
    ai = tl.load(a_ptr + a_off + 1, mask=mask2d, other=0.0)
    row_r = tl.sum(ar * xcr[None, :] - ai * xci[None, :], axis=1)
    row_i = tl.sum(ar * xci[None, :] + ai * xcr[None, :], axis=1)
    col_r = tl.sum(ar * xrr[:, None] + ai * xri[:, None], axis=0)
    col_i = tl.sum(ar * xri[:, None] - ai * xrr[:, None], axis=0)
    out_row_r = alpha_r * row_r - alpha_i * row_i
    out_row_i = alpha_r * row_i + alpha_i * row_r
    out_col_r = alpha_r * col_r - alpha_i * col_i
    out_col_i = alpha_r * col_i + alpha_i * col_r
    tl.atomic_add(y_ptr + y_rows_off, out_row_r, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_rows_off + 1, out_row_i, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_cols_off, out_col_r, mask=col_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_cols_off + 1, out_col_i, mask=col_mask, sem="relaxed")


@triton.jit
def chemv_swizzle_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r,
    alpha_i,
    n,
    lda,
    incx,
    incy,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles = tl.cdiv(n, BLOCK_SIZE)
    num_pid_in_group = GROUP_SIZE_M * tiles
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(tiles - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
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
    mask2d = row_mask[:, None] & col_mask[None, :]
    y_rows_off = rows * incy * 2
    y_cols_off = cols * incy * 2
    x_rows_bits = tl.load(x_ptr + rows * incx, mask=row_mask, other=0)
    x_cols_bits = tl.load(x_ptr + cols * incx, mask=col_mask, other=0)
    xrr = (x_rows_bits & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    xri = ((x_rows_bits >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    xcr = (x_cols_bits & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    xci = ((x_cols_bits >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)

    if pid_m == pid_n:
        i = rows[:, None]
        j = cols[None, :]
        if UPLO == 0:
            use_direct = j <= i
        else:
            use_direct = j >= i
        elem_off = tl.where(use_direct, i * lda + j, j * lda + i)
        a_bits = tl.load(a_ptr + elem_off, mask=mask2d, other=0)
        ar = (a_bits & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
        ai = ((a_bits >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
        ai = tl.where(use_direct, ai, -ai)
        ai = tl.where(i == j, 0.0, ai)
        acc_r = tl.sum(ar * xcr[None, :] - ai * xci[None, :], axis=1)
        acc_i = tl.sum(ar * xci[None, :] + ai * xcr[None, :], axis=1)
        out_r = alpha_r * acc_r - alpha_i * acc_i
        out_i = alpha_r * acc_i + alpha_i * acc_r
        tl.atomic_add(y_ptr + y_rows_off, out_r, mask=row_mask, sem="relaxed")
        tl.atomic_add(y_ptr + y_rows_off + 1, out_i, mask=row_mask, sem="relaxed")
        return

    elem_off = rows[:, None] * lda + cols[None, :]
    a_bits = tl.load(a_ptr + elem_off, mask=mask2d, other=0)
    ar = (a_bits & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    ai = ((a_bits >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    row_r = tl.sum(ar * xcr[None, :] - ai * xci[None, :], axis=1)
    row_i = tl.sum(ar * xci[None, :] + ai * xcr[None, :], axis=1)
    col_r = tl.sum(ar * xrr[:, None] + ai * xri[:, None], axis=0)
    col_i = tl.sum(ar * xri[:, None] - ai * xrr[:, None], axis=0)
    out_row_r = alpha_r * row_r - alpha_i * row_i
    out_row_i = alpha_r * row_i + alpha_i * row_r
    out_col_r = alpha_r * col_r - alpha_i * col_i
    out_col_i = alpha_r * col_i + alpha_i * col_r
    tl.atomic_add(y_ptr + y_rows_off, out_row_r, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_rows_off + 1, out_row_i, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_cols_off, out_col_r, mask=col_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_cols_off + 1, out_col_i, mask=col_mask, sem="relaxed")


chemv_hygon_swizzle_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("chemv_hygon_swizzle"),
        key=["n", "UPLO"],
        restore_value=["y_ptr"],
    )(chemv_swizzle_kernel)
)

zhemv_hygon_swizzle_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("zhemv_hygon_swizzle"),
        key=["n", "UPLO"],
        restore_value=["y_ptr"],
    )(_common_zhemv_kernel.fn)
)


def _scale_y(y_view, beta_r, beta_i):
    if beta_r == 0.0 and beta_i == 0.0:
        y_view.zero_()
    elif beta_r != 1.0 or beta_i != 0.0:
        y_view.mul_(complex(beta_r, beta_i))


def chemv(
    uplo: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    use_hygon_kernel = 0 < n <= 512 or n >= 3072
    if not use_hygon_kernel:
        return _common_chemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    assert A.dtype == torch.complex64 == x.dtype == y.dtype
    _check_common(A, x, y, uplo, n, lda, incx, incy)
    ar, ai, br, bi = _complex_scalars(alpha, beta)
    if ar == 0.0 and ai == 0.0:
        return _common_chemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    y_view = _strided_y(y, n, incy)
    y_real = torch.view_as_real(y)

    def grid(meta):
        tiles = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (tiles * (tiles + 1) // 2,)

    with torch_device_fn.device(A.device):
        _scale_y(y_view, br, bi)
        if n >= 3072:

            def swizzle_grid(meta):
                tiles = triton.cdiv(n, meta["BLOCK_SIZE"])
                return (tiles * tiles,)

            chemv_hygon_swizzle_kernel[swizzle_grid](
                A.view(torch.int64),
                x.view(torch.int64),
                y_real,
                ar,
                ai,
                n,
                lda,
                incx,
                incy,
                UPLO=uplo,
            )
            return

        chemv_hygon_packed_kernel[grid](
            A.view(torch.int64),
            x.view(torch.int64),
            y_real,
            ar,
            ai,
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
        )


def zhemv(
    uplo: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    if 1024 < n < 4096:
        return _common_zhemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    assert A.dtype == torch.complex128 == x.dtype == y.dtype
    _check_common(A, x, y, uplo, n, lda, incx, incy)
    if n == 0:
        return
    ar, ai, br, bi = _complex_scalars(alpha, beta)
    if ar == 0.0 and ai == 0.0:
        return _common_zhemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    y_view = _strided_y(y, n, incy)
    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    def grid(meta):
        tiles = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (tiles * (tiles + 1) // 2,)

    with torch_device_fn.device(A.device):
        _scale_y(y_view, br, bi)
        if n >= 4096:

            def swizzle_grid(meta):
                tiles = triton.cdiv(n, meta["BLOCK_SIZE"])
                return (tiles * tiles,)

            zhemv_hygon_swizzle_kernel[swizzle_grid](
                A_real,
                x_real,
                y_real,
                _f64_to_i64(ar),
                _f64_to_i64(ai),
                n,
                lda,
                incx,
                incy,
                UPLO=uplo,
            )
            return

        zhemv_hygon_triangular_kernel[grid](
            A_real,
            x_real,
            y_real,
            _f64_to_i64(ar),
            _f64_to_i64(ai),
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
        )


__all__ = ["chemv", "zhemv"]
