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
from flag_blas.ops.level2._constants import (
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_DIAG_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
)
from flag_blas.ops.level2.trmv import _check_trmv, _mode_key
from flag_blas.ops.level2.trmv import ctrmv as common_ctrmv
from flag_blas.ops.level2.trmv import dtrmv as common_dtrmv
from flag_blas.ops.level2.trmv import strmv as common_strmv
from flag_blas.ops.level2.trmv import ztrmv as common_ztrmv
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner
from flag_blas.utils.libentry import LibTuner

_HYGON_TRMV_KEY = ["n", "mode_key"]
_HYGON_TRMV_RESTORE = ["x_ptr"]
_HYGON_TRMV_TRANS_KEY = ["n", "mode_key", "FP64"]


@LibTuner.register_policy("hygon_trmv_stable")
def _hygon_trmv_stable_policy(bench_fn, configs, args, kwargs):
    timings = {config: bench_fn(config) for config in configs}
    best_config = min(timings, key=lambda config: timings[config][-1])
    return best_config, timings


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("real_trmv_n_lane_hygon"),
    key=_HYGON_TRMV_KEY,
    restore_value=_HYGON_TRMV_RESTORE,
)
@triton.jit
def real_trmv_n_lane_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    LDA,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    lanes = tl.arange(0, BLOCK_K)
    row_mask = rows < n
    if FP64:
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    if UPLO == 1:
        active_lo = row_start
        active_hi = n
    else:
        active_lo = 0
        active_hi = row_start + BLOCK_M

    for kb in tl.range(active_lo, active_hi, BLOCK_K):
        cols = kb + lanes
        col_mask = cols < n
        if UPLO == 1:
            tri = cols[None, :] >= rows[:, None]
        else:
            tri = cols[None, :] <= rows[:, None]
        if UNIT:
            tri = tri & (cols[None, :] != rows[:, None])
        mask = row_mask[:, None] & col_mask[None, :] & tri
        a = tl.load(
            a_ptr + rows[:, None] * LDA + cols[None, :],
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        xin = tl.load(
            xin_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        acc += a * xin[None, :]

    out = tl.sum(acc, axis=1)
    if UNIT:
        out += tl.load(xin_ptr + rows, mask=row_mask, other=0.0)
    tl.store(x_ptr + rows, out, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("real_trmv_row_hygon"),
    key=_HYGON_TRMV_KEY,
    restore_value=_HYGON_TRMV_RESTORE,
)
@triton.jit
def real_trmv_row_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    LDA,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    if FP64:
        acc = tl.zeros((BLOCK_K,), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    if UPLO == 1:
        active_lo = 0
        active_hi = row if UNIT else row + 1
    else:
        active_lo = row + 1 if UNIT else row
        active_hi = n

    for kb in tl.range(active_lo, active_hi, BLOCK_K):
        cols = kb + offs
        mask = cols < active_hi
        a = tl.load(
            a_ptr + cols * LDA + row,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        xin = tl.load(
            xin_ptr + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        acc += a * xin

    out = tl.sum(acc, axis=0)
    if UNIT:
        out += tl.load(xin_ptr + row)
    tl.store(x_ptr + row, out)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("real_trmv_trans_gemv_hygon"),
    key=_HYGON_TRMV_TRANS_KEY,
    restore_value=_HYGON_TRMV_RESTORE,
    policy="hygon_trmv_stable",
)
@triton.jit
def real_trmv_trans_gemv_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    LDA,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < n
    offs_k = tl.arange(0, BLOCK_K)
    diag_start = (row_start // BLOCK_K) * BLOCK_K
    if FP64:
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    if UPLO == 1:
        a_ptrs = a_ptr + rows[:, None] + offs_k[None, :] * LDA
        x_ptrs = xin_ptr + offs_k
        for kb in tl.range(0, diag_start, BLOCK_K):
            cols = kb + offs_k
            col_mask = cols < n
            mask = row_mask[:, None] & col_mask[None, :]
            a = tl.load(a_ptrs, mask=mask, other=0.0, eviction_policy="evict_first")
            xin = tl.load(
                x_ptrs, mask=col_mask, other=0.0, eviction_policy="evict_last"
            )
            acc += a * xin[None, :]
            a_ptrs += BLOCK_K * LDA
            x_ptrs += BLOCK_K

    cols = diag_start + offs_k
    col_mask = cols < n
    if UPLO == 1:
        tri = cols[None, :] <= rows[:, None]
    else:
        tri = cols[None, :] >= rows[:, None]
    if UNIT:
        tri = tri & (cols[None, :] != rows[:, None])
    mask = row_mask[:, None] & col_mask[None, :] & tri
    a = tl.load(
        a_ptr + cols[None, :] * LDA + rows[:, None],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    xin = tl.load(
        xin_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    )
    acc += a * xin[None, :]

    if UPLO == 0:
        post_start = diag_start + BLOCK_K
        a_ptrs = a_ptr + rows[:, None] + (post_start + offs_k)[None, :] * LDA
        x_ptrs = xin_ptr + post_start + offs_k
        for kb in tl.range(post_start, n, BLOCK_K):
            cols = kb + offs_k
            col_mask = cols < n
            mask = row_mask[:, None] & col_mask[None, :]
            a = tl.load(a_ptrs, mask=mask, other=0.0, eviction_policy="evict_first")
            xin = tl.load(
                x_ptrs, mask=col_mask, other=0.0, eviction_policy="evict_last"
            )
            acc += a * xin[None, :]
            a_ptrs += BLOCK_K * LDA
            x_ptrs += BLOCK_K

    out = tl.sum(acc, axis=1)
    if UNIT:
        out += tl.load(xin_ptr + rows, mask=row_mask, other=0.0)
    tl.store(x_ptr + rows, out, mask=row_mask)


def _real_trmv(
    common_op,
    dtype,
    fp64,
    min_n,
    uplo,
    trans,
    diag,
    n,
    A,
    lda,
    x,
    incx,
):
    use_trans_gemv_kernel = incx == 1 and trans != CUBLAS_OP_N and n >= min_n
    use_n_lane_kernel = (
        incx == 1
        and trans == CUBLAS_OP_N
        and (
            (fp64 and 1023 <= n < 16384)
            or (
                not fp64
                and (
                    (
                        diag == CUBLAS_DIAG_NON_UNIT
                        and uplo == CUBLAS_FILL_MODE_LOWER
                        and 127 <= n < 512
                    )
                    or (diag == CUBLAS_DIAG_UNIT and n == 127)
                )
            )
        )
    )
    if not use_trans_gemv_kernel and not use_n_lane_kernel:
        common_op(uplo, trans, diag, n, A, lda, x, incx)
        return

    assert A.dtype == dtype == x.dtype
    _check_trmv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=False)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    with torch_device_fn.device(A.device):
        xin = x.clone()
        if use_n_lane_kernel:

            def n_lane_grid(meta):
                return (triton.cdiv(n, meta["BLOCK_M"]),)

            real_trmv_n_lane_kernel[n_lane_grid](
                A,
                xin,
                x,
                n,
                lda,
                _mode_key(uplo, 0, unit),
                UPLO=uplo,
                UNIT=unit,
                FP64=fp64,
            )
            return

        def trans_grid(meta):
            return (triton.cdiv(n, meta["BLOCK_M"]),)

        real_trmv_trans_gemv_kernel[trans_grid](
            A,
            xin,
            x,
            n,
            lda,
            _mode_key(uplo, 1, unit),
            UPLO=uplo,
            UNIT=unit,
            FP64=fp64,
        )


def strmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    _real_trmv(
        common_strmv,
        torch.float32,
        False,
        127,
        uplo,
        trans,
        diag,
        n,
        A,
        lda,
        x,
        incx,
    )


def dtrmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    _real_trmv(
        common_dtrmv,
        torch.float64,
        True,
        127,
        uplo,
        trans,
        diag,
        n,
        A,
        lda,
        x,
        incx,
    )


@triton.jit
def ctrmv_atomic_init_kernel(
    xin_ptr,
    x_ptr,
    n,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n
    acc_r = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if UNIT:
        xin = tl.load(xin_ptr + offs, mask=mask, other=0)
        acc_r = xin.to(tl.int32).to(tl.float32, bitcast=True)
        acc_i = (xin >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    tl.store(x_ptr + offs * 2, acc_r, mask=mask)
    tl.store(x_ptr + offs * 2 + 1, acc_i, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ctrmv_row_hygon"),
    key=_HYGON_TRMV_KEY,
    restore_value=_HYGON_TRMV_RESTORE,
)
@triton.jit
def ctrmv_row_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    LDA,
    mode_key,
    UPLO: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    acc_r = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_K,), dtype=tl.float32)
    if UPLO == 1:
        active_lo = 0
        active_hi = row + 1
    else:
        active_lo = row
        active_hi = n

    for kb in tl.range(active_lo, active_hi, BLOCK_K):
        cols = kb + offs
        mask = cols < active_hi
        a = tl.load(
            a_ptr + cols * LDA + row,
            mask=mask,
            other=0,
            eviction_policy="evict_first",
        )
        xin = tl.load(
            xin_ptr + cols,
            mask=mask,
            other=0,
            eviction_policy="evict_last",
        )
        ar = a.to(tl.int32).to(tl.float32, bitcast=True)
        ai = (a >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        xr = xin.to(tl.int32).to(tl.float32, bitcast=True)
        xi = (xin >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        if CONJ:
            ai = -ai
        acc_r += ar * xr - ai * xi
        acc_i += ar * xi + ai * xr

    tl.store(x_ptr + row * 2, tl.sum(acc_r, axis=0))
    tl.store(x_ptr + row * 2 + 1, tl.sum(acc_i, axis=0))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ctrmv_n_lane_hygon"),
    key=_HYGON_TRMV_KEY,
    restore_value=_HYGON_TRMV_RESTORE,
)
@triton.jit
def ctrmv_n_lane_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    LDA,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    lanes = tl.arange(0, BLOCK_K)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    if UPLO == 1:
        active_lo = row_start
        active_hi = n
    else:
        active_lo = 0
        active_hi = row_start + BLOCK_M

    for kb in tl.range(active_lo, active_hi, BLOCK_K):
        cols = kb + lanes
        col_mask = cols < n
        if UPLO == 1:
            tri = cols[None, :] >= rows[:, None]
        else:
            tri = cols[None, :] <= rows[:, None]
        if UNIT:
            tri = tri & (cols[None, :] != rows[:, None])
        mask = row_mask[:, None] & col_mask[None, :] & tri
        a = tl.load(
            a_ptr + rows[:, None] * LDA + cols[None, :],
            mask=mask,
            other=0,
            eviction_policy="evict_first",
        )
        xin = tl.load(
            xin_ptr + cols,
            mask=col_mask,
            other=0,
            eviction_policy="evict_last",
        )
        ar = a.to(tl.int32).to(tl.float32, bitcast=True)
        ai = (a >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        xr = xin.to(tl.int32).to(tl.float32, bitcast=True)
        xi = (xin >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        acc_r += ar * xr[None, :] - ai * xi[None, :]
        acc_i += ar * xi[None, :] + ai * xr[None, :]

    out_r = tl.sum(acc_r, axis=1)
    out_i = tl.sum(acc_i, axis=1)
    if UNIT:
        diag_x = tl.load(xin_ptr + rows, mask=row_mask, other=0)
        out_r += diag_x.to(tl.int32).to(tl.float32, bitcast=True)
        out_i += (diag_x >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    tl.store(x_ptr + rows * 2, out_r, mask=row_mask)
    tl.store(x_ptr + rows * 2 + 1, out_i, mask=row_mask)


@triton.jit
def ctrmv_atomic_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    LDA,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    if UPLO == TRANS:
        if pid_k > pid_m:
            return
    else:
        if pid_k < pid_m:
            return

    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M)
    cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK_K), BLOCK_K)
    row_mask = rows < n
    col_mask = cols < n

    if TRANS == 0:
        if UPLO == 1:
            tri = cols[None, :] >= rows[:, None]
        else:
            tri = cols[None, :] <= rows[:, None]
        elem_off = rows[:, None] * LDA + cols[None, :]
    else:
        if UPLO == 1:
            tri = cols[None, :] <= rows[:, None]
        else:
            tri = cols[None, :] >= rows[:, None]
        elem_off = cols[None, :] * LDA + rows[:, None]
    if UNIT:
        tri = tri & (cols[None, :] != rows[:, None])
    mask = row_mask[:, None] & col_mask[None, :] & tri

    a = tl.load(a_ptr + elem_off, mask=mask, other=0, eviction_policy="evict_first")
    xin = tl.load(xin_ptr + cols, mask=col_mask, other=0, eviction_policy="evict_last")
    ar = a.to(tl.int32).to(tl.float32, bitcast=True)
    ai = (a >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    xr = xin.to(tl.int32).to(tl.float32, bitcast=True)
    xi = (xin >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    if CONJ:
        ai = -ai
    acc_r = tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
    acc_i = tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)
    tl.atomic_add(x_ptr + rows * 2, acc_r, mask=row_mask, sem="relaxed")
    tl.atomic_add(x_ptr + rows * 2 + 1, acc_i, mask=row_mask, sem="relaxed")


@triton.jit
def ctrmv_splitk_kernel(
    a_ptr,
    xin_ptr,
    partial_ptr,
    n,
    LDA,
    SPLIT_K: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    row_start = pid_m * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M)
    row_mask = rows < n
    offs_k = tl.arange(0, BLOCK_K)
    acc_r_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_K), dtype=tl.float32)
    acc_i_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_K), dtype=tl.float32)

    if UPLO != TRANS:
        active_lo = row_start
        active_hi = n
    else:
        active_lo = 0
        active_hi = row_start + BLOCK_SIZE_M
    active_len = active_hi - active_lo
    total_tiles = (active_len + BLOCK_K - 1) // BLOCK_K
    tiles_per_chunk = (total_tiles + SPLIT_K - 1) // SPLIT_K
    my_tile_lo = pid_k * tiles_per_chunk
    my_tile_hi = tl.minimum((pid_k + 1) * tiles_per_chunk, total_tiles)
    my_lo = active_lo + my_tile_lo * BLOCK_K
    my_hi = active_lo + my_tile_hi * BLOCK_K

    for kb in tl.range(my_lo, my_hi, BLOCK_K):
        cols = kb + offs_k
        cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK_K), BLOCK_K)
        col_mask = cols < n
        if TRANS == 0:
            if UPLO == 1:
                tri = cols[None, :] >= rows[:, None]
            else:
                tri = cols[None, :] <= rows[:, None]
            elem_off = rows[:, None] * LDA + cols[None, :]
        else:
            if UPLO == 1:
                tri = cols[None, :] <= rows[:, None]
            else:
                tri = cols[None, :] >= rows[:, None]
            elem_off = cols[None, :] * LDA + rows[:, None]
        if UNIT:
            tri = tri & (cols[None, :] != rows[:, None])
        mask = row_mask[:, None] & col_mask[None, :] & tri
        a = tl.load(
            a_ptr + elem_off,
            mask=mask,
            other=0,
            eviction_policy="evict_first",
        )
        xin = tl.load(
            xin_ptr + cols,
            mask=col_mask,
            other=0,
            eviction_policy="evict_last",
        )
        ar = a.to(tl.int32).to(tl.float32, bitcast=True)
        ai = (a >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        xr = xin.to(tl.int32).to(tl.float32, bitcast=True)
        xi = (xin >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        if CONJ:
            ai = -ai
        acc_r_2d += ar * xr[None, :] - ai * xi[None, :]
        acc_i_2d += ar * xi[None, :] + ai * xr[None, :]

    acc_r = tl.sum(acc_r_2d, axis=1)
    acc_i = tl.sum(acc_i_2d, axis=1)
    out_off = pid_k * n * 2 + rows * 2
    tl.store(partial_ptr + out_off, acc_r, mask=row_mask)
    tl.store(partial_ptr + out_off + 1, acc_i, mask=row_mask)


@triton.jit
def ctrmv_splitk_reduce_kernel(
    partial_ptr,
    xin_ptr,
    x_ptr,
    n,
    SPLIT_K: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n
    acc_r = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in tl.static_range(0, SPLIT_K):
        acc_r += tl.load(partial_ptr + k * n * 2 + offs * 2, mask=mask, other=0.0)
        acc_i += tl.load(partial_ptr + k * n * 2 + offs * 2 + 1, mask=mask, other=0.0)
    if UNIT:
        xin = tl.load(xin_ptr + offs, mask=mask, other=0)
        acc_r += xin.to(tl.int32).to(tl.float32, bitcast=True)
        acc_i += (xin >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    tl.store(x_ptr + offs * 2, acc_r, mask=mask)
    tl.store(x_ptr + offs * 2 + 1, acc_i, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ztrmv_n_lane_hygon"),
    key=_HYGON_TRMV_KEY,
    restore_value=_HYGON_TRMV_RESTORE,
)
@triton.jit
def ztrmv_n_lane_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    LDA,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    lanes = tl.arange(0, BLOCK_K)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_SIZE_M, BLOCK_K), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_SIZE_M, BLOCK_K), dtype=tl.float64)
    if UPLO == 1:
        active_lo = row_start
        active_hi = n
    else:
        active_lo = 0
        active_hi = row_start + BLOCK_SIZE_M

    for kb in tl.range(active_lo, active_hi, BLOCK_K):
        cols = kb + lanes
        col_mask = cols < n
        if UPLO == 1:
            tri = cols[None, :] >= rows[:, None]
        else:
            tri = cols[None, :] <= rows[:, None]
        if UNIT:
            tri = tri & (cols[None, :] != rows[:, None])
        mask = row_mask[:, None] & col_mask[None, :] & tri
        elem_off = (rows[:, None] * LDA + cols[None, :]) * 2
        ar = tl.load(
            a_ptr + elem_off,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        ai = tl.load(
            a_ptr + elem_off + 1,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        xr = tl.load(
            xin_ptr + cols * 2,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        xi = tl.load(
            xin_ptr + cols * 2 + 1,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        acc_r += ar * xr[None, :] - ai * xi[None, :]
        acc_i += ar * xi[None, :] + ai * xr[None, :]

    out_r = tl.sum(acc_r, axis=1)
    out_i = tl.sum(acc_i, axis=1)
    if UNIT:
        out_r += tl.load(xin_ptr + rows * 2, mask=row_mask, other=0.0)
        out_i += tl.load(xin_ptr + rows * 2 + 1, mask=row_mask, other=0.0)
    tl.store(x_ptr + rows * 2, out_r, mask=row_mask)
    tl.store(x_ptr + rows * 2 + 1, out_i, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ztrmv_row_hygon"),
    key=_HYGON_TRMV_KEY,
    restore_value=_HYGON_TRMV_RESTORE,
    policy="hygon_trmv_stable",
)
@triton.jit
def ztrmv_row_nonunit_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    LDA,
    mode_key,
    UPLO: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    acc_r = tl.zeros((BLOCK_K,), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_K,), dtype=tl.float64)
    if UPLO == 1:
        active_lo = 0
        active_hi = row + 1
    else:
        active_lo = row
        active_hi = n

    for kb in tl.range(active_lo, active_hi, BLOCK_K):
        cols = kb + offs
        mask = cols < active_hi
        elem_off = (cols * LDA + row) * 2
        ar = tl.load(
            a_ptr + elem_off,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        ai = tl.load(
            a_ptr + elem_off + 1,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        xr = tl.load(
            xin_ptr + cols * 2,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        xi = tl.load(
            xin_ptr + cols * 2 + 1,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        if CONJ:
            ai = -ai
        acc_r += ar * xr - ai * xi
        acc_i += ar * xi + ai * xr

    tl.store(x_ptr + row * 2, tl.sum(acc_r, axis=0))
    tl.store(x_ptr + row * 2 + 1, tl.sum(acc_i, axis=0))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ztrmv_row_hygon"),
    key=_HYGON_TRMV_KEY,
    restore_value=_HYGON_TRMV_RESTORE,
    policy="hygon_trmv_stable",
)
@triton.jit
def ztrmv_row_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    LDA,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    acc_r = tl.zeros((BLOCK_K,), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_K,), dtype=tl.float64)
    if UNIT:
        if UPLO == 1:
            active_lo = 0
            active_hi = row
        else:
            active_lo = row + 1
            active_hi = n
    else:
        if UPLO == 1:
            active_lo = 0
            active_hi = row + 1
        else:
            active_lo = row
            active_hi = n

    for kb in tl.range(active_lo, active_hi, BLOCK_K):
        cols = kb + offs
        mask = cols < active_hi
        elem_off = (cols * LDA + row) * 2
        ar = tl.load(
            a_ptr + elem_off,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        ai = tl.load(
            a_ptr + elem_off + 1,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        xr = tl.load(
            xin_ptr + cols * 2,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        xi = tl.load(
            xin_ptr + cols * 2 + 1,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        if CONJ:
            ai = -ai
        acc_r += ar * xr - ai * xi
        acc_i += ar * xi + ai * xr

    out_r = tl.sum(acc_r, axis=0)
    out_i = tl.sum(acc_i, axis=0)
    if UNIT:
        out_r += tl.load(xin_ptr + row * 2)
        out_i += tl.load(xin_ptr + row * 2 + 1)
    tl.store(x_ptr + row * 2, out_r)
    tl.store(x_ptr + row * 2 + 1, out_i)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("complex_trmv_n_gemv_hygon"),
    key=_HYGON_TRMV_TRANS_KEY,
    restore_value=_HYGON_TRMV_RESTORE,
    policy="hygon_trmv_stable",
)
@triton.jit
def complex_trmv_n_gemv_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    lda,
    mode_key,
    UPLO: tl.constexpr,
    FP64: tl.constexpr,
    UNIT: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS_PER_CTA
    rows = row_start + tl.arange(0, ROWS_PER_CTA)
    row_mask = rows < n
    offs_k = tl.arange(0, BLOCK_K)
    diag_start = (row_start // BLOCK_K) * BLOCK_K
    if FP64:
        acc_r = tl.zeros((ROWS_PER_CTA, BLOCK_K), dtype=tl.float64)
        acc_i = tl.zeros((ROWS_PER_CTA, BLOCK_K), dtype=tl.float64)
    else:
        acc_r = tl.zeros((ROWS_PER_CTA, BLOCK_K), dtype=tl.float32)
        acc_i = tl.zeros((ROWS_PER_CTA, BLOCK_K), dtype=tl.float32)

    if UPLO == 0:
        a_offsets = (rows[:, None] * lda + offs_k[None, :]) * 2
        x_offsets = offs_k * 2
        for kb in tl.range(0, diag_start, BLOCK_K):
            cols = kb + offs_k
            col_mask = cols < n
            mask = row_mask[:, None] & col_mask[None, :]
            ar = tl.load(
                a_ptr + a_offsets, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            ai = tl.load(
                a_ptr + a_offsets + 1,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
            xr = tl.load(
                xin_ptr + x_offsets,
                mask=col_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            xi = tl.load(
                xin_ptr + x_offsets + 1,
                mask=col_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            acc_r += ar * xr[None, :] - ai * xi[None, :]
            acc_i += ar * xi[None, :] + ai * xr[None, :]
            a_offsets += BLOCK_K * 2
            x_offsets += BLOCK_K * 2

    cols = diag_start + offs_k
    col_mask = cols < n
    if UPLO == 1:
        tri = cols[None, :] >= rows[:, None]
    else:
        tri = cols[None, :] <= rows[:, None]
    if UNIT:
        tri = tri & (cols[None, :] != rows[:, None])
    mask = row_mask[:, None] & col_mask[None, :] & tri
    a_offsets = (rows[:, None] * lda + cols[None, :]) * 2
    ar = tl.load(a_ptr + a_offsets, mask=mask, other=0.0, eviction_policy="evict_first")
    ai = tl.load(
        a_ptr + a_offsets + 1, mask=mask, other=0.0, eviction_policy="evict_first"
    )
    xr = tl.load(
        xin_ptr + cols * 2, mask=col_mask, other=0.0, eviction_policy="evict_last"
    )
    xi = tl.load(
        xin_ptr + cols * 2 + 1, mask=col_mask, other=0.0, eviction_policy="evict_last"
    )
    acc_r += ar * xr[None, :] - ai * xi[None, :]
    acc_i += ar * xi[None, :] + ai * xr[None, :]

    if UPLO == 1:
        post_start = diag_start + BLOCK_K
        a_offsets = (rows[:, None] * lda + post_start + offs_k[None, :]) * 2
        x_offsets = (post_start + offs_k) * 2
        for kb in tl.range(post_start, n, BLOCK_K):
            cols = kb + offs_k
            col_mask = cols < n
            mask = row_mask[:, None] & col_mask[None, :]
            ar = tl.load(
                a_ptr + a_offsets, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            ai = tl.load(
                a_ptr + a_offsets + 1,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
            xr = tl.load(
                xin_ptr + x_offsets,
                mask=col_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            xi = tl.load(
                xin_ptr + x_offsets + 1,
                mask=col_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            acc_r += ar * xr[None, :] - ai * xi[None, :]
            acc_i += ar * xi[None, :] + ai * xr[None, :]
            a_offsets += BLOCK_K * 2
            x_offsets += BLOCK_K * 2

    out_r = tl.sum(acc_r, axis=1)
    out_i = tl.sum(acc_i, axis=1)
    if UNIT:
        out_r += tl.load(xin_ptr + rows * 2, mask=row_mask, other=0.0)
        out_i += tl.load(xin_ptr + rows * 2 + 1, mask=row_mask, other=0.0)
    tl.store(x_ptr + rows * 2, out_r, mask=row_mask)
    tl.store(x_ptr + rows * 2 + 1, out_i, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ctrmv_trans_packed_hygon"),
    key=_HYGON_TRMV_KEY,
    restore_value=_HYGON_TRMV_RESTORE,
    policy="hygon_trmv_stable",
)
@triton.jit
def ctrmv_trans_packed_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    lda,
    mode_key,
    UPLO: tl.constexpr,
    CONJ: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < n
    offs_k = tl.arange(0, BLOCK_K)
    diag_start = (row_start // BLOCK_K) * BLOCK_K
    acc_r = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    if UPLO == 1:
        a_offsets = rows[:, None] + offs_k[None, :] * lda
        x_offsets = offs_k
        for kb in tl.range(0, diag_start, BLOCK_K):
            cols = kb + offs_k
            col_mask = cols < n
            mask = row_mask[:, None] & col_mask[None, :]
            av = tl.load(
                a_ptr + a_offsets, mask=mask, other=0, eviction_policy="evict_first"
            )
            xv = tl.load(
                xin_ptr + x_offsets,
                mask=col_mask,
                other=0,
                eviction_policy="evict_last",
            )
            ar = av.to(tl.int32).to(tl.float32, bitcast=True)
            ai = (av >> 32).to(tl.int32).to(tl.float32, bitcast=True)
            xr = xv.to(tl.int32).to(tl.float32, bitcast=True)
            xi = (xv >> 32).to(tl.int32).to(tl.float32, bitcast=True)
            if CONJ:
                ai = -ai
            acc_r += ar * xr[None, :] - ai * xi[None, :]
            acc_i += ar * xi[None, :] + ai * xr[None, :]
            a_offsets += BLOCK_K * lda
            x_offsets += BLOCK_K

    cols = diag_start + offs_k
    col_mask = cols < n
    tri = (
        cols[None, :] <= rows[:, None] if UPLO == 1 else cols[None, :] >= rows[:, None]
    )
    if UNIT:
        tri = tri & (cols[None, :] != rows[:, None])
    mask = row_mask[:, None] & col_mask[None, :] & tri
    av = tl.load(
        a_ptr + cols[None, :] * lda + rows[:, None],
        mask=mask,
        other=0,
        eviction_policy="evict_first",
    )
    xv = tl.load(xin_ptr + cols, mask=col_mask, other=0, eviction_policy="evict_last")
    ar = av.to(tl.int32).to(tl.float32, bitcast=True)
    ai = (av >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    xr = xv.to(tl.int32).to(tl.float32, bitcast=True)
    xi = (xv >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    if CONJ:
        ai = -ai
    acc_r += ar * xr[None, :] - ai * xi[None, :]
    acc_i += ar * xi[None, :] + ai * xr[None, :]

    if UPLO == 0:
        post = diag_start + BLOCK_K
        a_offsets = rows[:, None] + (post + offs_k)[None, :] * lda
        x_offsets = post + offs_k
        for kb in tl.range(post, n, BLOCK_K):
            cols = kb + offs_k
            col_mask = cols < n
            mask = row_mask[:, None] & col_mask[None, :]
            av = tl.load(
                a_ptr + a_offsets, mask=mask, other=0, eviction_policy="evict_first"
            )
            xv = tl.load(
                xin_ptr + x_offsets,
                mask=col_mask,
                other=0,
                eviction_policy="evict_last",
            )
            ar = av.to(tl.int32).to(tl.float32, bitcast=True)
            ai = (av >> 32).to(tl.int32).to(tl.float32, bitcast=True)
            xr = xv.to(tl.int32).to(tl.float32, bitcast=True)
            xi = (xv >> 32).to(tl.int32).to(tl.float32, bitcast=True)
            if CONJ:
                ai = -ai
            acc_r += ar * xr[None, :] - ai * xi[None, :]
            acc_i += ar * xi[None, :] + ai * xr[None, :]
            a_offsets += BLOCK_K * lda
            x_offsets += BLOCK_K

    out_r = tl.sum(acc_r, axis=1)
    out_i = tl.sum(acc_i, axis=1)
    if UNIT:
        diag_x = tl.load(xin_ptr + rows, mask=row_mask, other=0)
        out_r += diag_x.to(tl.int32).to(tl.float32, bitcast=True)
        out_i += (diag_x >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    tl.store(x_ptr + rows * 2, out_r, mask=row_mask)
    tl.store(x_ptr + rows * 2 + 1, out_i, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("complex_trmv_trans_gemv_hygon"),
    key=_HYGON_TRMV_TRANS_KEY,
    restore_value=_HYGON_TRMV_RESTORE,
    policy="hygon_trmv_stable",
)
@triton.jit
def complex_trmv_trans_gemv_kernel(
    a_ptr,
    xin_ptr,
    x_ptr,
    n,
    LDA,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if UPLO == 1:
        pid = tl.num_programs(0) - 1 - pid
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < n
    offs_k = tl.arange(0, BLOCK_K)
    diag_start = (row_start // BLOCK_K) * BLOCK_K
    if FP64:
        acc_r = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float64)
        acc_i = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float64)
    else:
        acc_r = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        acc_i = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    if UPLO == 1:
        a_offsets = (rows[:, None] + offs_k[None, :] * LDA) * 2
        x_offsets = offs_k * 2
        for kb in tl.range(0, diag_start, BLOCK_K):
            cols = kb + offs_k
            col_mask = cols < n
            mask = row_mask[:, None] & col_mask[None, :]
            ar = tl.load(
                a_ptr + a_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
            ai = tl.load(
                a_ptr + a_offsets + 1,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
            xr = tl.load(
                xin_ptr + x_offsets,
                mask=col_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            xi = tl.load(
                xin_ptr + x_offsets + 1,
                mask=col_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if CONJ:
                ai = -ai
            acc_r += ar * xr[None, :] - ai * xi[None, :]
            acc_i += ar * xi[None, :] + ai * xr[None, :]
            a_offsets += BLOCK_K * LDA * 2
            x_offsets += BLOCK_K * 2

    cols = diag_start + offs_k
    col_mask = cols < n
    if UPLO == 1:
        tri = cols[None, :] <= rows[:, None]
    else:
        tri = cols[None, :] >= rows[:, None]
    if UNIT:
        tri = tri & (cols[None, :] != rows[:, None])
    mask = row_mask[:, None] & col_mask[None, :] & tri
    a_offsets = (cols[None, :] * LDA + rows[:, None]) * 2
    ar = tl.load(
        a_ptr + a_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    ai = tl.load(
        a_ptr + a_offsets + 1,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    xr = tl.load(
        xin_ptr + cols * 2,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    )
    xi = tl.load(
        xin_ptr + cols * 2 + 1,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    )
    if CONJ:
        ai = -ai
    acc_r += ar * xr[None, :] - ai * xi[None, :]
    acc_i += ar * xi[None, :] + ai * xr[None, :]

    if UPLO == 0:
        post_start = diag_start + BLOCK_K
        a_offsets = (rows[:, None] + (post_start + offs_k)[None, :] * LDA) * 2
        x_offsets = (post_start + offs_k) * 2
        for kb in tl.range(post_start, n, BLOCK_K):
            cols = kb + offs_k
            col_mask = cols < n
            mask = row_mask[:, None] & col_mask[None, :]
            ar = tl.load(
                a_ptr + a_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
            ai = tl.load(
                a_ptr + a_offsets + 1,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
            xr = tl.load(
                xin_ptr + x_offsets,
                mask=col_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            xi = tl.load(
                xin_ptr + x_offsets + 1,
                mask=col_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if CONJ:
                ai = -ai
            acc_r += ar * xr[None, :] - ai * xi[None, :]
            acc_i += ar * xi[None, :] + ai * xr[None, :]
            a_offsets += BLOCK_K * LDA * 2
            x_offsets += BLOCK_K * 2

    out_r = tl.sum(acc_r, axis=1)
    out_i = tl.sum(acc_i, axis=1)
    if UNIT:
        out_r += tl.load(xin_ptr + rows * 2, mask=row_mask, other=0.0)
        out_i += tl.load(xin_ptr + rows * 2 + 1, mask=row_mask, other=0.0)
    tl.store(x_ptr + rows * 2, out_r, mask=row_mask)
    tl.store(x_ptr + rows * 2 + 1, out_i, mask=row_mask)


def ctrmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    use_n_lane = incx == 1 and trans == CUBLAS_OP_N and 127 <= n <= 1024
    use_n_gemv = incx == 1 and trans == CUBLAS_OP_N and n >= 1536
    use_trans_gemv = incx == 1 and trans != CUBLAS_OP_N and n >= 127
    use_row = False
    use_atomic = incx == 1 and trans == CUBLAS_OP_N and n > 1024
    if not use_n_lane and not use_n_gemv and not use_trans_gemv and not use_atomic:
        common_ctrmv(uplo, trans, diag, n, A, lda, x, incx)
        return

    assert A.dtype == torch.complex64 == x.dtype
    _check_trmv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=True)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    conj = 1 if trans == CUBLAS_OP_C else 0

    with torch_device_fn.device(A.device):
        xin = x.clone()
        A_packed = A.view(torch.int64)
        xin_packed = xin.view(torch.int64)
        x_real = torch.view_as_real(x)
        mode_key = _mode_key(uplo, trans_flag, unit) | (conj << 8)

        if use_n_lane:

            def n_lane_grid(meta):
                return (triton.cdiv(n, meta["BLOCK_M"]),)

            ctrmv_n_lane_kernel[n_lane_grid](
                A_packed,
                xin_packed,
                x_real,
                n,
                lda,
                mode_key,
                UPLO=uplo,
                UNIT=unit,
            )
            return

        if use_n_gemv:

            def n_gemv_grid(meta):
                return (triton.cdiv(n, meta["ROWS_PER_CTA"]),)

            complex_trmv_n_gemv_kernel[n_gemv_grid](
                torch.view_as_real(A),
                torch.view_as_real(xin),
                x_real,
                n,
                lda,
                mode_key,
                UPLO=uplo,
                UNIT=unit,
                FP64=0,
            )
            return

        if use_trans_gemv:

            def trans_grid(meta):
                return (triton.cdiv(n, meta["BLOCK_M"]),)

            ctrmv_trans_packed_kernel[trans_grid](
                A_packed,
                xin_packed,
                x_real,
                n,
                lda,
                mode_key,
                UPLO=uplo,
                UNIT=unit,
                CONJ=conj,
            )
            return

        if use_row:
            ctrmv_row_kernel[(n,)](
                A_packed,
                xin_packed,
                x_real,
                n,
                lda,
                mode_key,
                UPLO=uplo,
                CONJ=conj,
            )
            return

        if use_atomic:
            if trans_flag == 0:
                block_m = block_k = 64
                num_warps = 8
            else:
                use_large_tile = n < 6144
                block_m = block_k = 64 if use_large_tile else 32
                num_warps = 8 if use_large_tile else 4
            block_n = 1024
            ctrmv_atomic_init_kernel[(triton.cdiv(n, block_n),)](
                xin_packed,
                x_real,
                n,
                UNIT=unit,
                BLOCK_N=block_n,
                num_warps=4,
                num_stages=1,
            )
            ctrmv_atomic_kernel[(triton.cdiv(n, block_m), triton.cdiv(n, block_k))](
                A_packed,
                xin_packed,
                x_real,
                n,
                lda,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                CONJ=conj,
                BLOCK_SIZE_M=block_m,
                BLOCK_K=block_k,
                num_warps=num_warps,
                num_stages=1,
            )
            return

        split_k = 8
        block_m = block_k = 64 if n < 12288 else 32
        partial = torch.empty((split_k, n, 2), dtype=torch.float32, device=A.device)
        ctrmv_splitk_kernel[(triton.cdiv(n, block_m), split_k)](
            A_packed,
            xin_packed,
            partial,
            n,
            lda,
            SPLIT_K=split_k,
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
            CONJ=conj,
            BLOCK_SIZE_M=block_m,
            BLOCK_K=block_k,
            num_warps=8,
            num_stages=1,
        )
        block_n = 1024
        ctrmv_splitk_reduce_kernel[(triton.cdiv(n, block_n),)](
            partial,
            xin_packed,
            x_real,
            n,
            SPLIT_K=split_k,
            UNIT=unit,
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=1,
        )


def ztrmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    use_n_lane_kernel = incx == 1 and trans == CUBLAS_OP_N and 127 <= n < 2048
    use_n_gemv_kernel = incx == 1 and trans == CUBLAS_OP_N and n >= 2048
    use_trans_gemv_kernel = incx == 1 and trans != CUBLAS_OP_N and n >= 127
    if not use_n_lane_kernel and not use_n_gemv_kernel and not use_trans_gemv_kernel:
        common_ztrmv(uplo, trans, diag, n, A, lda, x, incx)
        return

    assert A.dtype == torch.complex128 == x.dtype
    _check_trmv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=True)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    conj = 1 if trans == CUBLAS_OP_C else 0
    mode_key = _mode_key(uplo, trans_flag, unit) | (conj << 8)

    with torch_device_fn.device(A.device):
        xin = x.clone()
        if use_n_lane_kernel:

            def n_lane_grid(meta):
                return (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)

            ztrmv_n_lane_kernel[n_lane_grid](
                torch.view_as_real(A),
                torch.view_as_real(xin),
                torch.view_as_real(x),
                n,
                lda,
                mode_key,
                UPLO=uplo,
                UNIT=unit,
            )
            return
        if use_n_gemv_kernel:

            def n_gemv_grid(meta):
                return (triton.cdiv(n, meta["ROWS_PER_CTA"]),)

            complex_trmv_n_gemv_kernel[n_gemv_grid](
                torch.view_as_real(A),
                torch.view_as_real(xin),
                torch.view_as_real(x),
                n,
                lda,
                mode_key,
                UPLO=uplo,
                UNIT=unit,
                FP64=1,
            )
            return

        def trans_grid(meta):
            return (triton.cdiv(n, meta["BLOCK_M"]),)

        complex_trmv_trans_gemv_kernel[trans_grid](
            torch.view_as_real(A),
            torch.view_as_real(xin),
            torch.view_as_real(x),
            n,
            lda,
            mode_key,
            UPLO=uplo,
            UNIT=unit,
            CONJ=conj,
            FP64=1,
        )
