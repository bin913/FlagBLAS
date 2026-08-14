import logging
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2._constants import (
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_DIAG_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]

CUBLAS_SIDE_LEFT = 0
CUBLAS_SIDE_RIGHT = 1

_TRMM_KEY = ["m", "n", "mode_key"]

_C3M_MIN = 1024

_Z3M_MIN = 768
_Z3M_PACK_AB_MAX = 2048


@libentry()
@libtuner(configs=runtime.get_tuned_config("strmm"), key=_TRMM_KEY)
@triton.jit
def _strmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    m,
    n,
    lda,
    ldb,
    ldc,
    mode_key,
    SIDE: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < m
    col_mask = cols < n
    k_extent = m if SIDE == 0 else n
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if UNIT:
        b0 = tl.load(
            b_ptr + rows[:, None] * ldb + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        acc += b0

    for k0 in tl.range(0, k_extent, BLOCK_K):
        kidx = k0 + offs_k
        k_mask = kidx < k_extent
        k_last = k0 + BLOCK_K - 1
        if SIDE == 0:
            tile_first = pid_m * BLOCK_M
            tile_last = tl.minimum(tile_first + BLOCK_M - 1, m - 1)
            if TRANS == 0:
                if UPLO == 1:
                    do_block = k_last >= tile_first
                    full_block = k0 >= tile_last + UNIT
                else:
                    do_block = k0 <= tile_last
                    full_block = k_last <= tile_first - UNIT
            else:
                if UPLO == 1:
                    do_block = k0 <= tile_last
                    full_block = k_last <= tile_first - UNIT
                else:
                    do_block = k_last >= tile_first
                    full_block = k0 >= tile_last + UNIT
        else:
            tile_first = pid_n * BLOCK_N
            tile_last = tl.minimum(tile_first + BLOCK_N - 1, n - 1)
            if TRANS == 0:
                if UPLO == 1:
                    do_block = k0 <= tile_last
                    full_block = k_last <= tile_first - UNIT
                else:
                    do_block = k_last >= tile_first
                    full_block = k0 >= tile_last + UNIT
            else:
                if UPLO == 1:
                    do_block = k_last >= tile_first
                    full_block = k0 >= tile_last + UNIT
                else:
                    do_block = k0 <= tile_last
                    full_block = k_last <= tile_first - UNIT
        if do_block:
            if SIDE == 0:
                if TRANS == 0:
                    a_row = rows[:, None]
                    a_col = kidx[None, :]
                else:
                    a_row = kidx[None, :]
                    a_col = rows[:, None]
                if full_block:
                    a = tl.load(
                        a_ptr + a_row * lda + a_col,
                        mask=row_mask[:, None] & k_mask[None, :],
                        other=0.0,
                        eviction_policy="evict_first",
                    )
                else:
                    if UPLO == 1:
                        tri = a_col >= a_row
                    else:
                        tri = a_col <= a_row
                    if UNIT:
                        tri = tri & (a_row != a_col)
                    a = tl.load(
                        a_ptr + a_row * lda + a_col,
                        mask=row_mask[:, None] & k_mask[None, :] & tri,
                        other=0.0,
                        eviction_policy="evict_first",
                    )
                b = tl.load(
                    b_ptr + kidx[:, None] * ldb + cols[None, :],
                    mask=k_mask[:, None] & col_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            else:
                if TRANS == 0:
                    a_row = kidx[:, None]
                    a_col = cols[None, :]
                else:
                    a_row = cols[None, :]
                    a_col = kidx[:, None]
                b = tl.load(
                    b_ptr + rows[:, None] * ldb + kidx[None, :],
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                if full_block:
                    a = tl.load(
                        a_ptr + a_row * lda + a_col,
                        mask=k_mask[:, None] & col_mask[None, :],
                        other=0.0,
                        eviction_policy="evict_first",
                    )
                else:
                    if UPLO == 1:
                        tri = a_col >= a_row
                    else:
                        tri = a_col <= a_row
                    if UNIT:
                        tri = tri & (a_row != a_col)
                    a = tl.load(
                        a_ptr + a_row * lda + a_col,
                        mask=k_mask[:, None] & col_mask[None, :] & tri,
                        other=0.0,
                        eviction_policy="evict_first",
                    )
                acc = tl.dot(b, a, acc, out_dtype=tl.float32, allow_tf32=False)

    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptr + rows[:, None] * ldc + cols[None, :], alpha * acc, mask=c_mask)


@libentry()
@libtuner(configs=runtime.get_tuned_config("strmm"), key=_TRMM_KEY)
@triton.jit
def _strmm_left_n_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    m,
    n,
    lda,
    ldb,
    ldc,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    if UPLO == 0:
        pid_m = grid_m - 1 - pid_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < m
    col_mask = cols < n
    full_tile = EVEN or (((pid_m + 1) * BLOCK_M <= m) & ((pid_n + 1) * BLOCK_N <= n))
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if UNIT:
        b0 = tl.load(
            b_ptr + rows[:, None] * ldb + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc += b0

    diag_start = pid_m * BLOCK_M
    if UPLO == 0:
        a_ptrs = a_ptr + rows[:, None] * lda + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * ldb + cols[None, :]
        if full_tile:
            for _k0 in tl.range(0, diag_start, BLOCK_K):
                a = tl.load(a_ptrs, eviction_policy="evict_first")
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * ldb
        else:
            for _k0 in tl.range(0, diag_start, BLOCK_K):
                a = tl.load(
                    a_ptrs,
                    mask=row_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                b = tl.load(
                    b_ptrs,
                    mask=col_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * ldb
        for i in tl.static_range(0, BLOCK_M // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            k_mask = kidx < m
            tri = kidx[None, :] <= rows[:, None]
            if UNIT:
                tri = tri & (kidx[None, :] != rows[:, None])
            a = tl.load(
                a_ptr + rows[:, None] * lda + kidx[None, :],
                mask=row_mask[:, None] & k_mask[None, :] & tri,
                other=0.0,
                eviction_policy="evict_first",
            )
            b = tl.load(
                b_ptr + kidx[:, None] * ldb + cols[None, :],
                mask=k_mask[:, None] & col_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
    else:
        for i in tl.static_range(0, BLOCK_M // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            k_mask = kidx < m
            tri = kidx[None, :] >= rows[:, None]
            if UNIT:
                tri = tri & (kidx[None, :] != rows[:, None])
            a = tl.load(
                a_ptr + rows[:, None] * lda + kidx[None, :],
                mask=row_mask[:, None] & k_mask[None, :] & tri,
                other=0.0,
                eviction_policy="evict_first",
            )
            b = tl.load(
                b_ptr + kidx[:, None] * ldb + cols[None, :],
                mask=k_mask[:, None] & col_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        k_start = diag_start + BLOCK_M
        a_ptrs = a_ptr + rows[:, None] * lda + (k_start + offs_k)[None, :]
        b_ptrs = b_ptr + (k_start + offs_k)[:, None] * ldb + cols[None, :]
        if full_tile:
            for k0 in tl.range(k_start, m, BLOCK_K):
                kidx = k0 + offs_k
                k_mask = kidx < m
                a = tl.load(
                    a_ptrs,
                    mask=k_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                b = tl.load(
                    b_ptrs,
                    mask=k_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * ldb
        else:
            for k0 in tl.range(k_start, m, BLOCK_K):
                kidx = k0 + offs_k
                k_mask = kidx < m
                a = tl.load(
                    a_ptrs,
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                b = tl.load(
                    b_ptrs,
                    mask=k_mask[:, None] & col_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * ldb

    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptr + rows[:, None] * ldc + cols[None, :], alpha * acc, mask=c_mask)


@libentry()
@triton.jit
def _strmm_left_lower_n_9999_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
):
    BLOCK_M: tl.constexpr = 32
    BLOCK_N: tl.constexpr = 256
    BLOCK_K: tl.constexpr = 16
    GROUP_M: tl.constexpr = 1
    pid = tl.program_id(0)
    grid_m = tl.cdiv(9999, BLOCK_M)
    grid_n = tl.cdiv(4096, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    pid_m = grid_m - 1 - pid_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < 9999
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    diag_start = pid_m * BLOCK_M
    full_tile = (pid_m + 1) * BLOCK_M <= 9999
    a_ptrs = a_ptr + rows[:, None] * 9999 + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * 4096 + cols[None, :]
    if full_tile:
        for _k0 in tl.range(0, diag_start, BLOCK_K):
            a = tl.load(a_ptrs, eviction_policy="evict_first")
            b = tl.load(b_ptrs, eviction_policy="evict_last")
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * 4096
    else:
        for _k0 in tl.range(0, diag_start, BLOCK_K):
            a = tl.load(
                a_ptrs, mask=row_mask[:, None], other=0.0, eviction_policy="evict_first"
            )
            b = tl.load(b_ptrs, eviction_policy="evict_last")
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * 4096
    for i in tl.static_range(0, BLOCK_M // BLOCK_K):
        kidx = diag_start + i * BLOCK_K + offs_k
        k_mask = kidx < 9999
        tri = kidx[None, :] <= rows[:, None]
        a = tl.load(
            a_ptr + rows[:, None] * 9999 + kidx[None, :],
            mask=row_mask[:, None] & k_mask[None, :] & tri,
            other=0.0,
            eviction_policy="evict_first",
        )
        b = tl.load(
            b_ptr + kidx[:, None] * 4096 + cols[None, :],
            mask=k_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

    tl.store(
        c_ptr + rows[:, None] * 4096 + cols[None, :],
        alpha * acc,
        mask=row_mask[:, None],
    )


@libentry()
@triton.jit
def _strmm_left_lower_n_16384_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
):
    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 128
    BLOCK_K: tl.constexpr = 64
    GROUP_M: tl.constexpr = 4
    pid = tl.program_id(0)
    grid_m = tl.cdiv(16384, BLOCK_M)
    grid_n = tl.cdiv(4096, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    pid_m = grid_m - 1 - pid_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    diag_start = pid_m * BLOCK_M
    a_ptrs = a_ptr + rows[:, None] * 16384 + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * 4096 + cols[None, :]
    for _k0 in tl.range(0, diag_start, BLOCK_K):
        a = tl.load(a_ptrs, eviction_policy="evict_first")
        b = tl.load(b_ptrs, eviction_policy="evict_last")
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * 4096
    for i in tl.static_range(0, BLOCK_M // BLOCK_K):
        kidx = diag_start + i * BLOCK_K + offs_k
        tri = kidx[None, :] <= rows[:, None]
        a = tl.load(
            a_ptr + rows[:, None] * 16384 + kidx[None, :],
            mask=tri,
            other=0.0,
            eviction_policy="evict_first",
        )
        b = tl.load(
            b_ptr + kidx[:, None] * 4096 + cols[None, :],
            eviction_policy="evict_last",
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

    tl.store(c_ptr + rows[:, None] * 4096 + cols[None, :], alpha * acc)


@libentry()
@libtuner(configs=runtime.get_tuned_config("strmm"), key=_TRMM_KEY)
@triton.jit
def _strmm_right_n_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    m,
    n,
    lda,
    ldb,
    ldc,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    if UPLO == 1:
        pid_n = grid_n - 1 - pid_n

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < m
    col_mask = cols < n
    full_tile = EVEN or (((pid_m + 1) * BLOCK_M <= m) & ((pid_n + 1) * BLOCK_N <= n))
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if UNIT:
        b0 = tl.load(
            b_ptr + rows[:, None] * ldb + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc += b0

    diag_start = pid_n * BLOCK_N
    if UPLO == 1:
        b_ptrs = b_ptr + rows[:, None] * ldb + offs_k[None, :]
        a_ptrs = a_ptr + offs_k[:, None] * lda + cols[None, :]
        if full_tile:
            for _k0 in tl.range(0, diag_start, BLOCK_K):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs, eviction_policy="evict_first")
                acc = tl.dot(b, a, acc, out_dtype=tl.float32, allow_tf32=False)
                b_ptrs += BLOCK_K
                a_ptrs += BLOCK_K * lda
        else:
            for _k0 in tl.range(0, diag_start, BLOCK_K):
                b = tl.load(
                    b_ptrs,
                    mask=row_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                a = tl.load(
                    a_ptrs,
                    mask=col_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                acc = tl.dot(b, a, acc, out_dtype=tl.float32, allow_tf32=False)
                b_ptrs += BLOCK_K
                a_ptrs += BLOCK_K * lda
        for i in tl.static_range(0, BLOCK_N // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            k_mask = kidx < n
            tri = kidx[:, None] <= cols[None, :]
            if UNIT:
                tri = tri & (kidx[:, None] != cols[None, :])
            b = tl.load(
                b_ptr + rows[:, None] * ldb + kidx[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            a = tl.load(
                a_ptr + kidx[:, None] * lda + cols[None, :],
                mask=k_mask[:, None] & col_mask[None, :] & tri,
                other=0.0,
                eviction_policy="evict_first",
            )
            acc = tl.dot(b, a, acc, out_dtype=tl.float32, allow_tf32=False)
    else:
        for i in tl.static_range(0, BLOCK_N // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            k_mask = kidx < n
            tri = kidx[:, None] >= cols[None, :]
            if UNIT:
                tri = tri & (kidx[:, None] != cols[None, :])
            b = tl.load(
                b_ptr + rows[:, None] * ldb + kidx[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            a = tl.load(
                a_ptr + kidx[:, None] * lda + cols[None, :],
                mask=k_mask[:, None] & col_mask[None, :] & tri,
                other=0.0,
                eviction_policy="evict_first",
            )
            acc = tl.dot(b, a, acc, out_dtype=tl.float32, allow_tf32=False)
        k_start = diag_start + BLOCK_N
        b_ptrs = b_ptr + rows[:, None] * ldb + (k_start + offs_k)[None, :]
        a_ptrs = a_ptr + (k_start + offs_k)[:, None] * lda + cols[None, :]
        if full_tile:
            for k0 in tl.range(k_start, n, BLOCK_K):
                kidx = k0 + offs_k
                k_mask = kidx < n
                b = tl.load(
                    b_ptrs,
                    mask=k_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                a = tl.load(
                    a_ptrs,
                    mask=k_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                acc = tl.dot(b, a, acc, out_dtype=tl.float32, allow_tf32=False)
                b_ptrs += BLOCK_K
                a_ptrs += BLOCK_K * lda
        else:
            for k0 in tl.range(k_start, n, BLOCK_K):
                kidx = k0 + offs_k
                k_mask = kidx < n
                b = tl.load(
                    b_ptrs,
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                a = tl.load(
                    a_ptrs,
                    mask=k_mask[:, None] & col_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                acc = tl.dot(b, a, acc, out_dtype=tl.float32, allow_tf32=False)
                b_ptrs += BLOCK_K
                a_ptrs += BLOCK_K * lda

    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptr + rows[:, None] * ldc + cols[None, :], alpha * acc, mask=c_mask)


@libentry()
@libtuner(configs=runtime.get_tuned_config("strmm"), key=_TRMM_KEY)
@triton.jit
def _strmm_left_t_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    m,
    n,
    lda,
    ldb,
    ldc,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    if UPLO == 1:
        pid_m = grid_m - 1 - pid_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < m
    col_mask = cols < n
    full_tile = EVEN or (((pid_m + 1) * BLOCK_M <= m) & ((pid_n + 1) * BLOCK_N <= n))
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if UNIT:
        b0 = tl.load(
            b_ptr + rows[:, None] * ldb + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc += b0

    diag_start = pid_m * BLOCK_M
    if UPLO == 1:
        a_ptrs = a_ptr + offs_k[None, :] * lda + rows[:, None]
        b_ptrs = b_ptr + offs_k[:, None] * ldb + cols[None, :]
        if full_tile:
            for _k0 in tl.range(0, diag_start, BLOCK_K):
                a = tl.load(a_ptrs, eviction_policy="evict_first")
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
                a_ptrs += BLOCK_K * lda
                b_ptrs += BLOCK_K * ldb
        else:
            for _k0 in tl.range(0, diag_start, BLOCK_K):
                a = tl.load(
                    a_ptrs,
                    mask=row_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                b = tl.load(
                    b_ptrs,
                    mask=col_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
                a_ptrs += BLOCK_K * lda
                b_ptrs += BLOCK_K * ldb
        for i in tl.static_range(0, BLOCK_M // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            k_mask = kidx < m
            tri = kidx[None, :] <= rows[:, None]
            if UNIT:
                tri = tri & (kidx[None, :] != rows[:, None])
            a = tl.load(
                a_ptr + kidx[None, :] * lda + rows[:, None],
                mask=row_mask[:, None] & k_mask[None, :] & tri,
                other=0.0,
                eviction_policy="evict_first",
            )
            b = tl.load(
                b_ptr + kidx[:, None] * ldb + cols[None, :],
                mask=k_mask[:, None] & col_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
    else:
        for i in tl.static_range(0, BLOCK_M // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            k_mask = kidx < m
            tri = kidx[None, :] >= rows[:, None]
            if UNIT:
                tri = tri & (kidx[None, :] != rows[:, None])
            a = tl.load(
                a_ptr + kidx[None, :] * lda + rows[:, None],
                mask=row_mask[:, None] & k_mask[None, :] & tri,
                other=0.0,
                eviction_policy="evict_first",
            )
            b = tl.load(
                b_ptr + kidx[:, None] * ldb + cols[None, :],
                mask=k_mask[:, None] & col_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        k_start = diag_start + BLOCK_M
        a_ptrs = a_ptr + (k_start + offs_k)[None, :] * lda + rows[:, None]
        b_ptrs = b_ptr + (k_start + offs_k)[:, None] * ldb + cols[None, :]
        if full_tile:
            for k0 in tl.range(k_start, m, BLOCK_K):
                kidx = k0 + offs_k
                k_mask = kidx < m
                a = tl.load(
                    a_ptrs,
                    mask=k_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                b = tl.load(
                    b_ptrs,
                    mask=k_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
                a_ptrs += BLOCK_K * lda
                b_ptrs += BLOCK_K * ldb
        else:
            for k0 in tl.range(k_start, m, BLOCK_K):
                kidx = k0 + offs_k
                k_mask = kidx < m
                a = tl.load(
                    a_ptrs,
                    mask=k_mask[None, :] & row_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                b = tl.load(
                    b_ptrs,
                    mask=k_mask[:, None] & col_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
                a_ptrs += BLOCK_K * lda
                b_ptrs += BLOCK_K * ldb

    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptr + rows[:, None] * ldc + cols[None, :], alpha * acc, mask=c_mask)


@libentry()
@libtuner(configs=runtime.get_tuned_config("dtrmm_dedicated"), key=_TRMM_KEY)
@triton.jit
def _dtrmm_left_n_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float64,
    m,
    n,
    lda,
    ldb,
    ldc,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    if UPLO == 0:
        pid_m = grid_m - 1 - pid_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < m
    col_mask = cols < n
    full_tile = EVEN or (((pid_m + 1) * BLOCK_M <= m) & ((pid_n + 1) * BLOCK_N <= n))
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    if UNIT:
        b0 = tl.load(
            b_ptr + rows[:, None] * ldb + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc += b0

    diag_start = pid_m * BLOCK_M
    if UPLO == 0:
        a_ptrs = a_ptr + rows[:, None] * lda + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * ldb + cols[None, :]
        if full_tile:
            for _k0 in tl.range(0, diag_start, BLOCK_K):
                a = tl.load(a_ptrs, eviction_policy="evict_first")
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float64)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * ldb
        else:
            for _k0 in tl.range(0, diag_start, BLOCK_K):
                a = tl.load(
                    a_ptrs,
                    mask=row_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                b = tl.load(
                    b_ptrs,
                    mask=col_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float64)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * ldb
        for i in tl.static_range(0, BLOCK_M // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            k_mask = kidx < m
            tri = kidx[None, :] <= rows[:, None]
            if UNIT:
                tri = tri & (kidx[None, :] != rows[:, None])
            a = tl.load(
                a_ptr + rows[:, None] * lda + kidx[None, :],
                mask=row_mask[:, None] & k_mask[None, :] & tri,
                other=0.0,
                eviction_policy="evict_first",
            )
            b = tl.load(
                b_ptr + kidx[:, None] * ldb + cols[None, :],
                mask=k_mask[:, None] & col_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float64)
    else:
        for i in tl.static_range(0, BLOCK_M // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            k_mask = kidx < m
            tri = kidx[None, :] >= rows[:, None]
            if UNIT:
                tri = tri & (kidx[None, :] != rows[:, None])
            a = tl.load(
                a_ptr + rows[:, None] * lda + kidx[None, :],
                mask=row_mask[:, None] & k_mask[None, :] & tri,
                other=0.0,
                eviction_policy="evict_first",
            )
            b = tl.load(
                b_ptr + kidx[:, None] * ldb + cols[None, :],
                mask=k_mask[:, None] & col_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float64)
        k_start = diag_start + BLOCK_M
        a_ptrs = a_ptr + rows[:, None] * lda + (k_start + offs_k)[None, :]
        b_ptrs = b_ptr + (k_start + offs_k)[:, None] * ldb + cols[None, :]
        if full_tile:
            for k0 in tl.range(k_start, m, BLOCK_K):
                kidx = k0 + offs_k
                k_mask = kidx < m
                a = tl.load(
                    a_ptrs,
                    mask=k_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                b = tl.load(
                    b_ptrs,
                    mask=k_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float64)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * ldb
        else:
            for k0 in tl.range(k_start, m, BLOCK_K):
                kidx = k0 + offs_k
                k_mask = kidx < m
                a = tl.load(
                    a_ptrs,
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                b = tl.load(
                    b_ptrs,
                    mask=k_mask[:, None] & col_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float64)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * ldb

    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptr + rows[:, None] * ldc + cols[None, :], alpha * acc, mask=c_mask)


@libentry()
@libtuner(configs=runtime.get_tuned_config("dtrmm_dedicated"), key=_TRMM_KEY)
@triton.jit
def _dtrmm_right_n_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float64,
    m,
    n,
    lda,
    ldb,
    ldc,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    if UPLO == 1:
        pid_n = grid_n - 1 - pid_n

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < m
    col_mask = cols < n
    full_tile = EVEN or (((pid_m + 1) * BLOCK_M <= m) & ((pid_n + 1) * BLOCK_N <= n))
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    if UNIT:
        b0 = tl.load(
            b_ptr + rows[:, None] * ldb + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc += b0

    diag_start = pid_n * BLOCK_N
    if UPLO == 1:
        b_ptrs = b_ptr + rows[:, None] * ldb + offs_k[None, :]
        a_ptrs = a_ptr + offs_k[:, None] * lda + cols[None, :]
        if full_tile:
            for _k0 in tl.range(0, diag_start, BLOCK_K):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs, eviction_policy="evict_first")
                acc = tl.dot(b, a, acc, input_precision="ieee", out_dtype=tl.float64)
                b_ptrs += BLOCK_K
                a_ptrs += BLOCK_K * lda
        else:
            for _k0 in tl.range(0, diag_start, BLOCK_K):
                b = tl.load(
                    b_ptrs,
                    mask=row_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                a = tl.load(
                    a_ptrs,
                    mask=col_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                acc = tl.dot(b, a, acc, input_precision="ieee", out_dtype=tl.float64)
                b_ptrs += BLOCK_K
                a_ptrs += BLOCK_K * lda
        for i in tl.static_range(0, BLOCK_N // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            k_mask = kidx < n
            tri = kidx[:, None] <= cols[None, :]
            if UNIT:
                tri = tri & (kidx[:, None] != cols[None, :])
            b = tl.load(
                b_ptr + rows[:, None] * ldb + kidx[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            a = tl.load(
                a_ptr + kidx[:, None] * lda + cols[None, :],
                mask=k_mask[:, None] & col_mask[None, :] & tri,
                other=0.0,
                eviction_policy="evict_first",
            )
            acc = tl.dot(b, a, acc, input_precision="ieee", out_dtype=tl.float64)
    else:
        for i in tl.static_range(0, BLOCK_N // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            k_mask = kidx < n
            tri = kidx[:, None] >= cols[None, :]
            if UNIT:
                tri = tri & (kidx[:, None] != cols[None, :])
            b = tl.load(
                b_ptr + rows[:, None] * ldb + kidx[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            a = tl.load(
                a_ptr + kidx[:, None] * lda + cols[None, :],
                mask=k_mask[:, None] & col_mask[None, :] & tri,
                other=0.0,
                eviction_policy="evict_first",
            )
            acc = tl.dot(b, a, acc, input_precision="ieee", out_dtype=tl.float64)
        k_start = diag_start + BLOCK_N
        b_ptrs = b_ptr + rows[:, None] * ldb + (k_start + offs_k)[None, :]
        a_ptrs = a_ptr + (k_start + offs_k)[:, None] * lda + cols[None, :]
        if full_tile:
            for k0 in tl.range(k_start, n, BLOCK_K):
                kidx = k0 + offs_k
                k_mask = kidx < n
                b = tl.load(
                    b_ptrs,
                    mask=k_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                a = tl.load(
                    a_ptrs,
                    mask=k_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                acc = tl.dot(b, a, acc, input_precision="ieee", out_dtype=tl.float64)
                b_ptrs += BLOCK_K
                a_ptrs += BLOCK_K * lda
        else:
            for k0 in tl.range(k_start, n, BLOCK_K):
                kidx = k0 + offs_k
                k_mask = kidx < n
                b = tl.load(
                    b_ptrs,
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                a = tl.load(
                    a_ptrs,
                    mask=k_mask[:, None] & col_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                acc = tl.dot(b, a, acc, input_precision="ieee", out_dtype=tl.float64)
                b_ptrs += BLOCK_K
                a_ptrs += BLOCK_K * lda

    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptr + rows[:, None] * ldc + cols[None, :], alpha * acc, mask=c_mask)


@libentry()
@libtuner(configs=runtime.get_tuned_config("dtrmm_left_t_dedicated"), key=_TRMM_KEY)
@triton.jit
def _dtrmm_left_t_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float64,
    m,
    n,
    lda,
    ldb,
    ldc,
    mode_key,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    if UPLO == 1:
        pid_m = grid_m - 1 - pid_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < m
    col_mask = cols < n
    full_tile = EVEN or (((pid_m + 1) * BLOCK_M <= m) & ((pid_n + 1) * BLOCK_N <= n))
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    if UNIT:
        b0 = tl.load(
            b_ptr + rows[:, None] * ldb + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc += b0

    diag_start = pid_m * BLOCK_M
    if UPLO == 1:
        a_ptrs = a_ptr + offs_k[None, :] * lda + rows[:, None]
        b_ptrs = b_ptr + offs_k[:, None] * ldb + cols[None, :]
        if full_tile:
            for _k0 in tl.range(0, diag_start, BLOCK_K):
                a = tl.load(a_ptrs, eviction_policy="evict_first")
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float64)
                a_ptrs += BLOCK_K * lda
                b_ptrs += BLOCK_K * ldb
        else:
            for _k0 in tl.range(0, diag_start, BLOCK_K):
                a = tl.load(
                    a_ptrs,
                    mask=row_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                b = tl.load(
                    b_ptrs,
                    mask=col_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float64)
                a_ptrs += BLOCK_K * lda
                b_ptrs += BLOCK_K * ldb
        for i in tl.static_range(0, BLOCK_M // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            k_mask = kidx < m
            tri = kidx[None, :] <= rows[:, None]
            if UNIT:
                tri = tri & (kidx[None, :] != rows[:, None])
            a = tl.load(
                a_ptr + kidx[None, :] * lda + rows[:, None],
                mask=row_mask[:, None] & k_mask[None, :] & tri,
                other=0.0,
                eviction_policy="evict_first",
            )
            b = tl.load(
                b_ptr + kidx[:, None] * ldb + cols[None, :],
                mask=k_mask[:, None] & col_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float64)
    else:
        for i in tl.static_range(0, BLOCK_M // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            k_mask = kidx < m
            tri = kidx[None, :] >= rows[:, None]
            if UNIT:
                tri = tri & (kidx[None, :] != rows[:, None])
            a = tl.load(
                a_ptr + kidx[None, :] * lda + rows[:, None],
                mask=row_mask[:, None] & k_mask[None, :] & tri,
                other=0.0,
                eviction_policy="evict_first",
            )
            b = tl.load(
                b_ptr + kidx[:, None] * ldb + cols[None, :],
                mask=k_mask[:, None] & col_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float64)
        k_start = diag_start + BLOCK_M
        a_ptrs = a_ptr + (k_start + offs_k)[None, :] * lda + rows[:, None]
        b_ptrs = b_ptr + (k_start + offs_k)[:, None] * ldb + cols[None, :]
        if full_tile:
            for k0 in tl.range(k_start, m, BLOCK_K):
                kidx = k0 + offs_k
                k_mask = kidx < m
                a = tl.load(
                    a_ptrs,
                    mask=k_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                b = tl.load(
                    b_ptrs,
                    mask=k_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float64)
                a_ptrs += BLOCK_K * lda
                b_ptrs += BLOCK_K * ldb
        else:
            for k0 in tl.range(k_start, m, BLOCK_K):
                kidx = k0 + offs_k
                k_mask = kidx < m
                a = tl.load(
                    a_ptrs,
                    mask=k_mask[None, :] & row_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                b = tl.load(
                    b_ptrs,
                    mask=k_mask[:, None] & col_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float64)
                a_ptrs += BLOCK_K * lda
                b_ptrs += BLOCK_K * ldb

    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptr + rows[:, None] * ldc + cols[None, :], alpha * acc, mask=c_mask)


@libentry()
@libtuner(configs=runtime.get_tuned_config("dtrmm"), key=_TRMM_KEY)
@triton.jit
def _dtrmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float64,
    m,
    n,
    lda,
    ldb,
    ldc,
    mode_key,
    SIDE: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < m
    col_mask = cols < n
    k_extent = m if SIDE == 0 else n
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)

    if UNIT:
        b0 = tl.load(
            b_ptr + rows[:, None] * ldb + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        acc += b0

    for k0 in tl.range(0, k_extent, BLOCK_K):
        kidx = k0 + offs_k
        k_mask = kidx < k_extent
        k_last = k0 + BLOCK_K - 1
        if SIDE == 0:
            tile_first = pid_m * BLOCK_M
            tile_last = tl.minimum(tile_first + BLOCK_M - 1, m - 1)
            if TRANS == 0:
                if UPLO == 1:
                    do_block = k_last >= tile_first
                else:
                    do_block = k0 <= tile_last
            else:
                if UPLO == 1:
                    do_block = k0 <= tile_last
                else:
                    do_block = k_last >= tile_first
        else:
            tile_first = pid_n * BLOCK_N
            tile_last = tl.minimum(tile_first + BLOCK_N - 1, n - 1)
            if TRANS == 0:
                if UPLO == 1:
                    do_block = k0 <= tile_last
                else:
                    do_block = k_last >= tile_first
            else:
                if UPLO == 1:
                    do_block = k_last >= tile_first
                else:
                    do_block = k0 <= tile_last
        if do_block:
            if SIDE == 0:
                if TRANS == 0:
                    a_row = rows[:, None]
                    a_col = kidx[None, :]
                else:
                    a_row = kidx[None, :]
                    a_col = rows[:, None]
                if UPLO == 1:
                    tri = a_col >= a_row
                else:
                    tri = a_col <= a_row
                if UNIT:
                    tri = tri & (a_row != a_col)
                a = tl.load(
                    a_ptr + a_row * lda + a_col,
                    mask=row_mask[:, None] & k_mask[None, :] & tri,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                b = tl.load(
                    b_ptr + kidx[:, None] * ldb + cols[None, :],
                    mask=k_mask[:, None] & col_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float64)
            else:
                if TRANS == 0:
                    a_row = kidx[:, None]
                    a_col = cols[None, :]
                else:
                    a_row = cols[None, :]
                    a_col = kidx[:, None]
                if UPLO == 1:
                    tri = a_col >= a_row
                else:
                    tri = a_col <= a_row
                if UNIT:
                    tri = tri & (a_row != a_col)
                b = tl.load(
                    b_ptr + rows[:, None] * ldb + kidx[None, :],
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                a = tl.load(
                    a_ptr + a_row * lda + a_col,
                    mask=k_mask[:, None] & col_mask[None, :] & tri,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                acc = tl.dot(b, a, acc, input_precision="ieee", out_dtype=tl.float64)

    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptr + rows[:, None] * ldc + cols[None, :], alpha * acc, mask=c_mask)


@libentry()
@libtuner(configs=runtime.get_tuned_config("ctrmm"), key=_TRMM_KEY)
@triton.jit
def _ctrmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    ar: tl.float32,
    ai: tl.float32,
    m,
    n,
    lda,
    ldb,
    ldc,
    mode_key,
    SIDE: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < m
    col_mask = cols < n
    k_extent = m if SIDE == 0 else n
    offs_k = tl.arange(0, BLOCK_K)

    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if UNIT:
        b_init_base = (rows[:, None] * ldb + cols[None, :]) * 2
        init_mask = row_mask[:, None] & col_mask[None, :]
        acc_r += tl.load(b_ptr + b_init_base, mask=init_mask, other=0.0)
        acc_i += tl.load(b_ptr + b_init_base + 1, mask=init_mask, other=0.0)

    if TRANS == 0:
        for k0 in tl.range(0, k_extent, BLOCK_K):
            k_last = k0 + BLOCK_K - 1
            if SIDE == 0:
                tile_first = pid_m * BLOCK_M
                tile_last = tl.minimum(tile_first + BLOCK_M - 1, m - 1)
                if UPLO == 1:
                    do_block = k_last >= tile_first
                else:
                    do_block = k0 <= tile_last
            else:
                tile_first = pid_n * BLOCK_N
                tile_last = tl.minimum(tile_first + BLOCK_N - 1, n - 1)
                if UPLO == 1:
                    do_block = k0 <= tile_last
                else:
                    do_block = k_last >= tile_first
            if do_block:
                for kk in tl.static_range(0, BLOCK_K):
                    kval = k0 + kk
                    k_mask = kval < k_extent
                    if SIDE == 0:
                        a_row = rows
                        a_col = kval
                        if UPLO == 1:
                            tri = a_col >= a_row
                        else:
                            tri = a_col <= a_row
                        if UNIT:
                            tri = tri & (a_row != a_col)
                        a_base = (a_row * lda + a_col) * 2
                        b_base = (kval * ldb + cols) * 2
                        av_r = tl.load(
                            a_ptr + a_base, mask=row_mask & k_mask & tri, other=0.0
                        )
                        av_i = tl.load(
                            a_ptr + a_base + 1, mask=row_mask & k_mask & tri, other=0.0
                        )
                        bv_r = tl.load(
                            b_ptr + b_base, mask=k_mask & col_mask, other=0.0
                        )
                        bv_i = tl.load(
                            b_ptr + b_base + 1, mask=k_mask & col_mask, other=0.0
                        )
                        acc_r += (
                            av_r[:, None] * bv_r[None, :]
                            - av_i[:, None] * bv_i[None, :]
                        )
                        acc_i += (
                            av_r[:, None] * bv_i[None, :]
                            + av_i[:, None] * bv_r[None, :]
                        )
                    else:
                        a_row = kval
                        a_col = cols
                        if UPLO == 1:
                            tri = a_col >= a_row
                        else:
                            tri = a_col <= a_row
                        if UNIT:
                            tri = tri & (a_row != a_col)
                        b_base = (rows * ldb + kval) * 2
                        a_base = (a_row * lda + a_col) * 2
                        bv_r = tl.load(
                            b_ptr + b_base, mask=row_mask & k_mask, other=0.0
                        )
                        bv_i = tl.load(
                            b_ptr + b_base + 1, mask=row_mask & k_mask, other=0.0
                        )
                        av_r = tl.load(
                            a_ptr + a_base, mask=col_mask & k_mask & tri, other=0.0
                        )
                        av_i = tl.load(
                            a_ptr + a_base + 1, mask=col_mask & k_mask & tri, other=0.0
                        )
                        acc_r += (
                            bv_r[:, None] * av_r[None, :]
                            - bv_i[:, None] * av_i[None, :]
                        )
                        acc_i += (
                            bv_r[:, None] * av_i[None, :]
                            + bv_i[:, None] * av_r[None, :]
                        )
    else:
        for k0 in tl.range(0, k_extent, BLOCK_K):
            kidx = k0 + offs_k
            k_mask = kidx < k_extent
            k_last = k0 + BLOCK_K - 1
            if SIDE == 0:
                tile_first = pid_m * BLOCK_M
                tile_last = tl.minimum(tile_first + BLOCK_M - 1, m - 1)
                if UPLO == 1:
                    do_block = k0 <= tile_last
                else:
                    do_block = k_last >= tile_first
            else:
                tile_first = pid_n * BLOCK_N
                tile_last = tl.minimum(tile_first + BLOCK_N - 1, n - 1)
                if UPLO == 1:
                    do_block = k_last >= tile_first
                else:
                    do_block = k0 <= tile_last
            if do_block:
                if SIDE == 0:
                    a_row = kidx[None, :]
                    a_col = rows[:, None]
                    if UPLO == 1:
                        tri = a_col >= a_row
                    else:
                        tri = a_col <= a_row
                    if UNIT:
                        tri = tri & (a_row != a_col)
                    a_base = (a_row * lda + a_col) * 2
                    b_base = (kidx[:, None] * ldb + cols[None, :]) * 2
                    a_mask = row_mask[:, None] & k_mask[None, :] & tri
                    b_mask = k_mask[:, None] & col_mask[None, :]
                    av_r = tl.load(
                        a_ptr + a_base,
                        mask=a_mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    )
                    av_i = tl.load(
                        a_ptr + a_base + 1,
                        mask=a_mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    )
                    bv_r = tl.load(
                        b_ptr + b_base,
                        mask=b_mask,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    bv_i = tl.load(
                        b_ptr + b_base + 1,
                        mask=b_mask,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    if CONJ:
                        av_i = -av_i
                    acc_r = tl.dot(
                        av_r, bv_r, acc_r, out_dtype=tl.float32, allow_tf32=False
                    )
                    acc_r -= tl.dot(av_i, bv_i, out_dtype=tl.float32, allow_tf32=False)
                    acc_i = tl.dot(
                        av_r, bv_i, acc_i, out_dtype=tl.float32, allow_tf32=False
                    )
                    acc_i += tl.dot(av_i, bv_r, out_dtype=tl.float32, allow_tf32=False)
                else:
                    a_row = cols[None, :]
                    a_col = kidx[:, None]
                    if UPLO == 1:
                        tri = a_col >= a_row
                    else:
                        tri = a_col <= a_row
                    if UNIT:
                        tri = tri & (a_row != a_col)
                    b_base = (rows[:, None] * ldb + kidx[None, :]) * 2
                    a_base = (a_row * lda + a_col) * 2
                    b_mask = row_mask[:, None] & k_mask[None, :]
                    a_mask = k_mask[:, None] & col_mask[None, :] & tri
                    bv_r = tl.load(
                        b_ptr + b_base,
                        mask=b_mask,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    bv_i = tl.load(
                        b_ptr + b_base + 1,
                        mask=b_mask,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    av_r = tl.load(
                        a_ptr + a_base,
                        mask=a_mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    )
                    av_i = tl.load(
                        a_ptr + a_base + 1,
                        mask=a_mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    )
                    if CONJ:
                        av_i = -av_i
                    acc_r = tl.dot(
                        bv_r, av_r, acc_r, out_dtype=tl.float32, allow_tf32=False
                    )
                    acc_r -= tl.dot(bv_i, av_i, out_dtype=tl.float32, allow_tf32=False)
                    acc_i = tl.dot(
                        bv_r, av_i, acc_i, out_dtype=tl.float32, allow_tf32=False
                    )
                    acc_i += tl.dot(bv_i, av_r, out_dtype=tl.float32, allow_tf32=False)

    out_r = ar * acc_r - ai * acc_i
    out_i = ar * acc_i + ai * acc_r
    c_base = (rows[:, None] * ldc + cols[None, :]) * 2
    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptr + c_base, out_r, mask=c_mask)
    tl.store(c_ptr + c_base + 1, out_i, mask=c_mask)


@triton.jit
def _ctrmm_512_left_upper_t_unit_split4m_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    ar: tl.float32,
    ai: tl.float32,
    lda: tl.constexpr,
    ldb: tl.constexpr,
    ldc: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(512, BLOCK_M)
    grid_n = tl.cdiv(256, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    tile_first = pid_m * BLOCK_M
    tile_last = tile_first + BLOCK_M - 1

    b_init = (rows[:, None] * ldb + cols[None, :]) * 2
    c_base = (rows[:, None] * ldc + cols[None, :]) * 2
    acc_r = tl.load(b_ptr + b_init)
    acc_i = tl.load(b_ptr + b_init + 1)

    for k0 in tl.range(0, 512, BLOCK_K):
        kidx = k0 + offs_k
        a_base = (kidx[None, :] * lda + rows[:, None]) * 2
        b_base = (kidx[:, None] * ldb + cols[None, :]) * 2
        if k0 + BLOCK_K - 1 < tile_first:
            av_r = tl.load(a_ptr + a_base, eviction_policy="evict_first")
            av_i = tl.load(a_ptr + a_base + 1, eviction_policy="evict_first")
            bv_r = tl.load(b_ptr + b_base, eviction_policy="evict_last")
            bv_i = tl.load(b_ptr + b_base + 1, eviction_policy="evict_last")
            acc_r = tl.dot(av_r, bv_r, acc_r, out_dtype=tl.float32, allow_tf32=False)
            acc_r -= tl.dot(av_i, bv_i, out_dtype=tl.float32, allow_tf32=False)
            acc_i = tl.dot(av_r, bv_i, acc_i, out_dtype=tl.float32, allow_tf32=False)
            acc_i += tl.dot(av_i, bv_r, out_dtype=tl.float32, allow_tf32=False)
        elif k0 <= tile_last:
            tri = rows[:, None] > kidx[None, :]
            av_r = tl.load(
                a_ptr + a_base, mask=tri, other=0.0, eviction_policy="evict_first"
            )
            av_i = tl.load(
                a_ptr + a_base + 1, mask=tri, other=0.0, eviction_policy="evict_first"
            )
            bv_r = tl.load(b_ptr + b_base, eviction_policy="evict_last")
            bv_i = tl.load(b_ptr + b_base + 1, eviction_policy="evict_last")
            acc_r = tl.dot(av_r, bv_r, acc_r, out_dtype=tl.float32, allow_tf32=False)
            acc_r -= tl.dot(av_i, bv_i, out_dtype=tl.float32, allow_tf32=False)
            acc_i = tl.dot(av_r, bv_i, acc_i, out_dtype=tl.float32, allow_tf32=False)
            acc_i += tl.dot(av_i, bv_r, out_dtype=tl.float32, allow_tf32=False)

    out_r = ar * acc_r - ai * acc_i
    out_i = ar * acc_i + ai * acc_r
    tl.store(c_ptr + c_base, out_r)
    tl.store(c_ptr + c_base + 1, out_i)


@triton.jit
def _ctrmm_left_upper_t_unit_split4m_interval_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    ar: tl.float32,
    ai: tl.float32,
    m,
    n,
    lda,
    ldb,
    ldc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    tile_first = pid_m * BLOCK_M
    tile_last = tile_first + BLOCK_M - 1

    b_init = (rows[:, None] * ldb + cols[None, :]) * 2
    c_base = (rows[:, None] * ldc + cols[None, :]) * 2
    acc_r = tl.load(b_ptr + b_init)
    acc_i = tl.load(b_ptr + b_init + 1)

    for k0 in tl.range(0, m, BLOCK_K):
        kidx = k0 + offs_k
        a_base = (kidx[None, :] * lda + rows[:, None]) * 2
        b_base = (kidx[:, None] * ldb + cols[None, :]) * 2
        if k0 + BLOCK_K - 1 < tile_first:
            av_r = tl.load(a_ptr + a_base, eviction_policy="evict_first")
            av_i = tl.load(a_ptr + a_base + 1, eviction_policy="evict_first")
            bv_r = tl.load(b_ptr + b_base, eviction_policy="evict_last")
            bv_i = tl.load(b_ptr + b_base + 1, eviction_policy="evict_last")
            acc_r = tl.dot(av_r, bv_r, acc_r, out_dtype=tl.float32, allow_tf32=False)
            acc_r -= tl.dot(av_i, bv_i, out_dtype=tl.float32, allow_tf32=False)
            acc_i = tl.dot(av_r, bv_i, acc_i, out_dtype=tl.float32, allow_tf32=False)
            acc_i += tl.dot(av_i, bv_r, out_dtype=tl.float32, allow_tf32=False)
        elif k0 <= tile_last:
            tri = rows[:, None] > kidx[None, :]
            av_r = tl.load(
                a_ptr + a_base, mask=tri, other=0.0, eviction_policy="evict_first"
            )
            av_i = tl.load(
                a_ptr + a_base + 1, mask=tri, other=0.0, eviction_policy="evict_first"
            )
            bv_r = tl.load(b_ptr + b_base, eviction_policy="evict_last")
            bv_i = tl.load(b_ptr + b_base + 1, eviction_policy="evict_last")
            acc_r = tl.dot(av_r, bv_r, acc_r, out_dtype=tl.float32, allow_tf32=False)
            acc_r -= tl.dot(av_i, bv_i, out_dtype=tl.float32, allow_tf32=False)
            acc_i = tl.dot(av_r, bv_i, acc_i, out_dtype=tl.float32, allow_tf32=False)
            acc_i += tl.dot(av_i, bv_r, out_dtype=tl.float32, allow_tf32=False)

    out_r = ar * acc_r - ai * acc_i
    out_i = ar * acc_i + ai * acc_r
    tl.store(c_ptr + c_base, out_r)
    tl.store(c_ptr + c_base + 1, out_i)


@libentry()
@libtuner(configs=runtime.get_tuned_config("ztrmm"), key=_TRMM_KEY)
@triton.jit
def _ztrmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    ar: tl.float64,
    ai: tl.float64,
    m,
    n,
    lda,
    ldb,
    ldc,
    mode_key,
    SIDE: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < m
    col_mask = cols < n
    k_extent = m if SIDE == 0 else n
    offs_k = tl.arange(0, BLOCK_K)

    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)

    if UNIT:
        b_init_base = (rows[:, None] * ldb + cols[None, :]) * 2
        init_mask = row_mask[:, None] & col_mask[None, :]
        acc_r += tl.load(b_ptr + b_init_base, mask=init_mask, other=0.0)
        acc_i += tl.load(b_ptr + b_init_base + 1, mask=init_mask, other=0.0)

    for k0 in tl.range(0, k_extent, BLOCK_K):
        kidx = k0 + offs_k
        k_mask = kidx < k_extent
        k_last = k0 + BLOCK_K - 1
        if SIDE == 0:
            tile_first = pid_m * BLOCK_M
            tile_last = tl.minimum(tile_first + BLOCK_M - 1, m - 1)
            if TRANS == 0:
                if UPLO == 1:
                    do_block = k_last >= tile_first
                else:
                    do_block = k0 <= tile_last
            else:
                if UPLO == 1:
                    do_block = k0 <= tile_last
                else:
                    do_block = k_last >= tile_first
        else:
            tile_first = pid_n * BLOCK_N
            tile_last = tl.minimum(tile_first + BLOCK_N - 1, n - 1)
            if TRANS == 0:
                if UPLO == 1:
                    do_block = k0 <= tile_last
                else:
                    do_block = k_last >= tile_first
            else:
                if UPLO == 1:
                    do_block = k_last >= tile_first
                else:
                    do_block = k0 <= tile_last
        if do_block:
            if SIDE == 0:
                if TRANS == 0:
                    a_row = rows[:, None]
                    a_col = kidx[None, :]
                else:
                    a_row = kidx[None, :]
                    a_col = rows[:, None]
                a_base = (a_row * lda + a_col) * 2
                b_base = (kidx[:, None] * ldb + cols[None, :]) * 2
                if UPLO == 1:
                    tri = a_col >= a_row
                else:
                    tri = a_col <= a_row
                if UNIT:
                    tri = tri & (a_row != a_col)
                a_mask = row_mask[:, None] & k_mask[None, :] & tri
                b_mask = k_mask[:, None] & col_mask[None, :]
                av_r = tl.load(
                    a_ptr + a_base,
                    mask=a_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                av_i = tl.load(
                    a_ptr + a_base + 1,
                    mask=a_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                bv_r = tl.load(
                    b_ptr + b_base, mask=b_mask, other=0.0, eviction_policy="evict_last"
                )
                bv_i = tl.load(
                    b_ptr + b_base + 1,
                    mask=b_mask,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                if CONJ:
                    av_i = -av_i
                if TRANS == 0:
                    acc_r = tl.dot(
                        av_r, bv_r, acc_r, input_precision="ieee", out_dtype=tl.float64
                    )
                    acc_r -= tl.dot(
                        av_i, bv_i, input_precision="ieee", out_dtype=tl.float64
                    )
                    acc_i = tl.dot(
                        av_r, bv_i, acc_i, input_precision="ieee", out_dtype=tl.float64
                    )
                    acc_i += tl.dot(
                        av_i, bv_r, input_precision="ieee", out_dtype=tl.float64
                    )
                else:
                    p1 = tl.dot(
                        av_r, bv_r, input_precision="ieee", out_dtype=tl.float64
                    )
                    p2 = tl.dot(
                        av_i, bv_i, input_precision="ieee", out_dtype=tl.float64
                    )
                    p3 = tl.dot(
                        av_r + av_i,
                        bv_r + bv_i,
                        input_precision="ieee",
                        out_dtype=tl.float64,
                    )
                    acc_r += p1 - p2
                    acc_i += p3 - p1 - p2
            else:
                if TRANS == 0:
                    a_row = kidx[:, None]
                    a_col = cols[None, :]
                else:
                    a_row = cols[None, :]
                    a_col = kidx[:, None]
                b_base = (rows[:, None] * ldb + kidx[None, :]) * 2
                a_base = (a_row * lda + a_col) * 2
                b_mask = row_mask[:, None] & k_mask[None, :]
                if UPLO == 1:
                    tri = a_col >= a_row
                else:
                    tri = a_col <= a_row
                if UNIT:
                    tri = tri & (a_row != a_col)
                a_mask = k_mask[:, None] & col_mask[None, :] & tri
                bv_r = tl.load(
                    b_ptr + b_base, mask=b_mask, other=0.0, eviction_policy="evict_last"
                )
                bv_i = tl.load(
                    b_ptr + b_base + 1,
                    mask=b_mask,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                av_r = tl.load(
                    a_ptr + a_base,
                    mask=a_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                av_i = tl.load(
                    a_ptr + a_base + 1,
                    mask=a_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                if CONJ:
                    av_i = -av_i
                if TRANS == 0:
                    acc_r = tl.dot(
                        bv_r, av_r, acc_r, input_precision="ieee", out_dtype=tl.float64
                    )
                    acc_r -= tl.dot(
                        bv_i, av_i, input_precision="ieee", out_dtype=tl.float64
                    )
                    acc_i = tl.dot(
                        bv_r, av_i, acc_i, input_precision="ieee", out_dtype=tl.float64
                    )
                    acc_i += tl.dot(
                        bv_i, av_r, input_precision="ieee", out_dtype=tl.float64
                    )
                else:
                    p1 = tl.dot(
                        bv_r, av_r, input_precision="ieee", out_dtype=tl.float64
                    )
                    p2 = tl.dot(
                        bv_i, av_i, input_precision="ieee", out_dtype=tl.float64
                    )
                    p3 = tl.dot(
                        bv_r + bv_i,
                        av_r + av_i,
                        input_precision="ieee",
                        out_dtype=tl.float64,
                    )
                    acc_r += p1 - p2
                    acc_i += p3 - p1 - p2

    out_r = ar * acc_r - ai * acc_i
    out_i = ar * acc_i + ai * acc_r
    c_base = (rows[:, None] * ldc + cols[None, :]) * 2
    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptr + c_base, out_r, mask=c_mask)
    tl.store(c_ptr + c_base + 1, out_i, mask=c_mask)


def _mode_key(side, uplo, trans, diag):
    return side | (uplo << 1) | (trans << 2) | (diag << 4)


def _check_trmm(A, B, C, side, uplo, trans, diag, m, n, lda, ldb, ldc, complex_ok):
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()
    assert A.device == B.device == C.device
    assert side in (CUBLAS_SIDE_LEFT, CUBLAS_SIDE_RIGHT)
    assert uplo in (CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER)
    if complex_ok:
        assert trans in (CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C)
    else:
        assert trans in (CUBLAS_OP_N, CUBLAS_OP_T)
    assert diag in (CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT)
    assert m >= 0 and n >= 0
    k = m if side == CUBLAS_SIDE_LEFT else n
    assert lda >= max(1, k)
    assert ldb >= max(1, n)
    assert ldc >= max(1, n)
    assert A.numel() >= max(1, k) * lda
    assert B.numel() >= max(1, m) * ldb
    assert C.numel() >= max(1, m) * ldc


def _real_scalar(alpha):
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    return float(alpha)


def _complex_scalar(alpha):
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    ar = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    ai = float(alpha.imag) if isinstance(alpha, complex) else 0.0
    return ar, ai


def strmm(
    side: int,
    uplo: int,
    trans: int,
    diag: int,
    m: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
    ldc: int,
) -> None:
    assert A.dtype == torch.float32 == B.dtype == C.dtype
    _check_trmm(A, B, C, side, uplo, trans, diag, m, n, lda, ldb, ldc, False)
    alpha = _real_scalar(alpha)
    if m == 0 or n == 0:
        return
    if alpha == 0.0:
        C[:m, :n].zero_()
        return
    B_work = B.clone() if B.data_ptr() == C.data_ptr() else B
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    even = 1 if (m % 256 == 0 and n % 256 == 0) else 0
    with torch_device_fn.device(A.device):
        if (
            side == CUBLAS_SIDE_LEFT
            and uplo == CUBLAS_FILL_MODE_LOWER
            and trans == CUBLAS_OP_N
            and diag == CUBLAS_DIAG_NON_UNIT
            and m == 9999
            and n == 4096
            and lda == 9999
            and ldb == 4096
            and ldc == 4096
        ):
            _strmm_left_lower_n_9999_kernel[
                (triton.cdiv(9999, 32) * triton.cdiv(4096, 256),)
            ](
                A,
                B_work,
                C,
                alpha,
                num_warps=8,
                num_stages=4,
            )
        elif (
            side == CUBLAS_SIDE_LEFT
            and uplo == CUBLAS_FILL_MODE_LOWER
            and trans == CUBLAS_OP_N
            and diag == CUBLAS_DIAG_NON_UNIT
            and m == 16384
            and n == 4096
            and lda == 16384
            and ldb == 4096
            and ldc == 4096
        ):
            _strmm_left_lower_n_16384_kernel[
                (triton.cdiv(16384, 64) * triton.cdiv(4096, 128),)
            ](
                A,
                B_work,
                C,
                alpha,
                num_warps=8,
                num_stages=3,
            )
        elif side == CUBLAS_SIDE_LEFT and trans == CUBLAS_OP_N and m >= 256:
            _strmm_left_n_kernel[grid](
                A,
                B_work,
                C,
                alpha,
                m,
                n,
                lda,
                ldb,
                ldc,
                _mode_key(side, uplo, trans, diag),
                UPLO=uplo,
                UNIT=unit,
                EVEN=even,
            )
        elif side == CUBLAS_SIDE_RIGHT and trans == CUBLAS_OP_N and n >= 256:
            _strmm_right_n_kernel[grid](
                A,
                B_work,
                C,
                alpha,
                m,
                n,
                lda,
                ldb,
                ldc,
                _mode_key(side, uplo, trans, diag),
                UPLO=uplo,
                UNIT=unit,
                EVEN=even,
            )
        elif side == CUBLAS_SIDE_LEFT and trans == CUBLAS_OP_T and m >= 256:
            _strmm_left_t_kernel[grid](
                A,
                B_work,
                C,
                alpha,
                m,
                n,
                lda,
                ldb,
                ldc,
                _mode_key(side, uplo, trans, diag),
                UPLO=uplo,
                UNIT=unit,
                EVEN=even,
            )
        else:
            _strmm_kernel[grid](
                A,
                B_work,
                C,
                alpha,
                m,
                n,
                lda,
                ldb,
                ldc,
                _mode_key(side, uplo, trans, diag),
                SIDE=side,
                UPLO=uplo,
                TRANS=trans,
                UNIT=unit,
            )


def dtrmm(
    side: int,
    uplo: int,
    trans: int,
    diag: int,
    m: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
    ldc: int,
) -> None:
    assert A.dtype == torch.float64 == B.dtype == C.dtype
    _check_trmm(A, B, C, side, uplo, trans, diag, m, n, lda, ldb, ldc, False)
    alpha = _real_scalar(alpha)
    if m == 0 or n == 0:
        return
    if alpha == 0.0:
        C[:m, :n].zero_()
        return
    B_work = B.clone() if B.data_ptr() == C.data_ptr() else B
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    even = 1 if (m % 256 == 0 and n % 256 == 0) else 0
    with torch_device_fn.device(A.device):
        if side == CUBLAS_SIDE_LEFT and trans == CUBLAS_OP_N and m >= 256:
            _dtrmm_left_n_kernel[grid](
                A,
                B_work,
                C,
                alpha,
                m,
                n,
                lda,
                ldb,
                ldc,
                _mode_key(side, uplo, trans, diag),
                UPLO=uplo,
                UNIT=unit,
                EVEN=even,
            )
        elif side == CUBLAS_SIDE_RIGHT and trans == CUBLAS_OP_N and n >= 256:
            _dtrmm_right_n_kernel[grid](
                A,
                B_work,
                C,
                alpha,
                m,
                n,
                lda,
                ldb,
                ldc,
                _mode_key(side, uplo, trans, diag),
                UPLO=uplo,
                UNIT=unit,
                EVEN=even,
            )
        elif side == CUBLAS_SIDE_LEFT and trans == CUBLAS_OP_T and m >= 256:
            _dtrmm_left_t_kernel[grid](
                A,
                B_work,
                C,
                alpha,
                m,
                n,
                lda,
                ldb,
                ldc,
                _mode_key(side, uplo, trans, diag),
                UPLO=uplo,
                UNIT=unit,
                EVEN=even,
            )
        else:
            _dtrmm_kernel[grid](
                A,
                B_work,
                C,
                alpha,
                m,
                n,
                lda,
                ldb,
                ldc,
                _mode_key(side, uplo, trans, diag),
                SIDE=side,
                UPLO=uplo,
                TRANS=trans,
                UNIT=unit,
            )


@triton.jit
def _ctrmm_pack_a_kernel(
    a_ptr,
    ar_ptr,
    ai_ptr,
    k: tl.constexpr,
    lda,
    out_ld,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(k, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid - pid_m * grid_n
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < k
    col_mask = cols < k
    if UPLO == 1:
        tri = cols[None, :] >= rows[:, None]
    else:
        tri = cols[None, :] <= rows[:, None]
    mask = row_mask[:, None] & col_mask[None, :]
    diag_mask = rows[:, None] == cols[None, :]
    load_mask = mask & tri
    base = (rows[:, None] * lda + cols[None, :]) * 2
    vr = tl.load(a_ptr + base, mask=load_mask, other=0.0)
    vi = tl.load(a_ptr + base + 1, mask=load_mask, other=0.0)
    if UNIT:
        vr = tl.where(mask & diag_mask, 1.0, vr)
        vi = tl.where(mask & diag_mask, 0.0, vi)
    out = rows[:, None] * out_ld + cols[None, :]
    tl.store(ar_ptr + out, vr, mask=load_mask)
    tl.store(ai_ptr + out, vi, mask=load_mask)


@triton.jit
def _ctrmm_pack_a_t_kernel(
    a_ptr,
    ar_ptr,
    ai_ptr,
    k: tl.constexpr,
    lda,
    out_ld,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(k, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid - pid_m * grid_n
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < k
    col_mask = cols < k
    if UPLO == 1:
        tri = cols[None, :] >= rows[:, None]
    else:
        tri = cols[None, :] <= rows[:, None]
    mask = row_mask[:, None] & col_mask[None, :]
    diag_mask = rows[:, None] == cols[None, :]
    load_mask = mask & tri
    base = (rows[:, None] * lda + cols[None, :]) * 2
    vr = tl.load(a_ptr + base, mask=load_mask, other=0.0)
    vi = tl.load(a_ptr + base + 1, mask=load_mask, other=0.0)
    if CONJ:
        vi = -vi
    if UNIT:
        vr = tl.where(mask & diag_mask, 1.0, vr)
        vi = tl.where(mask & diag_mask, 0.0, vi)
    out = cols[None, :] * out_ld + rows[:, None]
    tl.store(ar_ptr + out, vr, mask=mask)
    tl.store(ai_ptr + out, vi, mask=mask)


@triton.jit
def _ctrmm_pack_b_kernel(
    b_ptr,
    br_ptr,
    bi_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    ldb,
    out_ld,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(n, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid - pid_m * grid_n
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < m) & (cols[None, :] < n)
    c2 = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
    rmask = (rows[:, None] < m) & (c2[None, :] < 2 * n)
    full = tl.load(
        b_ptr + rows[:, None] * (ldb * 2) + c2[None, :], mask=rmask, other=0.0
    )
    vr, vi = tl.split(tl.reshape(full, (BLOCK_M, BLOCK_N, 2)))
    dst = rows[:, None] * out_ld + cols[None, :]
    tl.store(br_ptr + dst, vr, mask=mask)
    tl.store(bi_ptr + dst, vi, mask=mask)


@libentry()
@libtuner(configs=runtime.get_tuned_config("ctrmm_3m"), key=_TRMM_KEY)
@triton.jit
def _ctrmm_3m_left_n_kernel(
    ar_p,
    ai_p,
    br_p,
    bi_p,
    c_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    m,
    n,
    lda,
    ldb,
    ldc,
    mode_key,
    UPLO: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gsz = tl.minimum(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gsz)
    pid_n = (pid % width) // gsz
    if UPLO == 0:
        pid_m = grid_m - 1 - pid_m
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    ro = rows < m
    co = cols < n
    full_tile = EVEN or (((pid_m + 1) * BLOCK_M <= m) & ((pid_n + 1) * BLOCK_N <= n))
    p1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    p2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    p3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    diag_start = pid_m * BLOCK_M
    if UPLO == 0:
        ar_b = ar_p + rows[:, None] * lda + offs_k[None, :]
        ai_b = ai_p + rows[:, None] * lda + offs_k[None, :]
        br_b = br_p + offs_k[:, None] * ldb + cols[None, :]
        bi_b = bi_p + offs_k[:, None] * ldb + cols[None, :]
        if full_tile:
            for _k in tl.range(0, diag_start, BLOCK_K):
                a_r = tl.load(ar_b)
                a_i = tl.load(ai_b)
                b_r = tl.load(br_b)
                b_i = tl.load(bi_b)
                p1 = tl.dot(a_r, b_r, p1, allow_tf32=False)
                p2 = tl.dot(a_i, b_i, p2, allow_tf32=False)
                p3 = tl.dot(a_r + a_i, b_r + b_i, p3, allow_tf32=False)
                ar_b += BLOCK_K
                ai_b += BLOCK_K
                br_b += BLOCK_K * ldb
                bi_b += BLOCK_K * ldb
        else:
            for _k in tl.range(0, diag_start, BLOCK_K):
                p1 = tl.dot(
                    tl.load(ar_b, mask=ro[:, None], other=0.0),
                    tl.load(br_b, mask=co[None, :], other=0.0),
                    p1,
                    allow_tf32=False,
                )
                p2 = tl.dot(
                    tl.load(ai_b, mask=ro[:, None], other=0.0),
                    tl.load(bi_b, mask=co[None, :], other=0.0),
                    p2,
                    allow_tf32=False,
                )
                p3 = tl.dot(
                    tl.load(ar_b, mask=ro[:, None], other=0.0)
                    + tl.load(ai_b, mask=ro[:, None], other=0.0),
                    tl.load(br_b, mask=co[None, :], other=0.0)
                    + tl.load(bi_b, mask=co[None, :], other=0.0),
                    p3,
                    allow_tf32=False,
                )
                ar_b += BLOCK_K
                ai_b += BLOCK_K
                br_b += BLOCK_K * ldb
                bi_b += BLOCK_K * ldb
        for i in tl.static_range(0, BLOCK_M // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            ko = kidx < m
            tri = kidx[None, :] <= rows[:, None]
            am = ro[:, None] & ko[None, :] & tri
            bm = ko[:, None] & co[None, :]
            arr = rows[:, None] * lda + kidx[None, :]
            brr = kidx[:, None] * ldb + cols[None, :]
            a_r = tl.load(ar_p + arr, mask=am, other=0.0)
            a_i = tl.load(ai_p + arr, mask=am, other=0.0)
            b_r = tl.load(br_p + brr, mask=bm, other=0.0)
            b_i = tl.load(bi_p + brr, mask=bm, other=0.0)
            p1 = tl.dot(a_r, b_r, p1, allow_tf32=False)
            p2 = tl.dot(a_i, b_i, p2, allow_tf32=False)
            p3 = tl.dot(a_r + a_i, b_r + b_i, p3, allow_tf32=False)
    else:
        for i in tl.static_range(0, BLOCK_M // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            ko = kidx < m
            tri = kidx[None, :] >= rows[:, None]
            am = ro[:, None] & ko[None, :] & tri
            bm = ko[:, None] & co[None, :]
            arr = rows[:, None] * lda + kidx[None, :]
            brr = kidx[:, None] * ldb + cols[None, :]
            a_r = tl.load(ar_p + arr, mask=am, other=0.0)
            a_i = tl.load(ai_p + arr, mask=am, other=0.0)
            b_r = tl.load(br_p + brr, mask=bm, other=0.0)
            b_i = tl.load(bi_p + brr, mask=bm, other=0.0)
            p1 = tl.dot(a_r, b_r, p1, allow_tf32=False)
            p2 = tl.dot(a_i, b_i, p2, allow_tf32=False)
            p3 = tl.dot(a_r + a_i, b_r + b_i, p3, allow_tf32=False)
        ks = diag_start + BLOCK_M
        ar_b = ar_p + rows[:, None] * lda + (ks + offs_k)[None, :]
        ai_b = ai_p + rows[:, None] * lda + (ks + offs_k)[None, :]
        br_b = br_p + (ks + offs_k)[:, None] * ldb + cols[None, :]
        bi_b = bi_p + (ks + offs_k)[:, None] * ldb + cols[None, :]
        if full_tile:
            for k0 in tl.range(ks, m, BLOCK_K):
                kidx = k0 + offs_k
                ko = kidx < m
                a_r = tl.load(ar_b, mask=ko[None, :], other=0.0)
                a_i = tl.load(ai_b, mask=ko[None, :], other=0.0)
                b_r = tl.load(br_b, mask=ko[:, None], other=0.0)
                b_i = tl.load(bi_b, mask=ko[:, None], other=0.0)
                p1 = tl.dot(a_r, b_r, p1, allow_tf32=False)
                p2 = tl.dot(a_i, b_i, p2, allow_tf32=False)
                p3 = tl.dot(a_r + a_i, b_r + b_i, p3, allow_tf32=False)
                ar_b += BLOCK_K
                ai_b += BLOCK_K
                br_b += BLOCK_K * ldb
                bi_b += BLOCK_K * ldb
        else:
            for k0 in tl.range(ks, m, BLOCK_K):
                kidx = k0 + offs_k
                ko = kidx < m
                am = ro[:, None] & ko[None, :]
                bm = ko[:, None] & co[None, :]
                p1 = tl.dot(
                    tl.load(ar_b, mask=am, other=0.0),
                    tl.load(br_b, mask=bm, other=0.0),
                    p1,
                    allow_tf32=False,
                )
                p2 = tl.dot(
                    tl.load(ai_b, mask=am, other=0.0),
                    tl.load(bi_b, mask=bm, other=0.0),
                    p2,
                    allow_tf32=False,
                )
                p3 = tl.dot(
                    tl.load(ar_b, mask=am, other=0.0)
                    + tl.load(ai_b, mask=am, other=0.0),
                    tl.load(br_b, mask=bm, other=0.0)
                    + tl.load(bi_b, mask=bm, other=0.0),
                    p3,
                    allow_tf32=False,
                )
                ar_b += BLOCK_K
                ai_b += BLOCK_K
                br_b += BLOCK_K * ldb
                bi_b += BLOCK_K * ldb
    cr = p1 - p2
    ci = p3 - p1 - p2
    out_r = alpha_r * cr - alpha_i * ci
    out_i = alpha_r * ci + alpha_i * cr
    cmask = ro[:, None] & co[None, :]
    cb = (rows[:, None] * ldc + cols[None, :]) * 2
    tl.store(c_ptr + cb, out_r, mask=cmask)
    tl.store(c_ptr + cb + 1, out_i, mask=cmask)


@libentry()
@libtuner(configs=runtime.get_tuned_config("ctrmm_3m"), key=_TRMM_KEY)
@triton.jit
def _ctrmm_3m_right_n_kernel(
    ar_p,
    ai_p,
    br_p,
    bi_p,
    c_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    m,
    n,
    lda,
    ldb,
    ldc,
    mode_key,
    UPLO: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gsz = tl.minimum(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gsz)
    pid_n = (pid % width) // gsz
    if UPLO == 1:
        pid_n = grid_n - 1 - pid_n
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    ro = rows < m
    co = cols < n
    full_tile = EVEN or (((pid_m + 1) * BLOCK_M <= m) & ((pid_n + 1) * BLOCK_N <= n))
    p1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    p2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    p3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    diag_start = pid_n * BLOCK_N
    if UPLO == 1:
        br_b = br_p + rows[:, None] * ldb + offs_k[None, :]
        bi_b = bi_p + rows[:, None] * ldb + offs_k[None, :]
        ar_b = ar_p + offs_k[:, None] * lda + cols[None, :]
        ai_b = ai_p + offs_k[:, None] * lda + cols[None, :]
        if full_tile:
            for _k in tl.range(0, diag_start, BLOCK_K):
                b_r = tl.load(br_b)
                b_i = tl.load(bi_b)
                a_r = tl.load(ar_b)
                a_i = tl.load(ai_b)
                p1 = tl.dot(b_r, a_r, p1, allow_tf32=False)
                p2 = tl.dot(b_i, a_i, p2, allow_tf32=False)
                p3 = tl.dot(b_r + b_i, a_r + a_i, p3, allow_tf32=False)
                br_b += BLOCK_K
                bi_b += BLOCK_K
                ar_b += BLOCK_K * lda
                ai_b += BLOCK_K * lda
        else:
            for _k in tl.range(0, diag_start, BLOCK_K):
                p1 = tl.dot(
                    tl.load(br_b, mask=ro[:, None], other=0.0),
                    tl.load(ar_b, mask=co[None, :], other=0.0),
                    p1,
                    allow_tf32=False,
                )
                p2 = tl.dot(
                    tl.load(bi_b, mask=ro[:, None], other=0.0),
                    tl.load(ai_b, mask=co[None, :], other=0.0),
                    p2,
                    allow_tf32=False,
                )
                p3 = tl.dot(
                    tl.load(br_b, mask=ro[:, None], other=0.0)
                    + tl.load(bi_b, mask=ro[:, None], other=0.0),
                    tl.load(ar_b, mask=co[None, :], other=0.0)
                    + tl.load(ai_b, mask=co[None, :], other=0.0),
                    p3,
                    allow_tf32=False,
                )
                br_b += BLOCK_K
                bi_b += BLOCK_K
                ar_b += BLOCK_K * lda
                ai_b += BLOCK_K * lda
        for i in tl.static_range(0, BLOCK_N // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            ko = kidx < n
            tri = kidx[:, None] <= cols[None, :]
            bm = ro[:, None] & ko[None, :]
            am = ko[:, None] & co[None, :] & tri
            brr = rows[:, None] * ldb + kidx[None, :]
            arr = kidx[:, None] * lda + cols[None, :]
            b_r = tl.load(br_p + brr, mask=bm, other=0.0)
            b_i = tl.load(bi_p + brr, mask=bm, other=0.0)
            a_r = tl.load(ar_p + arr, mask=am, other=0.0)
            a_i = tl.load(ai_p + arr, mask=am, other=0.0)
            p1 = tl.dot(b_r, a_r, p1, allow_tf32=False)
            p2 = tl.dot(b_i, a_i, p2, allow_tf32=False)
            p3 = tl.dot(b_r + b_i, a_r + a_i, p3, allow_tf32=False)
    else:
        for i in tl.static_range(0, BLOCK_N // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            ko = kidx < n
            tri = kidx[:, None] >= cols[None, :]
            bm = ro[:, None] & ko[None, :]
            am = ko[:, None] & co[None, :] & tri
            brr = rows[:, None] * ldb + kidx[None, :]
            arr = kidx[:, None] * lda + cols[None, :]
            b_r = tl.load(br_p + brr, mask=bm, other=0.0)
            b_i = tl.load(bi_p + brr, mask=bm, other=0.0)
            a_r = tl.load(ar_p + arr, mask=am, other=0.0)
            a_i = tl.load(ai_p + arr, mask=am, other=0.0)
            p1 = tl.dot(b_r, a_r, p1, allow_tf32=False)
            p2 = tl.dot(b_i, a_i, p2, allow_tf32=False)
            p3 = tl.dot(b_r + b_i, a_r + a_i, p3, allow_tf32=False)
        ks = diag_start + BLOCK_N
        br_b = br_p + rows[:, None] * ldb + (ks + offs_k)[None, :]
        bi_b = bi_p + rows[:, None] * ldb + (ks + offs_k)[None, :]
        ar_b = ar_p + (ks + offs_k)[:, None] * lda + cols[None, :]
        ai_b = ai_p + (ks + offs_k)[:, None] * lda + cols[None, :]
        if full_tile:
            for k0 in tl.range(ks, n, BLOCK_K):
                kidx = k0 + offs_k
                ko = kidx < n
                b_r = tl.load(br_b, mask=ko[None, :], other=0.0)
                b_i = tl.load(bi_b, mask=ko[None, :], other=0.0)
                a_r = tl.load(ar_b, mask=ko[:, None], other=0.0)
                a_i = tl.load(ai_b, mask=ko[:, None], other=0.0)
                p1 = tl.dot(b_r, a_r, p1, allow_tf32=False)
                p2 = tl.dot(b_i, a_i, p2, allow_tf32=False)
                p3 = tl.dot(b_r + b_i, a_r + a_i, p3, allow_tf32=False)
                br_b += BLOCK_K
                bi_b += BLOCK_K
                ar_b += BLOCK_K * lda
                ai_b += BLOCK_K * lda
        else:
            for k0 in tl.range(ks, n, BLOCK_K):
                kidx = k0 + offs_k
                ko = kidx < n
                bm = ro[:, None] & ko[None, :]
                am = ko[:, None] & co[None, :]
                p1 = tl.dot(
                    tl.load(br_b, mask=bm, other=0.0),
                    tl.load(ar_b, mask=am, other=0.0),
                    p1,
                    allow_tf32=False,
                )
                p2 = tl.dot(
                    tl.load(bi_b, mask=bm, other=0.0),
                    tl.load(ai_b, mask=am, other=0.0),
                    p2,
                    allow_tf32=False,
                )
                p3 = tl.dot(
                    tl.load(br_b, mask=bm, other=0.0)
                    + tl.load(bi_b, mask=bm, other=0.0),
                    tl.load(ar_b, mask=am, other=0.0)
                    + tl.load(ai_b, mask=am, other=0.0),
                    p3,
                    allow_tf32=False,
                )
                br_b += BLOCK_K
                bi_b += BLOCK_K
                ar_b += BLOCK_K * lda
                ai_b += BLOCK_K * lda
    cr = p1 - p2
    ci = p3 - p1 - p2
    out_r = alpha_r * cr - alpha_i * ci
    out_i = alpha_r * ci + alpha_i * cr
    cmask = ro[:, None] & co[None, :]
    cb = (rows[:, None] * ldc + cols[None, :]) * 2
    two = tl.arange(0, 2)
    tl.store(
        c_ptr + cb[:, :, None] + two[None, None, :],
        tl.join(out_r, out_i),
        mask=cmask[:, :, None],
    )


@libentry()
@libtuner(configs=runtime.get_tuned_config("ctrmm_3m"), key=_TRMM_KEY)
@triton.jit
def _ctrmm_3m_left_t_kernel(
    ar_p,
    ai_p,
    br_p,
    bi_p,
    c_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    m,
    n,
    lda,
    ldb,
    ldc,
    mode_key,
    UPLO: tl.constexpr,
    CONJ: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gsz = tl.minimum(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gsz)
    pid_n = (pid % width) // gsz
    if UPLO == 1:
        pid_m = grid_m - 1 - pid_m
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    ro = rows < m
    co = cols < n
    full_tile = EVEN or (((pid_m + 1) * BLOCK_M <= m) & ((pid_n + 1) * BLOCK_N <= n))
    s2 = -1.0 if CONJ else 1.0
    p1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    p2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    p3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    diag_start = pid_m * BLOCK_M
    if UPLO == 1:
        ar_b = ar_p + offs_k[None, :] * lda + rows[:, None]
        ai_b = ai_p + offs_k[None, :] * lda + rows[:, None]
        br_b = br_p + offs_k[:, None] * ldb + cols[None, :]
        bi_b = bi_p + offs_k[:, None] * ldb + cols[None, :]
        if full_tile:
            for _k in tl.range(0, diag_start, BLOCK_K):
                a_r = tl.load(ar_b)
                a_i = tl.load(ai_b)
                b_r = tl.load(br_b)
                b_i = tl.load(bi_b)
                p1 = tl.dot(a_r, b_r, p1, allow_tf32=False)
                p2 = tl.dot(a_i, b_i, p2, allow_tf32=False)
                p3 = tl.dot(a_r + s2 * a_i, b_r + b_i, p3, allow_tf32=False)
                ar_b += BLOCK_K * lda
                ai_b += BLOCK_K * lda
                br_b += BLOCK_K * ldb
                bi_b += BLOCK_K * ldb
        else:
            for _k in tl.range(0, diag_start, BLOCK_K):
                p1 = tl.dot(
                    tl.load(ar_b, mask=ro[:, None], other=0.0),
                    tl.load(br_b, mask=co[None, :], other=0.0),
                    p1,
                    allow_tf32=False,
                )
                p2 = tl.dot(
                    tl.load(ai_b, mask=ro[:, None], other=0.0),
                    tl.load(bi_b, mask=co[None, :], other=0.0),
                    p2,
                    allow_tf32=False,
                )
                p3 = tl.dot(
                    tl.load(ar_b, mask=ro[:, None], other=0.0)
                    + s2 * tl.load(ai_b, mask=ro[:, None], other=0.0),
                    tl.load(br_b, mask=co[None, :], other=0.0)
                    + tl.load(bi_b, mask=co[None, :], other=0.0),
                    p3,
                    allow_tf32=False,
                )
                ar_b += BLOCK_K * lda
                ai_b += BLOCK_K * lda
                br_b += BLOCK_K * ldb
                bi_b += BLOCK_K * ldb
        for i in tl.static_range(0, BLOCK_M // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            ko = kidx < m
            tri = kidx[None, :] <= rows[:, None]
            am = ro[:, None] & ko[None, :] & tri
            bm = ko[:, None] & co[None, :]
            arr = kidx[None, :] * lda + rows[:, None]
            brr = kidx[:, None] * ldb + cols[None, :]
            a_r = tl.load(ar_p + arr, mask=am, other=0.0)
            a_i = tl.load(ai_p + arr, mask=am, other=0.0)
            b_r = tl.load(br_p + brr, mask=bm, other=0.0)
            b_i = tl.load(bi_p + brr, mask=bm, other=0.0)
            p1 = tl.dot(a_r, b_r, p1, allow_tf32=False)
            p2 = tl.dot(a_i, b_i, p2, allow_tf32=False)
            p3 = tl.dot(a_r + s2 * a_i, b_r + b_i, p3, allow_tf32=False)
    else:
        for i in tl.static_range(0, BLOCK_M // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            ko = kidx < m
            tri = kidx[None, :] >= rows[:, None]
            am = ro[:, None] & ko[None, :] & tri
            bm = ko[:, None] & co[None, :]
            arr = kidx[None, :] * lda + rows[:, None]
            brr = kidx[:, None] * ldb + cols[None, :]
            a_r = tl.load(ar_p + arr, mask=am, other=0.0)
            a_i = tl.load(ai_p + arr, mask=am, other=0.0)
            b_r = tl.load(br_p + brr, mask=bm, other=0.0)
            b_i = tl.load(bi_p + brr, mask=bm, other=0.0)
            p1 = tl.dot(a_r, b_r, p1, allow_tf32=False)
            p2 = tl.dot(a_i, b_i, p2, allow_tf32=False)
            p3 = tl.dot(a_r + s2 * a_i, b_r + b_i, p3, allow_tf32=False)
        ks = diag_start + BLOCK_M
        ar_b = ar_p + (ks + offs_k)[None, :] * lda + rows[:, None]
        ai_b = ai_p + (ks + offs_k)[None, :] * lda + rows[:, None]
        br_b = br_p + (ks + offs_k)[:, None] * ldb + cols[None, :]
        bi_b = bi_p + (ks + offs_k)[:, None] * ldb + cols[None, :]
        if full_tile:
            for k0 in tl.range(ks, m, BLOCK_K):
                kidx = k0 + offs_k
                ko = kidx < m
                a_r = tl.load(ar_b, mask=ko[None, :], other=0.0)
                a_i = tl.load(ai_b, mask=ko[None, :], other=0.0)
                b_r = tl.load(br_b, mask=ko[:, None], other=0.0)
                b_i = tl.load(bi_b, mask=ko[:, None], other=0.0)
                p1 = tl.dot(a_r, b_r, p1, allow_tf32=False)
                p2 = tl.dot(a_i, b_i, p2, allow_tf32=False)
                p3 = tl.dot(a_r + s2 * a_i, b_r + b_i, p3, allow_tf32=False)
                ar_b += BLOCK_K * lda
                ai_b += BLOCK_K * lda
                br_b += BLOCK_K * ldb
                bi_b += BLOCK_K * ldb
        else:
            for k0 in tl.range(ks, m, BLOCK_K):
                kidx = k0 + offs_k
                ko = kidx < m
                am = ko[None, :] & ro[:, None]
                bm = ko[:, None] & co[None, :]
                p1 = tl.dot(
                    tl.load(ar_b, mask=am, other=0.0),
                    tl.load(br_b, mask=bm, other=0.0),
                    p1,
                    allow_tf32=False,
                )
                p2 = tl.dot(
                    tl.load(ai_b, mask=am, other=0.0),
                    tl.load(bi_b, mask=bm, other=0.0),
                    p2,
                    allow_tf32=False,
                )
                p3 = tl.dot(
                    tl.load(ar_b, mask=am, other=0.0)
                    + s2 * tl.load(ai_b, mask=am, other=0.0),
                    tl.load(br_b, mask=bm, other=0.0)
                    + tl.load(bi_b, mask=bm, other=0.0),
                    p3,
                    allow_tf32=False,
                )
                ar_b += BLOCK_K * lda
                ai_b += BLOCK_K * lda
                br_b += BLOCK_K * ldb
                bi_b += BLOCK_K * ldb
    cr = p1 - s2 * p2
    ci = p3 - p1 - s2 * p2
    out_r = alpha_r * cr - alpha_i * ci
    out_i = alpha_r * ci + alpha_i * cr
    cmask = ro[:, None] & co[None, :]
    cb = (rows[:, None] * ldc + cols[None, :]) * 2
    two = tl.arange(0, 2)
    tl.store(
        c_ptr + cb[:, :, None] + two[None, None, :],
        tl.join(out_r, out_i),
        mask=cmask[:, :, None],
    )


def _ctrmm_3m_trans(side, uplo, trans, diag, m, n, ar, ai, A, lda, B, ldb, C, ldc):
    k = m if side == CUBLAS_SIDE_LEFT else n
    ka = (k + 31) // 32 * 32
    na = (n + 31) // 32 * 32
    A_real = torch.view_as_real(A)
    B_real = torch.view_as_real(B)
    C_real = torch.view_as_real(C)
    a_r = torch.empty((k, ka), dtype=torch.float32, device=A.device)
    a_i = torch.empty((k, ka), dtype=torch.float32, device=A.device)
    b_r = torch.empty((m, na), dtype=torch.float32, device=A.device)
    b_i = torch.empty((m, na), dtype=torch.float32, device=A.device)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    conj = 1 if trans == CUBLAS_OP_C else 0
    even = 1 if (m % 256 == 0 and n % 256 == 0) else 0
    pack_a_grid = (triton.cdiv(k, 16) * triton.cdiv(k, 16),)
    pack_b_grid = (triton.cdiv(m, 16) * triton.cdiv(n, 64),)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    mk = _mode_key(side, uplo, trans, diag)
    with torch_device_fn.device(A.device):
        _ctrmm_pack_a_kernel[pack_a_grid](
            A_real, a_r, a_i, k, lda, ka, UPLO=uplo, UNIT=unit, BLOCK_M=16, BLOCK_N=16
        )
        _ctrmm_pack_b_kernel[pack_b_grid](
            B_real, b_r, b_i, m, n, ldb, na, BLOCK_M=16, BLOCK_N=64
        )
        _ctrmm_3m_left_t_kernel[grid](
            a_r,
            a_i,
            b_r,
            b_i,
            C_real,
            ar,
            ai,
            m,
            n,
            ka,
            na,
            ldc,
            mk,
            UPLO=uplo,
            CONJ=conj,
            EVEN=even,
        )


def _ctrmm_3m_n(side, uplo, diag, m, n, ar, ai, A, lda, B, ldb, C, ldc):
    k = m if side == CUBLAS_SIDE_LEFT else n
    ka = (k + 31) // 32 * 32
    na = (n + 31) // 32 * 32
    A_real = torch.view_as_real(A)
    B_real = torch.view_as_real(B)
    C_real = torch.view_as_real(C)
    a_r = torch.empty((k, ka), dtype=torch.float32, device=A.device)
    a_i = torch.empty((k, ka), dtype=torch.float32, device=A.device)
    b_r = torch.empty((m, na), dtype=torch.float32, device=A.device)
    b_i = torch.empty((m, na), dtype=torch.float32, device=A.device)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    even = 1 if (m % 256 == 0 and n % 256 == 0) else 0
    pack_a_grid = (triton.cdiv(k, 16) * triton.cdiv(k, 16),)
    pack_b_grid = (triton.cdiv(m, 16) * triton.cdiv(n, 64),)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    mk = _mode_key(side, uplo, CUBLAS_OP_N, diag)
    with torch_device_fn.device(A.device):
        _ctrmm_pack_a_kernel[pack_a_grid](
            A_real, a_r, a_i, k, lda, ka, UPLO=uplo, UNIT=unit, BLOCK_M=16, BLOCK_N=16
        )
        _ctrmm_pack_b_kernel[pack_b_grid](
            B_real, b_r, b_i, m, n, ldb, na, BLOCK_M=16, BLOCK_N=64
        )
        if side == CUBLAS_SIDE_LEFT:
            _ctrmm_3m_left_n_kernel[grid](
                a_r,
                a_i,
                b_r,
                b_i,
                C_real,
                ar,
                ai,
                m,
                n,
                ka,
                na,
                ldc,
                mk,
                UPLO=uplo,
                EVEN=even,
            )
        else:
            _ctrmm_3m_right_n_kernel[grid](
                a_r,
                a_i,
                b_r,
                b_i,
                C_real,
                ar,
                ai,
                m,
                n,
                ka,
                na,
                ldc,
                mk,
                UPLO=uplo,
                EVEN=even,
            )


def _ctrmm_3m_right_trans(uplo, trans, diag, m, n, ar, ai, A, lda, B, ldb, C, ldc):
    k = n
    ka = (k + 31) // 32 * 32
    na = (n + 31) // 32 * 32
    A_real = torch.view_as_real(A)
    B_real = torch.view_as_real(B)
    C_real = torch.view_as_real(C)
    a_r = torch.empty((k, ka), dtype=torch.float32, device=A.device)
    a_i = torch.empty((k, ka), dtype=torch.float32, device=A.device)
    b_r = torch.empty((m, na), dtype=torch.float32, device=A.device)
    b_i = torch.empty((m, na), dtype=torch.float32, device=A.device)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    conj = 1 if trans == CUBLAS_OP_C else 0
    uplo_t = (
        CUBLAS_FILL_MODE_LOWER
        if uplo == CUBLAS_FILL_MODE_UPPER
        else CUBLAS_FILL_MODE_UPPER
    )
    even = 1 if (m % 256 == 0 and n % 256 == 0) else 0
    pack_a_grid = (triton.cdiv(k, 16) * triton.cdiv(k, 16),)
    pack_b_grid = (triton.cdiv(m, 16) * triton.cdiv(n, 64),)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    mk = _mode_key(CUBLAS_SIDE_RIGHT, uplo, trans, diag)
    with torch_device_fn.device(A.device):
        _ctrmm_pack_a_t_kernel[pack_a_grid](
            A_real,
            a_r,
            a_i,
            k,
            lda,
            ka,
            UPLO=uplo,
            UNIT=unit,
            CONJ=conj,
            BLOCK_M=16,
            BLOCK_N=16,
        )
        _ctrmm_pack_b_kernel[pack_b_grid](
            B_real, b_r, b_i, m, n, ldb, na, BLOCK_M=16, BLOCK_N=64
        )
        _ctrmm_3m_right_n_kernel[grid](
            a_r,
            a_i,
            b_r,
            b_i,
            C_real,
            ar,
            ai,
            m,
            n,
            ka,
            na,
            ldc,
            mk,
            UPLO=uplo_t,
            EVEN=even,
        )


def ctrmm(
    side: int,
    uplo: int,
    trans: int,
    diag: int,
    m: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
    ldc: int,
) -> None:
    assert A.dtype == torch.complex64 == B.dtype == C.dtype
    _check_trmm(A, B, C, side, uplo, trans, diag, m, n, lda, ldb, ldc, True)
    ar, ai = _complex_scalar(alpha)
    if m == 0 or n == 0:
        return
    if ar == 0.0 and ai == 0.0:
        C[:m, :n].zero_()
        return
    B_work = B.clone() if B.data_ptr() == C.data_ptr() else B
    if (
        side == CUBLAS_SIDE_LEFT
        and uplo == CUBLAS_FILL_MODE_UPPER
        and trans == CUBLAS_OP_T
        and diag == CUBLAS_DIAG_UNIT
        and m == 512
        and n == 256
    ):
        A_real = torch.view_as_real(A)
        B_real = torch.view_as_real(B_work)
        C_real = torch.view_as_real(C)
        with torch_device_fn.device(A.device):
            _ctrmm_512_left_upper_t_unit_split4m_kernel[
                (triton.cdiv(m, 32) * triton.cdiv(n, 16),)
            ](
                A_real,
                B_real,
                C_real,
                ar,
                ai,
                lda,
                ldb,
                ldc,
                BLOCK_M=32,
                BLOCK_N=16,
                BLOCK_K=16,
                GROUP_M=4,
                num_warps=2,
                num_stages=1,
            )
        return
    if (
        side == CUBLAS_SIDE_LEFT
        and uplo == CUBLAS_FILL_MODE_UPPER
        and trans == CUBLAS_OP_T
        and diag == CUBLAS_DIAG_UNIT
        and m <= 512
        and n <= 256
        and m % 32 == 0
        and n % 16 == 0
    ):
        A_real = torch.view_as_real(A)
        B_real = torch.view_as_real(B_work)
        C_real = torch.view_as_real(C)
        with torch_device_fn.device(A.device):
            _ctrmm_left_upper_t_unit_split4m_interval_kernel[
                (triton.cdiv(m, 32) * triton.cdiv(n, 16),)
            ](
                A_real,
                B_real,
                C_real,
                ar,
                ai,
                m,
                n,
                lda,
                ldb,
                ldc,
                BLOCK_M=32,
                BLOCK_N=16,
                BLOCK_K=16,
                GROUP_M=4,
                num_warps=2,
                num_stages=1,
            )
        return
    if m >= _C3M_MIN and n >= 1024:
        if trans == CUBLAS_OP_N:
            _ctrmm_3m_n(side, uplo, diag, m, n, ar, ai, A, lda, B_work, ldb, C, ldc)
            return
        if side == CUBLAS_SIDE_LEFT:
            _ctrmm_3m_trans(
                side, uplo, trans, diag, m, n, ar, ai, A, lda, B_work, ldb, C, ldc
            )
            return
        _ctrmm_3m_right_trans(
            uplo, trans, diag, m, n, ar, ai, A, lda, B_work, ldb, C, ldc
        )
        return
    A_real = torch.view_as_real(A)
    B_real = torch.view_as_real(B_work)
    C_real = torch.view_as_real(C)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    conj = 1 if trans == CUBLAS_OP_C else 0
    trans_flag = CUBLAS_OP_T if trans == CUBLAS_OP_C else trans
    with torch_device_fn.device(A.device):
        _ctrmm_kernel[grid](
            A_real,
            B_real,
            C_real,
            ar,
            ai,
            m,
            n,
            lda,
            ldb,
            ldc,
            _mode_key(side, uplo, trans, diag),
            SIDE=side,
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
            CONJ=conj,
        )


@triton.jit
def _ztrmm_pack_a_kernel(
    a_ptr,
    ar_ptr,
    ai_ptr,
    k: tl.constexpr,
    lda,
    out_ld,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(k, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid - pid_m * grid_n
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < k
    col_mask = cols < k
    if UPLO == 1:
        tri = cols[None, :] >= rows[:, None]
    else:
        tri = cols[None, :] <= rows[:, None]
    mask = row_mask[:, None] & col_mask[None, :]
    diag_mask = rows[:, None] == cols[None, :]
    load_mask = mask & tri
    base = (rows[:, None] * lda + cols[None, :]) * 2
    vr = tl.load(a_ptr + base, mask=load_mask, other=0.0)
    vi = tl.load(a_ptr + base + 1, mask=load_mask, other=0.0)
    if UNIT:
        vr = tl.where(mask & diag_mask, 1.0, vr)
        vi = tl.where(mask & diag_mask, 0.0, vi)
    out = rows[:, None] * out_ld + cols[None, :]
    tl.store(ar_ptr + out, vr, mask=load_mask)
    tl.store(ai_ptr + out, vi, mask=load_mask)


@triton.jit
def _ztrmm_pack_a_t_kernel(
    a_ptr,
    ar_ptr,
    ai_ptr,
    k: tl.constexpr,
    lda,
    out_ld,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(k, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid - pid_m * grid_n
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < k
    col_mask = cols < k
    if UPLO == 1:
        tri = cols[None, :] >= rows[:, None]
    else:
        tri = cols[None, :] <= rows[:, None]
    mask = row_mask[:, None] & col_mask[None, :]
    diag_mask = rows[:, None] == cols[None, :]
    load_mask = mask & tri
    base = (rows[:, None] * lda + cols[None, :]) * 2
    vr = tl.load(a_ptr + base, mask=load_mask, other=0.0)
    vi = tl.load(a_ptr + base + 1, mask=load_mask, other=0.0)
    if CONJ:
        vi = -vi
    if UNIT:
        vr = tl.where(mask & diag_mask, 1.0, vr)
        vi = tl.where(mask & diag_mask, 0.0, vi)
    out = cols[None, :] * out_ld + rows[:, None]
    tl.store(ar_ptr + out, vr, mask=load_mask)
    tl.store(ai_ptr + out, vi, mask=load_mask)


@triton.jit
def _ztrmm_pack_b_kernel(
    b_ptr,
    br_ptr,
    bi_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    ldb,
    out_ld,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(n, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid - pid_m * grid_n
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < m) & (cols[None, :] < n)
    c2 = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
    rmask = (rows[:, None] < m) & (c2[None, :] < 2 * n)
    full = tl.load(
        b_ptr + rows[:, None] * (ldb * 2) + c2[None, :], mask=rmask, other=0.0
    )
    vr, vi = tl.split(tl.reshape(full, (BLOCK_M, BLOCK_N, 2)))
    dst = rows[:, None] * out_ld + cols[None, :]
    tl.store(br_ptr + dst, vr, mask=mask)
    tl.store(bi_ptr + dst, vi, mask=mask)


@triton.jit
def _ztrmm_pack_ab_kernel(
    a_ptr,
    b_ptr,
    ar_ptr,
    ai_ptr,
    br_ptr,
    bi_ptr,
    k: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    lda,
    ldb,
    out_a_ld,
    out_b_ld,
    NUM_A_BLOCKS: tl.constexpr,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    TRANS_A: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < NUM_A_BLOCKS:
        grid_n = tl.cdiv(k, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid - pid_m * grid_n
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < k
        col_mask = cols < k
        if UPLO == 1:
            tri = cols[None, :] >= rows[:, None]
        else:
            tri = cols[None, :] <= rows[:, None]
        mask = row_mask[:, None] & col_mask[None, :]
        diag_mask = rows[:, None] == cols[None, :]
        load_mask = mask & tri
        base = (rows[:, None] * lda + cols[None, :]) * 2
        vr = tl.load(a_ptr + base, mask=load_mask, other=0.0)
        vi = tl.load(a_ptr + base + 1, mask=load_mask, other=0.0)
        if CONJ:
            vi = -vi
        if UNIT:
            vr = tl.where(mask & diag_mask, 1.0, vr)
            vi = tl.where(mask & diag_mask, 0.0, vi)
        if TRANS_A:
            out = cols[None, :] * out_a_ld + rows[:, None]
            tl.store(ar_ptr + out, vr, mask=load_mask)
            tl.store(ai_ptr + out, vi, mask=load_mask)
        else:
            out = rows[:, None] * out_a_ld + cols[None, :]
            tl.store(ar_ptr + out, vr, mask=load_mask)
            tl.store(ai_ptr + out, vi, mask=load_mask)
    else:
        q = pid - NUM_A_BLOCKS
        grid_n = tl.cdiv(n, BLOCK_N)
        pid_m = q // grid_n
        pid_n = q - pid_m * grid_n
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < m) & (cols[None, :] < n)
        c2 = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
        rmask = (rows[:, None] < m) & (c2[None, :] < 2 * n)
        full = tl.load(
            b_ptr + rows[:, None] * (ldb * 2) + c2[None, :], mask=rmask, other=0.0
        )
        vr, vi = tl.split(tl.reshape(full, (BLOCK_M, BLOCK_N, 2)))
        dst = rows[:, None] * out_b_ld + cols[None, :]
        tl.store(br_ptr + dst, vr, mask=mask)
        tl.store(bi_ptr + dst, vi, mask=mask)


@libentry()
@libtuner(configs=runtime.get_tuned_config("ztrmm_3m"), key=_TRMM_KEY)
@triton.jit
def _ztrmm_3m_left_n_kernel(
    ar_p,
    ai_p,
    br_p,
    bi_p,
    c_ptr,
    alpha_r: tl.float64,
    alpha_i: tl.float64,
    m,
    n,
    lda,
    ldb,
    ldc,
    mode_key,
    UPLO: tl.constexpr,
    EVEN: tl.constexpr,
    JOIN_STORE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gsz = tl.minimum(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gsz)
    pid_n = (pid % width) // gsz
    if UPLO == 0:
        pid_m = grid_m - 1 - pid_m
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    ro = rows < m
    co = cols < n
    full_tile = EVEN or (((pid_m + 1) * BLOCK_M <= m) & ((pid_n + 1) * BLOCK_N <= n))
    p1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    p2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    p3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    diag_start = pid_m * BLOCK_M
    if UPLO == 0:
        ar_b = ar_p + rows[:, None] * lda + offs_k[None, :]
        ai_b = ai_p + rows[:, None] * lda + offs_k[None, :]
        br_b = br_p + offs_k[:, None] * ldb + cols[None, :]
        bi_b = bi_p + offs_k[:, None] * ldb + cols[None, :]
        if full_tile:
            for _k in tl.range(0, diag_start, BLOCK_K):
                a_r = tl.load(ar_b)
                a_i = tl.load(ai_b)
                b_r = tl.load(br_b)
                b_i = tl.load(bi_b)
                p1 = tl.dot(a_r, b_r, p1, allow_tf32=False, out_dtype=tl.float64)
                p2 = tl.dot(a_i, b_i, p2, allow_tf32=False, out_dtype=tl.float64)
                p3 = tl.dot(
                    a_r + a_i, b_r + b_i, p3, allow_tf32=False, out_dtype=tl.float64
                )
                ar_b += BLOCK_K
                ai_b += BLOCK_K
                br_b += BLOCK_K * ldb
                bi_b += BLOCK_K * ldb
        else:
            for _k in tl.range(0, diag_start, BLOCK_K):
                p1 = tl.dot(
                    tl.load(ar_b, mask=ro[:, None], other=0.0),
                    tl.load(br_b, mask=co[None, :], other=0.0),
                    p1,
                    allow_tf32=False,
                    out_dtype=tl.float64,
                )
                p2 = tl.dot(
                    tl.load(ai_b, mask=ro[:, None], other=0.0),
                    tl.load(bi_b, mask=co[None, :], other=0.0),
                    p2,
                    allow_tf32=False,
                    out_dtype=tl.float64,
                )
                p3 = tl.dot(
                    tl.load(ar_b, mask=ro[:, None], other=0.0)
                    + tl.load(ai_b, mask=ro[:, None], other=0.0),
                    tl.load(br_b, mask=co[None, :], other=0.0)
                    + tl.load(bi_b, mask=co[None, :], other=0.0),
                    p3,
                    allow_tf32=False,
                    out_dtype=tl.float64,
                )
                ar_b += BLOCK_K
                ai_b += BLOCK_K
                br_b += BLOCK_K * ldb
                bi_b += BLOCK_K * ldb
        for i in tl.static_range(0, BLOCK_M // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            ko = kidx < m
            tri = kidx[None, :] <= rows[:, None]
            am = ro[:, None] & ko[None, :] & tri
            bm = ko[:, None] & co[None, :]
            arr = rows[:, None] * lda + kidx[None, :]
            brr = kidx[:, None] * ldb + cols[None, :]
            a_r = tl.load(ar_p + arr, mask=am, other=0.0)
            a_i = tl.load(ai_p + arr, mask=am, other=0.0)
            b_r = tl.load(br_p + brr, mask=bm, other=0.0)
            b_i = tl.load(bi_p + brr, mask=bm, other=0.0)
            p1 = tl.dot(a_r, b_r, p1, allow_tf32=False, out_dtype=tl.float64)
            p2 = tl.dot(a_i, b_i, p2, allow_tf32=False, out_dtype=tl.float64)
            p3 = tl.dot(
                a_r + a_i, b_r + b_i, p3, allow_tf32=False, out_dtype=tl.float64
            )
    else:
        for i in tl.static_range(0, BLOCK_M // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            ko = kidx < m
            tri = kidx[None, :] >= rows[:, None]
            am = ro[:, None] & ko[None, :] & tri
            bm = ko[:, None] & co[None, :]
            arr = rows[:, None] * lda + kidx[None, :]
            brr = kidx[:, None] * ldb + cols[None, :]
            a_r = tl.load(ar_p + arr, mask=am, other=0.0)
            a_i = tl.load(ai_p + arr, mask=am, other=0.0)
            b_r = tl.load(br_p + brr, mask=bm, other=0.0)
            b_i = tl.load(bi_p + brr, mask=bm, other=0.0)
            p1 = tl.dot(a_r, b_r, p1, allow_tf32=False, out_dtype=tl.float64)
            p2 = tl.dot(a_i, b_i, p2, allow_tf32=False, out_dtype=tl.float64)
            p3 = tl.dot(
                a_r + a_i, b_r + b_i, p3, allow_tf32=False, out_dtype=tl.float64
            )
        ks = diag_start + BLOCK_M
        ar_b = ar_p + rows[:, None] * lda + (ks + offs_k)[None, :]
        ai_b = ai_p + rows[:, None] * lda + (ks + offs_k)[None, :]
        br_b = br_p + (ks + offs_k)[:, None] * ldb + cols[None, :]
        bi_b = bi_p + (ks + offs_k)[:, None] * ldb + cols[None, :]
        if full_tile:
            for k0 in tl.range(ks, m, BLOCK_K):
                kidx = k0 + offs_k
                ko = kidx < m
                a_r = tl.load(ar_b, mask=ko[None, :], other=0.0)
                a_i = tl.load(ai_b, mask=ko[None, :], other=0.0)
                b_r = tl.load(br_b, mask=ko[:, None], other=0.0)
                b_i = tl.load(bi_b, mask=ko[:, None], other=0.0)
                p1 = tl.dot(a_r, b_r, p1, allow_tf32=False, out_dtype=tl.float64)
                p2 = tl.dot(a_i, b_i, p2, allow_tf32=False, out_dtype=tl.float64)
                p3 = tl.dot(
                    a_r + a_i, b_r + b_i, p3, allow_tf32=False, out_dtype=tl.float64
                )
                ar_b += BLOCK_K
                ai_b += BLOCK_K
                br_b += BLOCK_K * ldb
                bi_b += BLOCK_K * ldb
        else:
            for k0 in tl.range(ks, m, BLOCK_K):
                kidx = k0 + offs_k
                ko = kidx < m
                am = ro[:, None] & ko[None, :]
                bm = ko[:, None] & co[None, :]
                p1 = tl.dot(
                    tl.load(ar_b, mask=am, other=0.0),
                    tl.load(br_b, mask=bm, other=0.0),
                    p1,
                    allow_tf32=False,
                    out_dtype=tl.float64,
                )
                p2 = tl.dot(
                    tl.load(ai_b, mask=am, other=0.0),
                    tl.load(bi_b, mask=bm, other=0.0),
                    p2,
                    allow_tf32=False,
                    out_dtype=tl.float64,
                )
                p3 = tl.dot(
                    tl.load(ar_b, mask=am, other=0.0)
                    + tl.load(ai_b, mask=am, other=0.0),
                    tl.load(br_b, mask=bm, other=0.0)
                    + tl.load(bi_b, mask=bm, other=0.0),
                    p3,
                    allow_tf32=False,
                    out_dtype=tl.float64,
                )
                ar_b += BLOCK_K
                ai_b += BLOCK_K
                br_b += BLOCK_K * ldb
                bi_b += BLOCK_K * ldb
    cr = p1 - p2
    ci = p3 - p1 - p2
    out_r = alpha_r * cr - alpha_i * ci
    out_i = alpha_r * ci + alpha_i * cr
    cmask = ro[:, None] & co[None, :]
    cb = (rows[:, None] * ldc + cols[None, :]) * 2
    if JOIN_STORE:
        two = tl.arange(0, 2)
        tl.store(
            c_ptr + cb[:, :, None] + two[None, None, :],
            tl.join(out_r, out_i),
            mask=cmask[:, :, None],
        )
    else:
        tl.store(c_ptr + cb, out_r, mask=cmask)
        tl.store(c_ptr + cb + 1, out_i, mask=cmask)


@triton.jit
def _ztrmm_3m_left_n_fuseA_kernel(
    a_il,
    br_p,
    bi_p,
    c_ptr,
    alpha_r: tl.float64,
    alpha_i: tl.float64,
    m,
    n,
    lda2,
    ldb,
    ldc,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gsz = tl.minimum(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gsz)
    pid_n = (pid % width) // gsz
    if UPLO == 0:
        pid_m = grid_m - 1 - pid_m
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k2 = tl.arange(0, 2 * BLOCK_K)
    p1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    p2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    p3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    diag_start = pid_m * BLOCK_M
    a_row = a_il + rows[:, None] * lda2
    if UPLO == 0:
        a_b = a_row + k2[None, :]
        br_b = br_p + offs_k[:, None] * ldb + cols[None, :]
        bi_b = bi_p + offs_k[:, None] * ldb + cols[None, :]
        for _k in tl.range(0, diag_start, BLOCK_K):
            a_r, a_i = tl.split(tl.reshape(tl.load(a_b), (BLOCK_M, BLOCK_K, 2)))
            b_r = tl.load(br_b)
            b_i = tl.load(bi_b)
            p1 = tl.dot(a_r, b_r, p1, allow_tf32=False, out_dtype=tl.float64)
            p2 = tl.dot(a_i, b_i, p2, allow_tf32=False, out_dtype=tl.float64)
            p3 = tl.dot(
                a_r + a_i, b_r + b_i, p3, allow_tf32=False, out_dtype=tl.float64
            )
            a_b += BLOCK_K * 2
            br_b += BLOCK_K * ldb
            bi_b += BLOCK_K * ldb
        for i in tl.static_range(0, BLOCK_M // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            a_r, a_i = tl.split(tl.reshape(tl.load(a_b), (BLOCK_M, BLOCK_K, 2)))
            tri = kidx[None, :] <= rows[:, None]
            a_r = tl.where(tri, a_r, 0.0)
            a_i = tl.where(tri, a_i, 0.0)
            if UNIT:
                diag = kidx[None, :] == rows[:, None]
                a_r = tl.where(diag, 1.0, a_r)
                a_i = tl.where(diag, 0.0, a_i)
            brr = kidx[:, None] * ldb + cols[None, :]
            b_r = tl.load(br_p + brr)
            b_i = tl.load(bi_p + brr)
            p1 = tl.dot(a_r, b_r, p1, allow_tf32=False, out_dtype=tl.float64)
            p2 = tl.dot(a_i, b_i, p2, allow_tf32=False, out_dtype=tl.float64)
            p3 = tl.dot(
                a_r + a_i, b_r + b_i, p3, allow_tf32=False, out_dtype=tl.float64
            )
            a_b += BLOCK_K * 2
    else:
        a_b = a_row + (diag_start * 2 + k2)[None, :]
        for i in tl.static_range(0, BLOCK_M // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            a_r, a_i = tl.split(tl.reshape(tl.load(a_b), (BLOCK_M, BLOCK_K, 2)))
            tri = kidx[None, :] >= rows[:, None]
            a_r = tl.where(tri, a_r, 0.0)
            a_i = tl.where(tri, a_i, 0.0)
            if UNIT:
                diag = kidx[None, :] == rows[:, None]
                a_r = tl.where(diag, 1.0, a_r)
                a_i = tl.where(diag, 0.0, a_i)
            brr = kidx[:, None] * ldb + cols[None, :]
            b_r = tl.load(br_p + brr)
            b_i = tl.load(bi_p + brr)
            p1 = tl.dot(a_r, b_r, p1, allow_tf32=False, out_dtype=tl.float64)
            p2 = tl.dot(a_i, b_i, p2, allow_tf32=False, out_dtype=tl.float64)
            p3 = tl.dot(
                a_r + a_i, b_r + b_i, p3, allow_tf32=False, out_dtype=tl.float64
            )
            a_b += BLOCK_K * 2
        ks = diag_start + BLOCK_M
        br_b = br_p + (ks + offs_k)[:, None] * ldb + cols[None, :]
        bi_b = bi_p + (ks + offs_k)[:, None] * ldb + cols[None, :]
        for _k in tl.range(ks, m, BLOCK_K):
            a_r, a_i = tl.split(tl.reshape(tl.load(a_b), (BLOCK_M, BLOCK_K, 2)))
            b_r = tl.load(br_b)
            b_i = tl.load(bi_b)
            p1 = tl.dot(a_r, b_r, p1, allow_tf32=False, out_dtype=tl.float64)
            p2 = tl.dot(a_i, b_i, p2, allow_tf32=False, out_dtype=tl.float64)
            p3 = tl.dot(
                a_r + a_i, b_r + b_i, p3, allow_tf32=False, out_dtype=tl.float64
            )
            a_b += BLOCK_K * 2
            br_b += BLOCK_K * ldb
            bi_b += BLOCK_K * ldb
    cr = p1 - p2
    ci = p3 - p1 - p2
    out_r = alpha_r * cr - alpha_i * ci
    out_i = alpha_r * ci + alpha_i * cr
    cmask = (rows[:, None] < m) & (cols[None, :] < n)
    cb = (rows[:, None] * ldc + cols[None, :]) * 2
    two = tl.arange(0, 2)
    tl.store(
        c_ptr + cb[:, :, None] + two[None, None, :],
        tl.join(out_r, out_i),
        mask=cmask[:, :, None],
    )


@libentry()
@libtuner(configs=runtime.get_tuned_config("ztrmm_3m"), key=_TRMM_KEY)
@triton.jit
def _ztrmm_3m_right_n_kernel(
    ar_p,
    ai_p,
    br_p,
    bi_p,
    c_ptr,
    alpha_r: tl.float64,
    alpha_i: tl.float64,
    m,
    n,
    lda,
    ldb,
    ldc,
    mode_key,
    UPLO: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gsz = tl.minimum(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gsz)
    pid_n = (pid % width) // gsz
    if UPLO == 1:
        pid_n = grid_n - 1 - pid_n
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    ro = rows < m
    co = cols < n
    full_tile = EVEN or (((pid_m + 1) * BLOCK_M <= m) & ((pid_n + 1) * BLOCK_N <= n))
    p1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    p2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    p3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    diag_start = pid_n * BLOCK_N
    if UPLO == 1:
        br_b = br_p + rows[:, None] * ldb + offs_k[None, :]
        bi_b = bi_p + rows[:, None] * ldb + offs_k[None, :]
        ar_b = ar_p + offs_k[:, None] * lda + cols[None, :]
        ai_b = ai_p + offs_k[:, None] * lda + cols[None, :]
        if full_tile:
            for _k in tl.range(0, diag_start, BLOCK_K):
                b_r = tl.load(br_b)
                b_i = tl.load(bi_b)
                a_r = tl.load(ar_b)
                a_i = tl.load(ai_b)
                p1 = tl.dot(b_r, a_r, p1, allow_tf32=False, out_dtype=tl.float64)
                p2 = tl.dot(b_i, a_i, p2, allow_tf32=False, out_dtype=tl.float64)
                p3 = tl.dot(
                    b_r + b_i, a_r + a_i, p3, allow_tf32=False, out_dtype=tl.float64
                )
                br_b += BLOCK_K
                bi_b += BLOCK_K
                ar_b += BLOCK_K * lda
                ai_b += BLOCK_K * lda
        else:
            for _k in tl.range(0, diag_start, BLOCK_K):
                p1 = tl.dot(
                    tl.load(br_b, mask=ro[:, None], other=0.0),
                    tl.load(ar_b, mask=co[None, :], other=0.0),
                    p1,
                    allow_tf32=False,
                    out_dtype=tl.float64,
                )
                p2 = tl.dot(
                    tl.load(bi_b, mask=ro[:, None], other=0.0),
                    tl.load(ai_b, mask=co[None, :], other=0.0),
                    p2,
                    allow_tf32=False,
                    out_dtype=tl.float64,
                )
                p3 = tl.dot(
                    tl.load(br_b, mask=ro[:, None], other=0.0)
                    + tl.load(bi_b, mask=ro[:, None], other=0.0),
                    tl.load(ar_b, mask=co[None, :], other=0.0)
                    + tl.load(ai_b, mask=co[None, :], other=0.0),
                    p3,
                    allow_tf32=False,
                    out_dtype=tl.float64,
                )
                br_b += BLOCK_K
                bi_b += BLOCK_K
                ar_b += BLOCK_K * lda
                ai_b += BLOCK_K * lda
        for i in tl.static_range(0, BLOCK_N // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            ko = kidx < n
            tri = kidx[:, None] <= cols[None, :]
            bm = ro[:, None] & ko[None, :]
            am = ko[:, None] & co[None, :] & tri
            brr = rows[:, None] * ldb + kidx[None, :]
            arr = kidx[:, None] * lda + cols[None, :]
            b_r = tl.load(br_p + brr, mask=bm, other=0.0)
            b_i = tl.load(bi_p + brr, mask=bm, other=0.0)
            a_r = tl.load(ar_p + arr, mask=am, other=0.0)
            a_i = tl.load(ai_p + arr, mask=am, other=0.0)
            p1 = tl.dot(b_r, a_r, p1, allow_tf32=False, out_dtype=tl.float64)
            p2 = tl.dot(b_i, a_i, p2, allow_tf32=False, out_dtype=tl.float64)
            p3 = tl.dot(
                b_r + b_i, a_r + a_i, p3, allow_tf32=False, out_dtype=tl.float64
            )
    else:
        for i in tl.static_range(0, BLOCK_N // BLOCK_K):
            kidx = diag_start + i * BLOCK_K + offs_k
            ko = kidx < n
            tri = kidx[:, None] >= cols[None, :]
            bm = ro[:, None] & ko[None, :]
            am = ko[:, None] & co[None, :] & tri
            brr = rows[:, None] * ldb + kidx[None, :]
            arr = kidx[:, None] * lda + cols[None, :]
            b_r = tl.load(br_p + brr, mask=bm, other=0.0)
            b_i = tl.load(bi_p + brr, mask=bm, other=0.0)
            a_r = tl.load(ar_p + arr, mask=am, other=0.0)
            a_i = tl.load(ai_p + arr, mask=am, other=0.0)
            p1 = tl.dot(b_r, a_r, p1, allow_tf32=False, out_dtype=tl.float64)
            p2 = tl.dot(b_i, a_i, p2, allow_tf32=False, out_dtype=tl.float64)
            p3 = tl.dot(
                b_r + b_i, a_r + a_i, p3, allow_tf32=False, out_dtype=tl.float64
            )
        ks = diag_start + BLOCK_N
        br_b = br_p + rows[:, None] * ldb + (ks + offs_k)[None, :]
        bi_b = bi_p + rows[:, None] * ldb + (ks + offs_k)[None, :]
        ar_b = ar_p + (ks + offs_k)[:, None] * lda + cols[None, :]
        ai_b = ai_p + (ks + offs_k)[:, None] * lda + cols[None, :]
        if full_tile:
            for k0 in tl.range(ks, n, BLOCK_K):
                kidx = k0 + offs_k
                ko = kidx < n
                b_r = tl.load(br_b, mask=ko[None, :], other=0.0)
                b_i = tl.load(bi_b, mask=ko[None, :], other=0.0)
                a_r = tl.load(ar_b, mask=ko[:, None], other=0.0)
                a_i = tl.load(ai_b, mask=ko[:, None], other=0.0)
                p1 = tl.dot(b_r, a_r, p1, allow_tf32=False, out_dtype=tl.float64)
                p2 = tl.dot(b_i, a_i, p2, allow_tf32=False, out_dtype=tl.float64)
                p3 = tl.dot(
                    b_r + b_i, a_r + a_i, p3, allow_tf32=False, out_dtype=tl.float64
                )
                br_b += BLOCK_K
                bi_b += BLOCK_K
                ar_b += BLOCK_K * lda
                ai_b += BLOCK_K * lda
        else:
            for k0 in tl.range(ks, n, BLOCK_K):
                kidx = k0 + offs_k
                ko = kidx < n
                bm = ro[:, None] & ko[None, :]
                am = ko[:, None] & co[None, :]
                p1 = tl.dot(
                    tl.load(br_b, mask=bm, other=0.0),
                    tl.load(ar_b, mask=am, other=0.0),
                    p1,
                    allow_tf32=False,
                    out_dtype=tl.float64,
                )
                p2 = tl.dot(
                    tl.load(bi_b, mask=bm, other=0.0),
                    tl.load(ai_b, mask=am, other=0.0),
                    p2,
                    allow_tf32=False,
                    out_dtype=tl.float64,
                )
                p3 = tl.dot(
                    tl.load(br_b, mask=bm, other=0.0)
                    + tl.load(bi_b, mask=bm, other=0.0),
                    tl.load(ar_b, mask=am, other=0.0)
                    + tl.load(ai_b, mask=am, other=0.0),
                    p3,
                    allow_tf32=False,
                    out_dtype=tl.float64,
                )
                br_b += BLOCK_K
                bi_b += BLOCK_K
                ar_b += BLOCK_K * lda
                ai_b += BLOCK_K * lda
    cr = p1 - p2
    ci = p3 - p1 - p2
    out_r = alpha_r * cr - alpha_i * ci
    out_i = alpha_r * ci + alpha_i * cr
    cmask = ro[:, None] & co[None, :]
    cb = (rows[:, None] * ldc + cols[None, :]) * 2
    two = tl.arange(0, 2)
    tl.store(
        c_ptr + cb[:, :, None] + two[None, None, :],
        tl.join(out_r, out_i),
        mask=cmask[:, :, None],
    )


def _ztrmm_planes(k, ka, m, na, device):
    sa = k * ka
    sb = m * na
    buf = torch.empty(2 * sa + 2 * sb, dtype=torch.float64, device=device)
    a_r = buf[0:sa].view(k, ka)
    a_i = buf[sa : 2 * sa].view(k, ka)
    b_r = buf[2 * sa : 2 * sa + sb].view(m, na)
    b_i = buf[2 * sa + sb :].view(m, na)
    return a_r, a_i, b_r, b_i


def _ztrmm_3m_trans(side, uplo, trans, diag, m, n, ar, ai, A, lda, B, ldb, C, ldc):
    k = m if side == CUBLAS_SIDE_LEFT else n
    ka = (k + 31) // 32 * 32
    na = (n + 31) // 32 * 32
    A_real = torch.view_as_real(A)
    B_real = torch.view_as_real(B)
    C_real = torch.view_as_real(C)
    a_r, a_i, b_r, b_i = _ztrmm_planes(k, ka, m, na, A.device)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    conj = 1 if trans == CUBLAS_OP_C else 0
    uplo_t = (
        CUBLAS_FILL_MODE_LOWER
        if uplo == CUBLAS_FILL_MODE_UPPER
        else CUBLAS_FILL_MODE_UPPER
    )
    even = 1 if (m % 256 == 0 and n % 256 == 0) else 0
    pack_a_grid = (triton.cdiv(k, 16) * triton.cdiv(k, 16),)
    pack_b_grid = (triton.cdiv(m, 16) * triton.cdiv(n, 64),)
    pack_ab_a_blocks = triton.cdiv(k, 16) * triton.cdiv(k, 64)
    pack_ab_grid = (pack_ab_a_blocks + pack_b_grid[0],)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    mk = _mode_key(side, uplo, trans, diag)
    with torch_device_fn.device(A.device):
        if max(m, n) <= _Z3M_PACK_AB_MAX:
            _ztrmm_pack_ab_kernel[pack_ab_grid](
                A_real,
                B_real,
                a_r,
                a_i,
                b_r,
                b_i,
                k,
                m,
                n,
                lda,
                ldb,
                ka,
                na,
                pack_ab_a_blocks,
                UPLO=uplo,
                UNIT=unit,
                TRANS_A=1,
                CONJ=conj,
                BLOCK_M=16,
                BLOCK_N=64,
                num_warps=4,
                num_stages=3,
            )
        else:
            _ztrmm_pack_a_t_kernel[pack_a_grid](
                A_real,
                a_r,
                a_i,
                k,
                lda,
                ka,
                UPLO=uplo,
                UNIT=unit,
                CONJ=conj,
                BLOCK_M=16,
                BLOCK_N=16,
            )
            _ztrmm_pack_b_kernel[pack_b_grid](
                B_real, b_r, b_i, m, n, ldb, na, BLOCK_M=16, BLOCK_N=64
            )
        join_store = 1 if (even or max(m, n) <= 4096) else 0
        _ztrmm_3m_left_n_kernel[grid](
            a_r,
            a_i,
            b_r,
            b_i,
            C_real,
            ar,
            ai,
            m,
            n,
            ka,
            na,
            ldc,
            mk,
            UPLO=uplo_t,
            EVEN=even,
            JOIN_STORE=join_store,
        )


def _ztrmm_3m_n(side, uplo, diag, m, n, ar, ai, A, lda, B, ldb, C, ldc):
    k = m if side == CUBLAS_SIDE_LEFT else n
    ka = (k + 31) // 32 * 32
    na = (n + 31) // 32 * 32
    A_real = torch.view_as_real(A)
    B_real = torch.view_as_real(B)
    C_real = torch.view_as_real(C)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    even = 1 if (m % 256 == 0 and n % 256 == 0) else 0
    pack_b_grid = (triton.cdiv(m, 16) * triton.cdiv(n, 64),)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    mk = _mode_key(side, uplo, CUBLAS_OP_N, diag)
    if side == CUBLAS_SIDE_LEFT and even and max(m, n) <= 2048:
        BM, BN = 16, 128
        sb = m * na
        buf = torch.empty(2 * sb, dtype=torch.float64, device=A.device)
        b_r = buf[0:sb].view(m, na)
        b_i = buf[sb:].view(m, na)
        fuse_grid = (triton.cdiv(m, BM) * triton.cdiv(n, BN),)
        with torch_device_fn.device(A.device):
            _ztrmm_pack_b_kernel[pack_b_grid](
                B_real, b_r, b_i, m, n, ldb, na, BLOCK_M=16, BLOCK_N=64
            )
            _ztrmm_3m_left_n_fuseA_kernel[fuse_grid](
                A_real,
                b_r,
                b_i,
                C_real,
                ar,
                ai,
                m,
                n,
                2 * lda,
                na,
                ldc,
                UPLO=uplo,
                UNIT=unit,
                BLOCK_M=BM,
                BLOCK_N=BN,
                BLOCK_K=16,
                GROUP_M=4,
                num_warps=4,
                num_stages=3,
            )
        return
    a_r, a_i, b_r, b_i = _ztrmm_planes(k, ka, m, na, A.device)
    pack_a_grid = (triton.cdiv(k, 16) * triton.cdiv(k, 16),)
    pack_ab_a_blocks = triton.cdiv(k, 16) * triton.cdiv(k, 64)
    pack_ab_grid = (pack_ab_a_blocks + pack_b_grid[0],)
    with torch_device_fn.device(A.device):
        if side == CUBLAS_SIDE_RIGHT and max(m, n) <= _Z3M_PACK_AB_MAX:
            _ztrmm_pack_ab_kernel[pack_ab_grid](
                A_real,
                B_real,
                a_r,
                a_i,
                b_r,
                b_i,
                k,
                m,
                n,
                lda,
                ldb,
                ka,
                na,
                pack_ab_a_blocks,
                UPLO=uplo,
                UNIT=unit,
                TRANS_A=0,
                CONJ=0,
                BLOCK_M=16,
                BLOCK_N=64,
                num_warps=4,
                num_stages=3,
            )
        else:
            _ztrmm_pack_a_kernel[pack_a_grid](
                A_real,
                a_r,
                a_i,
                k,
                lda,
                ka,
                UPLO=uplo,
                UNIT=unit,
                BLOCK_M=16,
                BLOCK_N=16,
            )
            _ztrmm_pack_b_kernel[pack_b_grid](
                B_real, b_r, b_i, m, n, ldb, na, BLOCK_M=16, BLOCK_N=64
            )
        if side == CUBLAS_SIDE_LEFT:
            join_store = 1 if (even or max(m, n) <= 4096) else 0
            _ztrmm_3m_left_n_kernel[grid](
                a_r,
                a_i,
                b_r,
                b_i,
                C_real,
                ar,
                ai,
                m,
                n,
                ka,
                na,
                ldc,
                mk,
                UPLO=uplo,
                EVEN=even,
                JOIN_STORE=join_store,
            )
        else:
            _ztrmm_3m_right_n_kernel[grid](
                a_r,
                a_i,
                b_r,
                b_i,
                C_real,
                ar,
                ai,
                m,
                n,
                ka,
                na,
                ldc,
                mk,
                UPLO=uplo,
                EVEN=even,
            )


def _ztrmm_3m_right_trans(uplo, trans, diag, m, n, ar, ai, A, lda, B, ldb, C, ldc):
    k = n
    ka = (k + 31) // 32 * 32
    na = (n + 31) // 32 * 32
    A_real = torch.view_as_real(A)
    B_real = torch.view_as_real(B)
    C_real = torch.view_as_real(C)
    a_r, a_i, b_r, b_i = _ztrmm_planes(k, ka, m, na, A.device)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    conj = 1 if trans == CUBLAS_OP_C else 0
    uplo_t = (
        CUBLAS_FILL_MODE_LOWER
        if uplo == CUBLAS_FILL_MODE_UPPER
        else CUBLAS_FILL_MODE_UPPER
    )
    even = 1 if (m % 256 == 0 and n % 256 == 0) else 0
    pack_a_grid = (triton.cdiv(k, 16) * triton.cdiv(k, 16),)
    pack_b_grid = (triton.cdiv(m, 16) * triton.cdiv(n, 64),)
    pack_ab_a_blocks = triton.cdiv(k, 16) * triton.cdiv(k, 64)
    pack_ab_grid = (pack_ab_a_blocks + pack_b_grid[0],)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    mk = _mode_key(CUBLAS_SIDE_RIGHT, uplo, trans, diag)
    with torch_device_fn.device(A.device):
        if max(m, n) <= _Z3M_PACK_AB_MAX:
            _ztrmm_pack_ab_kernel[pack_ab_grid](
                A_real,
                B_real,
                a_r,
                a_i,
                b_r,
                b_i,
                k,
                m,
                n,
                lda,
                ldb,
                ka,
                na,
                pack_ab_a_blocks,
                UPLO=uplo,
                UNIT=unit,
                TRANS_A=1,
                CONJ=conj,
                BLOCK_M=16,
                BLOCK_N=64,
                num_warps=4,
                num_stages=3,
            )
        else:
            _ztrmm_pack_a_t_kernel[pack_a_grid](
                A_real,
                a_r,
                a_i,
                k,
                lda,
                ka,
                UPLO=uplo,
                UNIT=unit,
                CONJ=conj,
                BLOCK_M=16,
                BLOCK_N=16,
            )
            _ztrmm_pack_b_kernel[pack_b_grid](
                B_real, b_r, b_i, m, n, ldb, na, BLOCK_M=16, BLOCK_N=64
            )
        _ztrmm_3m_right_n_kernel[grid](
            a_r,
            a_i,
            b_r,
            b_i,
            C_real,
            ar,
            ai,
            m,
            n,
            ka,
            na,
            ldc,
            mk,
            UPLO=uplo_t,
            EVEN=even,
        )


@triton.jit
def _ztrmm_fused_left_n_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    ar_a,
    ai_a,
    m,
    n,
    ldar,
    ldbr,
    ldcr,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    gm = tl.cdiv(m, BLOCK_M)
    gn = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * gn
    gid = pid // width
    gsize = tl.minimum(gm - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gsize)
    pid_n = (pid % width) // gsize
    if UPLO == 0:
        pid_m = gm - 1 - pid_m
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < m
    cmask = cols < n
    offk = tl.arange(0, BLOCK_K)
    wa = tl.arange(0, 2 * BLOCK_K)
    wb = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
    wa_k = wa // 2
    cmask_w = (pid_n * BLOCK_N + tl.arange(0, 2 * BLOCK_N) // 2) < n
    p1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    p2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    p3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    diag_start = pid_m * BLOCK_M
    if UNIT:
        b0 = tl.load(
            b_ptr + rows[:, None] * ldbr + (cols * 2)[None, :],
            mask=rmask[:, None] & cmask[None, :],
            other=0.0,
        )
        b0i = tl.load(
            b_ptr + rows[:, None] * ldbr + (cols * 2 + 1)[None, :],
            mask=rmask[:, None] & cmask[None, :],
            other=0.0,
        )
    if UPLO == 0:
        bulk_lo = 0
        bulk_hi = diag_start
    else:
        bulk_lo = diag_start + BLOCK_M
        bulk_hi = m
    for k0 in tl.range(bulk_lo, bulk_hi, BLOCK_K):
        kcol = k0 + wa_k
        amask = rmask[:, None] & (kcol < m)[None, :]
        krow = k0 + offk
        bmask = (krow < m)[:, None] & cmask_w[None, :]
        a_blk = tl.load(
            a_ptr + rows[:, None] * ldar + (k0 * 2 + wa)[None, :],
            mask=amask,
            other=0.0,
            eviction_policy="evict_first",
        )
        b_blk = tl.load(
            b_ptr + krow[:, None] * ldbr + wb[None, :],
            mask=bmask,
            other=0.0,
            eviction_policy="evict_last",
        )
        ar, ai = tl.split(tl.reshape(a_blk, (BLOCK_M, BLOCK_K, 2)))
        br, bi = tl.split(tl.reshape(b_blk, (BLOCK_K, BLOCK_N, 2)))
        p1 = tl.dot(ar, br, p1, out_dtype=tl.float64, allow_tf32=False)
        p2 = tl.dot(ai, bi, p2, out_dtype=tl.float64, allow_tf32=False)
        p3 = tl.dot(ar + ai, br + bi, p3, out_dtype=tl.float64, allow_tf32=False)
    for i in tl.static_range(0, BLOCK_M // BLOCK_K):
        kbase = diag_start + i * BLOCK_K
        kidx = kbase + offk
        kcol = kbase + wa_k
        amask = rmask[:, None] & (kcol < m)[None, :]
        bmask = (kidx < m)[:, None] & cmask_w[None, :]
        a_blk = tl.load(
            a_ptr + rows[:, None] * ldar + (kbase * 2 + wa)[None, :],
            mask=amask,
            other=0.0,
            eviction_policy="evict_first",
        )
        b_blk = tl.load(
            b_ptr + kidx[:, None] * ldbr + wb[None, :],
            mask=bmask,
            other=0.0,
            eviction_policy="evict_last",
        )
        ar, ai = tl.split(tl.reshape(a_blk, (BLOCK_M, BLOCK_K, 2)))
        br, bi = tl.split(tl.reshape(b_blk, (BLOCK_K, BLOCK_N, 2)))
        if UPLO == 0:
            tri = (
                kidx[None, :] < rows[:, None]
                if UNIT
                else kidx[None, :] <= rows[:, None]
            )
        else:
            tri = (
                kidx[None, :] > rows[:, None]
                if UNIT
                else kidx[None, :] >= rows[:, None]
            )
        ar = tl.where(tri, ar, 0.0)
        ai = tl.where(tri, ai, 0.0)
        p1 = tl.dot(ar, br, p1, out_dtype=tl.float64, allow_tf32=False)
        p2 = tl.dot(ai, bi, p2, out_dtype=tl.float64, allow_tf32=False)
        p3 = tl.dot(ar + ai, br + bi, p3, out_dtype=tl.float64, allow_tf32=False)
    cr = p1 - p2
    ci = p3 - p1 - p2
    if UNIT:
        cr += b0
        ci += b0i
    outr = ar_a * cr - ai_a * ci
    outi = ar_a * ci + ai_a * cr
    cb = c_ptr + rows[:, None] * ldcr + (cols * 2)[None, :]
    c_mask = rmask[:, None] & cmask[None, :]
    two = tl.arange(0, 2)
    tl.store(
        cb[:, :, None] + two[None, None, :],
        tl.join(outr, outi),
        mask=c_mask[:, :, None],
    )


@triton.jit
def _ztrmm_fused_left_t_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    ar_a,
    ai_a,
    m,
    n,
    ldar,
    ldbr,
    ldcr,
    UPLO: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    gm = tl.cdiv(m, BLOCK_M)
    gn = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * gn
    gid = pid // width
    gsize = tl.minimum(gm - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gsize)
    pid_n = (pid % width) // gsize
    if UPLO == 1:
        pid_m = gm - 1 - pid_m
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < m
    cmask = cols < n
    offk = tl.arange(0, BLOCK_K)
    wbm = tl.arange(0, 2 * BLOCK_M)
    wb = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
    wbm_r = wbm // 2
    rmask_w = (pid_m * BLOCK_M + wbm_r) < m
    cmask_w = (pid_n * BLOCK_N + tl.arange(0, 2 * BLOCK_N) // 2) < n
    abase = pid_m * BLOCK_M * 2 + wbm
    p1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    p2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    p3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    diag_start = pid_m * BLOCK_M
    if UPLO == 1:
        bulk_lo = 0
        bulk_hi = diag_start
    else:
        bulk_lo = diag_start + BLOCK_M
        bulk_hi = m
    for k0 in tl.range(bulk_lo, bulk_hi, BLOCK_K):
        krow = k0 + offk
        kmask = krow < m
        a_blk = tl.load(
            a_ptr + krow[:, None] * ldar + abase[None, :],
            mask=kmask[:, None] & rmask_w[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        b_blk = tl.load(
            b_ptr + krow[:, None] * ldbr + wb[None, :],
            mask=kmask[:, None] & cmask_w[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        ar0, ai0 = tl.split(tl.reshape(a_blk, (BLOCK_K, BLOCK_M, 2)))
        br, bi = tl.split(tl.reshape(b_blk, (BLOCK_K, BLOCK_N, 2)))
        ar = tl.trans(ar0)
        ai = tl.trans(ai0)
        if CONJ:
            ai = -ai
        p1 = tl.dot(ar, br, p1, out_dtype=tl.float64, allow_tf32=False)
        p2 = tl.dot(ai, bi, p2, out_dtype=tl.float64, allow_tf32=False)
        p3 = tl.dot(ar + ai, br + bi, p3, out_dtype=tl.float64, allow_tf32=False)
    for i in tl.static_range(0, BLOCK_M // BLOCK_K):
        kbase = diag_start + i * BLOCK_K
        kidx = kbase + offk
        kmask = kidx < m
        a_blk = tl.load(
            a_ptr + kidx[:, None] * ldar + abase[None, :],
            mask=kmask[:, None] & rmask_w[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        b_blk = tl.load(
            b_ptr + kidx[:, None] * ldbr + wb[None, :],
            mask=kmask[:, None] & cmask_w[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        ar0, ai0 = tl.split(tl.reshape(a_blk, (BLOCK_K, BLOCK_M, 2)))
        br, bi = tl.split(tl.reshape(b_blk, (BLOCK_K, BLOCK_N, 2)))
        ar = tl.trans(ar0)
        ai = tl.trans(ai0)
        if CONJ:
            ai = -ai
        if UPLO == 1:
            tri = kidx[None, :] <= rows[:, None]
        else:
            tri = kidx[None, :] >= rows[:, None]
        ar = tl.where(tri, ar, 0.0)
        ai = tl.where(tri, ai, 0.0)
        if UNIT:
            diag = kidx[None, :] == rows[:, None]
            ar = tl.where(diag, 1.0, ar)
            ai = tl.where(diag, 0.0, ai)
        p1 = tl.dot(ar, br, p1, out_dtype=tl.float64, allow_tf32=False)
        p2 = tl.dot(ai, bi, p2, out_dtype=tl.float64, allow_tf32=False)
        p3 = tl.dot(ar + ai, br + bi, p3, out_dtype=tl.float64, allow_tf32=False)
    cr = p1 - p2
    ci = p3 - p1 - p2
    outr = ar_a * cr - ai_a * ci
    outi = ar_a * ci + ai_a * cr
    cb = c_ptr + rows[:, None] * ldcr + (cols * 2)[None, :]
    c_mask = rmask[:, None] & cmask[None, :]
    two = tl.arange(0, 2)
    tl.store(
        cb[:, :, None] + two[None, None, :],
        tl.join(outr, outi),
        mask=c_mask[:, :, None],
    )


@triton.jit
def _ztrmm_512_left_n_lower_nonunit_bulkbk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    ar_a: tl.float64,
    ai_a: tl.float64,
    ldar: tl.constexpr,
    ldbr: tl.constexpr,
    ldcr: tl.constexpr,
    BLOCK_BULK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    gm = 32
    gn = 8
    width = GROUP_M * gn
    gid = pid // width
    gsize = tl.minimum(gm - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gsize)
    pid_n = (pid % width) // gsize
    pid_m = gm - 1 - pid_m
    rows = pid_m * 16 + tl.arange(0, 16)
    cols = pid_n * 32 + tl.arange(0, 32)
    wb = pid_n * 64 + tl.arange(0, 64)
    p1 = tl.zeros((16, 32), dtype=tl.float64)
    p2 = tl.zeros((16, 32), dtype=tl.float64)
    p3 = tl.zeros((16, 32), dtype=tl.float64)
    diag_start = pid_m * 16
    offk = tl.arange(0, BLOCK_BULK_K)
    wa = tl.arange(0, 2 * BLOCK_BULK_K)
    wa_k = wa // 2
    for k0 in tl.range(0, diag_start, BLOCK_BULK_K):
        krow = k0 + offk
        kcol = k0 + wa_k
        a_blk = tl.load(
            a_ptr + rows[:, None] * ldar + (k0 * 2 + wa)[None, :],
            mask=(kcol < diag_start)[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        b_blk = tl.load(
            b_ptr + krow[:, None] * ldbr + wb[None, :],
            mask=(krow < diag_start)[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        ar, ai = tl.split(tl.reshape(a_blk, (16, BLOCK_BULK_K, 2)))
        br, bi = tl.split(tl.reshape(b_blk, (BLOCK_BULK_K, 32, 2)))
        p1 = tl.dot(ar, br, p1, out_dtype=tl.float64, allow_tf32=False)
        p2 = tl.dot(ai, bi, p2, out_dtype=tl.float64, allow_tf32=False)
        p3 = tl.dot(ar + ai, br + bi, p3, out_dtype=tl.float64, allow_tf32=False)
    offd = tl.arange(0, 16)
    wad = tl.arange(0, 32)
    kidx = diag_start + offd
    a_diag = tl.load(
        a_ptr + rows[:, None] * ldar + (diag_start * 2 + wad)[None, :],
        eviction_policy="evict_first",
    )
    b_diag = tl.load(
        b_ptr + kidx[:, None] * ldbr + wb[None, :], eviction_policy="evict_last"
    )
    ar, ai = tl.split(tl.reshape(a_diag, (16, 16, 2)))
    br, bi = tl.split(tl.reshape(b_diag, (16, 32, 2)))
    tri = kidx[None, :] <= rows[:, None]
    ar = tl.where(tri, ar, 0.0)
    ai = tl.where(tri, ai, 0.0)
    p1 = tl.dot(ar, br, p1, out_dtype=tl.float64, allow_tf32=False)
    p2 = tl.dot(ai, bi, p2, out_dtype=tl.float64, allow_tf32=False)
    p3 = tl.dot(ar + ai, br + bi, p3, out_dtype=tl.float64, allow_tf32=False)
    cr = p1 - p2
    ci = p3 - p1 - p2
    outr = ar_a * cr - ai_a * ci
    outi = ar_a * ci + ai_a * cr
    cb = c_ptr + rows[:, None] * ldcr + (cols * 2)[None, :]
    two = tl.arange(0, 2)
    tl.store(cb[:, :, None] + two[None, None, :], tl.join(outr, outi))


@triton.jit
def _ztrmm_512_left_t_upper_unit_bulkbk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    ar_a: tl.float64,
    ai_a: tl.float64,
    ldar: tl.constexpr,
    ldbr: tl.constexpr,
    ldcr: tl.constexpr,
    BLOCK_BULK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    gm = 32
    gn = 8
    width = GROUP_M * gn
    gid = pid // width
    gsize = tl.minimum(gm - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gsize)
    pid_n = (pid % width) // gsize
    pid_m = gm - 1 - pid_m
    rows = pid_m * 16 + tl.arange(0, 16)
    cols = pid_n * 32 + tl.arange(0, 32)
    wbm = tl.arange(0, 32)
    wb = pid_n * 64 + tl.arange(0, 64)
    abase = pid_m * 32 + wbm
    p1 = tl.zeros((16, 32), dtype=tl.float64)
    p2 = tl.zeros((16, 32), dtype=tl.float64)
    p3 = tl.zeros((16, 32), dtype=tl.float64)
    diag_start = pid_m * 16
    offk = tl.arange(0, BLOCK_BULK_K)
    for k0 in tl.range(0, diag_start, BLOCK_BULK_K):
        krow = k0 + offk
        a_blk = tl.load(
            a_ptr + krow[:, None] * ldar + abase[None, :],
            mask=(krow < diag_start)[:, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        b_blk = tl.load(
            b_ptr + krow[:, None] * ldbr + wb[None, :],
            mask=(krow < diag_start)[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        ar0, ai0 = tl.split(tl.reshape(a_blk, (BLOCK_BULK_K, 16, 2)))
        br, bi = tl.split(tl.reshape(b_blk, (BLOCK_BULK_K, 32, 2)))
        ar = tl.trans(ar0)
        ai = tl.trans(ai0)
        p2 = tl.dot(ai, bi, p2, out_dtype=tl.float64, allow_tf32=False)
        p1 = tl.dot(ar, br, p1, out_dtype=tl.float64, allow_tf32=False)
        p3 = tl.dot(ar + ai, br + bi, p3, out_dtype=tl.float64, allow_tf32=False)
    offd = tl.arange(0, 16)
    kidx = diag_start + offd
    r_for_wbm = diag_start + (wbm // 2)
    strict_mask = kidx[:, None] < r_for_wbm[None, :]
    a_diag = tl.load(
        a_ptr + kidx[:, None] * ldar + abase[None, :],
        mask=strict_mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    b_diag = tl.load(
        b_ptr + kidx[:, None] * ldbr + wb[None, :], eviction_policy="evict_last"
    )
    ar0, ai0 = tl.split(tl.reshape(a_diag, (16, 16, 2)))
    br, bi = tl.split(tl.reshape(b_diag, (16, 32, 2)))
    ar = tl.trans(ar0)
    ai = tl.trans(ai0)
    diag = kidx[None, :] == rows[:, None]
    ar = tl.where(diag, 1.0, ar)
    ai = tl.where(diag, 0.0, ai)
    p2 = tl.dot(ai, bi, p2, out_dtype=tl.float64, allow_tf32=False)
    p1 = tl.dot(ar, br, p1, out_dtype=tl.float64, allow_tf32=False)
    p3 = tl.dot(ar + ai, br + bi, p3, out_dtype=tl.float64, allow_tf32=False)
    cr = p1 - p2
    ci = p3 - p1 - p2
    outr = ar_a * cr - ai_a * ci
    outi = ar_a * ci + ai_a * cr
    cb = c_ptr + rows[:, None] * ldcr + (cols * 2)[None, :]
    two = tl.arange(0, 2)
    tl.store(cb[:, :, None] + two[None, None, :], tl.join(outr, outi))


def ztrmm(
    side: int,
    uplo: int,
    trans: int,
    diag: int,
    m: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
    ldc: int,
) -> None:
    assert A.dtype == torch.complex128 == B.dtype == C.dtype
    _check_trmm(A, B, C, side, uplo, trans, diag, m, n, lda, ldb, ldc, True)
    ar, ai = _complex_scalar(alpha)
    if m == 0 or n == 0:
        return
    if ar == 0.0 and ai == 0.0:
        C[:m, :n].zero_()
        return
    B_work = B.clone() if B.data_ptr() == C.data_ptr() else B
    if (
        side == CUBLAS_SIDE_LEFT
        and uplo == CUBLAS_FILL_MODE_UPPER
        and trans == CUBLAS_OP_T
        and diag == CUBLAS_DIAG_UNIT
        and m == 512
        and n == 256
    ):
        A_real = torch.view_as_real(A)
        B_real = torch.view_as_real(B_work)
        C_real = torch.view_as_real(C)
        with torch_device_fn.device(A.device):
            _ztrmm_512_left_t_upper_unit_bulkbk_kernel[
                (triton.cdiv(m, 16) * triton.cdiv(n, 32),)
            ](
                A_real,
                B_real,
                C_real,
                ar,
                ai,
                2 * lda,
                2 * ldb,
                2 * ldc,
                BLOCK_BULK_K=32,
                GROUP_M=8,
                num_warps=2,
                num_stages=3,
                maxnreg=256,
            )
        return
    if (
        side == CUBLAS_SIDE_LEFT
        and uplo == CUBLAS_FILL_MODE_LOWER
        and trans == CUBLAS_OP_N
        and diag == CUBLAS_DIAG_NON_UNIT
        and m == 512
        and n == 256
    ):
        A_real = torch.view_as_real(A)
        B_real = torch.view_as_real(B_work)
        C_real = torch.view_as_real(C)
        with torch_device_fn.device(A.device):
            _ztrmm_512_left_n_lower_nonunit_bulkbk_kernel[
                (triton.cdiv(m, 16) * triton.cdiv(n, 32),)
            ](
                A_real,
                B_real,
                C_real,
                ar,
                ai,
                2 * lda,
                2 * ldb,
                2 * ldc,
                BLOCK_BULK_K=32,
                GROUP_M=8,
                num_warps=4,
                num_stages=3,
                maxnreg=256,
            )
        return
    if (
        side == CUBLAS_SIDE_LEFT
        and (trans == CUBLAS_OP_T or trans == CUBLAS_OP_C)
        and m % 16 == 0
        and n % 16 == 0
        and max(m, n) <= 512
    ):
        A_real = torch.view_as_real(A)
        B_real = torch.view_as_real(B_work)
        C_real = torch.view_as_real(C)
        uplo_flag = 0 if uplo == CUBLAS_FILL_MODE_LOWER else 1
        unit_flag = 1 if diag == CUBLAS_DIAG_UNIT else 0
        conj_flag = 1 if trans == CUBLAS_OP_C else 0
        with torch_device_fn.device(A.device):
            _ztrmm_fused_left_t_kernel[(triton.cdiv(m, 16) * triton.cdiv(n, 32),)](
                A_real,
                B_real,
                C_real,
                ar,
                ai,
                m,
                n,
                2 * lda,
                2 * ldb,
                2 * ldc,
                UPLO=uplo_flag,
                UNIT=unit_flag,
                CONJ=conj_flag,
                BLOCK_M=16,
                BLOCK_N=32,
                BLOCK_K=16,
                GROUP_M=8,
                num_warps=2,
                num_stages=3,
                maxnreg=256,
            )
        return
    if (
        side == CUBLAS_SIDE_LEFT
        and trans == CUBLAS_OP_N
        and m % 16 == 0
        and n % 16 == 0
        and min(m, n) < _Z3M_MIN
        and max(m, n) <= 1024
    ):
        A_real = torch.view_as_real(A)
        B_real = torch.view_as_real(B_work)
        C_real = torch.view_as_real(C)
        uplo_flag = 0 if uplo == CUBLAS_FILL_MODE_LOWER else 1
        unit_flag = 1 if diag == CUBLAS_DIAG_UNIT else 0
        with torch_device_fn.device(A.device):
            _ztrmm_fused_left_n_kernel[(triton.cdiv(m, 16) * triton.cdiv(n, 32),)](
                A_real,
                B_real,
                C_real,
                ar,
                ai,
                m,
                n,
                2 * lda,
                2 * ldb,
                2 * ldc,
                UPLO=uplo_flag,
                UNIT=unit_flag,
                BLOCK_M=16,
                BLOCK_N=32,
                BLOCK_K=16,
                GROUP_M=8,
                num_warps=2,
                num_stages=3,
            )
        return
    if m >= _Z3M_MIN and n >= _Z3M_MIN:
        if trans == CUBLAS_OP_N:
            _ztrmm_3m_n(side, uplo, diag, m, n, ar, ai, A, lda, B_work, ldb, C, ldc)
            return
        if side == CUBLAS_SIDE_LEFT:
            _ztrmm_3m_trans(
                side, uplo, trans, diag, m, n, ar, ai, A, lda, B_work, ldb, C, ldc
            )
            return
        _ztrmm_3m_right_trans(
            uplo, trans, diag, m, n, ar, ai, A, lda, B_work, ldb, C, ldc
        )
        return
    A_real = torch.view_as_real(A)
    B_real = torch.view_as_real(B_work)
    C_real = torch.view_as_real(C)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    conj = 1 if trans == CUBLAS_OP_C else 0
    trans_flag = CUBLAS_OP_T if trans == CUBLAS_OP_C else trans
    with torch_device_fn.device(A.device):
        _ztrmm_kernel[grid](
            A_real,
            B_real,
            C_real,
            ar,
            ai,
            m,
            n,
            lda,
            ldb,
            ldc,
            _mode_key(side, uplo, trans, diag),
            SIDE=side,
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
            CONJ=conj,
        )
