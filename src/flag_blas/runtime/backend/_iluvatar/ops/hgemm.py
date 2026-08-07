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

from typing import Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn

ScalarType = Union[float, int, complex, torch.Tensor]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2


@triton.jit
def _hgemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    CHECK_BOUNDS: tl.constexpr,
    SKIP_FULL: tl.constexpr,
    FULL_GRID_M: tl.constexpr,
    FULL_GRID_N: tl.constexpr,
    CACHE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    if SKIP_FULL and pid_m < FULL_GRID_M and pid_n < FULL_GRID_N:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if CHECK_BOUNDS:
        is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
        is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
        k_full_iters = k // BLOCK_K
        k_remainder = k % BLOCK_K

        for ki in range(k_full_iters):
            offs_k = ki * BLOCK_K + offs_k_base
            if TRANS_A:
                a_ptrs = a_ptr + offs_k[None, :] * lda + offs_m[:, None]
            else:
                a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
            if TRANS_B:
                b_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k[:, None]
            else:
                b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

            if is_full_m and is_full_n:
                a = tl.load(a_ptrs, cache_modifier=CACHE)
                b = tl.load(b_ptrs, cache_modifier=CACHE)
            else:
                a = tl.load(a_ptrs, mask=offs_m[:, None] < m, other=0.0, cache_modifier=CACHE)
                b = tl.load(b_ptrs, mask=offs_n[None, :] < n, other=0.0, cache_modifier=CACHE)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

        if k_remainder > 0:
            offs_k = k_full_iters * BLOCK_K + offs_k_base
            if TRANS_A:
                a_ptrs = a_ptr + offs_k[None, :] * lda + offs_m[:, None]
            else:
                a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
            if TRANS_B:
                b_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k[:, None]
            else:
                b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
            a_mask = (offs_m[:, None] < m) & (offs_k[None, :] < k)
            b_mask = (offs_k[:, None] < k) & (offs_n[None, :] < n)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0, cache_modifier=CACHE)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=CACHE)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
    else:
        if UNROLL >= 4:
            k_unroll = BLOCK_K * 4
            k_full = (k // k_unroll) * k_unroll
            for k_start in range(0, k_full, k_unroll):
                offs_k0 = k_start + offs_k_base
                offs_k1 = k_start + BLOCK_K + offs_k_base
                offs_k2 = k_start + 2 * BLOCK_K + offs_k_base
                offs_k3 = k_start + 3 * BLOCK_K + offs_k_base
                if TRANS_A:
                    a0_ptrs = a_ptr + offs_k0[None, :] * lda + offs_m[:, None]
                    a1_ptrs = a_ptr + offs_k1[None, :] * lda + offs_m[:, None]
                    a2_ptrs = a_ptr + offs_k2[None, :] * lda + offs_m[:, None]
                    a3_ptrs = a_ptr + offs_k3[None, :] * lda + offs_m[:, None]
                else:
                    a0_ptrs = a_ptr + offs_m[:, None] * lda + offs_k0[None, :]
                    a1_ptrs = a_ptr + offs_m[:, None] * lda + offs_k1[None, :]
                    a2_ptrs = a_ptr + offs_m[:, None] * lda + offs_k2[None, :]
                    a3_ptrs = a_ptr + offs_m[:, None] * lda + offs_k3[None, :]
                if TRANS_B:
                    b0_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k0[:, None]
                    b1_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k1[:, None]
                    b2_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k2[:, None]
                    b3_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k3[:, None]
                else:
                    b0_ptrs = b_ptr + offs_k0[:, None] * ldb + offs_n[None, :]
                    b1_ptrs = b_ptr + offs_k1[:, None] * ldb + offs_n[None, :]
                    b2_ptrs = b_ptr + offs_k2[:, None] * ldb + offs_n[None, :]
                    b3_ptrs = b_ptr + offs_k3[:, None] * ldb + offs_n[None, :]
                a0 = tl.load(a0_ptrs, cache_modifier=CACHE)
                b0 = tl.load(b0_ptrs, cache_modifier=CACHE)
                acc = tl.dot(a0, b0, acc, out_dtype=tl.float32, allow_tf32=False)
                a1 = tl.load(a1_ptrs, cache_modifier=CACHE)
                b1 = tl.load(b1_ptrs, cache_modifier=CACHE)
                acc = tl.dot(a1, b1, acc, out_dtype=tl.float32, allow_tf32=False)
                a2 = tl.load(a2_ptrs, cache_modifier=CACHE)
                b2 = tl.load(b2_ptrs, cache_modifier=CACHE)
                acc = tl.dot(a2, b2, acc, out_dtype=tl.float32, allow_tf32=False)
                a3 = tl.load(a3_ptrs, cache_modifier=CACHE)
                b3 = tl.load(b3_ptrs, cache_modifier=CACHE)
                acc = tl.dot(a3, b3, acc, out_dtype=tl.float32, allow_tf32=False)
            for k_start in range(k_full, k, BLOCK_K):
                offs_k = k_start + offs_k_base
                if TRANS_A:
                    a_ptrs = a_ptr + offs_k[None, :] * lda + offs_m[:, None]
                else:
                    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
                if TRANS_B:
                    b_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k[:, None]
                else:
                    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
                a = tl.load(a_ptrs, cache_modifier=CACHE)
                b = tl.load(b_ptrs, cache_modifier=CACHE)
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        else:
            for k_start in range(0, k, BLOCK_K):
                offs_k = k_start + offs_k_base
                if TRANS_A:
                    a_ptrs = a_ptr + offs_k[None, :] * lda + offs_m[:, None]
                else:
                    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
                if TRANS_B:
                    b_ptrs = b_ptr + offs_n[None, :] * ldb + offs_k[:, None]
                else:
                    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
                a = tl.load(a_ptrs, cache_modifier=CACHE)
                b = tl.load(b_ptrs, cache_modifier=CACHE)
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)


    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = alpha * acc
    if CHECK_BOUNDS:
        c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
        if not BETA_IS_ZERO:
            result += beta * tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, result.to(tl.float16), mask=c_mask)
    else:
        if not BETA_IS_ZERO:
            result += beta * tl.load(c_ptrs).to(tl.float32)
        tl.store(c_ptrs, result.to(tl.float16))


@triton.jit
def _hgemm_tt_transpose_dot_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    CACHE: tl.constexpr,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k_start in range(0, k, BLOCK_K):
        offs_k = k_start + offs_k_base
        a = tl.load(
            a_ptr + offs_k[:, None] * lda + offs_m[None, :],
            cache_modifier=CACHE,
        )
        b = tl.load(
            b_ptr + offs_n[:, None] * ldb + offs_k[None, :],
            cache_modifier=CACHE,
        )
        acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    acc = tl.trans(acc_t)
    result = alpha * acc
    if not BETA_IS_ZERO:
        result += beta * tl.load(c_ptrs).to(tl.float32)
    tl.store(c_ptrs, result.to(tl.float16))


@triton.jit
def _hgemm_tn_transpose_dot_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    CACHE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """TN variant of the transposed-dot kernel.

    For TN (C = A^T @ B) with A stored (k, m) and B stored (k, n) both
    row-major, loading A as [BLOCK_K, BLOCK_M] is fully coalesced (inner
    dimension m is contiguous), whereas the standard kernel's [BLOCK_M,
    BLOCK_K] A tile gathers along k.  The result is accumulated in the
    transposed orientation C^T = B^T @ A and transposed once before store.
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k_start in range(0, k, BLOCK_K):
        offs_k = k_start + offs_k_base
        a = tl.load(
            a_ptr + offs_k[:, None] * lda + offs_m[None, :],
            cache_modifier=CACHE,
        )
        b = tl.load(
            b_ptr + offs_k[None, :] * ldb + offs_n[:, None],
            cache_modifier=CACHE,
        )
        acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    acc = tl.trans(acc_t)
    result = alpha * acc
    if not BETA_IS_ZERO:
        result += beta * tl.load(c_ptrs).to(tl.float32)
    tl.store(c_ptrs, result.to(tl.float16))


def _select_hgemm_config(m: int, n: int, k: int, transa: int, transb: int):
    """Select (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, group_m, num_stages).

    Configurations are derived from extensive sweeps on Iluvatar BI-V150.
    """
    # ---- Smallest square ----
    if m == 64 and n == 64 and k == 64:
        return 64, 64, 64, 4, 8, 3, 1

    # ---- Tall / skinny (small m) ----
    if m <= 64:
        return 64, 64, 128, 4, 8, 4, 1

    # ---- Short / wide (small n) ----
    if n <= 64:
        return 64, 64, 128, 4, 8, 4, 1

    # ---- Small squares (max dim <= 512) ----
    if max(m, n, k) <= 512:
        return 64, 64, 128, 4, 8, 4, 1

    # ---- Exact core-shape fixes for previously under-threshold cases ----
    if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
        if m == 2048 and n == 2048 and k == 2048:
            return 128, 128, 64, 8, 4, 5, 1
        if m == 4096 and n == 4096 and k == 4096:
            return 128, 128, 64, 8, 4, 3, 1
        if m == 8192 and n == 8192 and k == 8192:
            return 128, 128, 64, 8, 8, 2, 1
        if m == 16384 and n == 16384 and k == 16384:
            return 256, 256, 64, 8, 8, 2, 1
        if m == 2048 and n == 12288 and k == 4096:
            return 128, 128, 64, 8, 8, 5, 1
        if m == 2048 and n == 11008 and k == 4096:
            return 128, 128, 64, 8, 8, 3, 1
        if m == 2048 and n == 4096 and k == 11008:
            return 128, 128, 64, 8, 8, 4, 1
        if m == 4096 and n == 24576 and k == 8192:
            return 128, 128, 64, 8, 8, 2, 1
        if m == 4096 and n == 8192 and k == 28672:
            return 128, 128, 64, 8, 8, 4, 1
        if m == 8192 and n == 28672 and k == 8192:
            return 128, 128, 64, 8, 8, 4, 1
        if m == 16384 and n == 2048 and k == 2048:
            return 128, 128, 64, 8, 4, 3, 1
        if m == 2048 and n == 16384 and k == 2048:
            return 128, 128, 64, 8, 16, 3, 1
        if m == 32768 and n == 1024 and k == 1024:
            return 128, 128, 64, 8, 8, 5, 1
        if m == 4096 and n == 128 and k == 1024:
            return 128, 128, 128, 16, 16, 3, 4
        if m == 8192 and n == 256 and k == 2048:
            return 128, 128, 64, 16, 4, 2, 1
        if m == 16384 and n == 512 and k == 4096:
            return 128, 128, 64, 8, 8, 4, 1
        if m == 512 and n == 16384 and k == 4096:
            return 128, 128, 64, 8, 8, 5, 1
    if transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
        if m == 2048 and n == 2048 and k == 2048:
            return 128, 128, 64, 16, 4, 2, 1
        if m == 4096 and n == 4096 and k == 4096:
            return 128, 128, 64, 16, 8, 2, 1
        if m == 8192 and n == 8192 and k == 8192:
            return 128, 128, 64, 16, 4, 5, 1
        if m == 16384 and n == 16384 and k == 16384:
            return 128, 128, 64, 16, 8, 3, 1
        if m == 2048 and n == 16384 and k == 2048:
            return 128, 128, 64, 16, 16, 2, 1
        if m == 16384 and n == 2048 and k == 2048:
            return 128, 128, 64, 16, 4, 10, 1
        if m == 16384 and n == 512 and k == 4096:
            return 128, 128, 64, 16, 2, 2, 1
        if m == 512 and n == 16384 and k == 4096:
            return 256, 256, 64, 8, 8, 8, 1
        if m == 2048 and n == 2048 and k == 16384:
            return 128, 128, 64, 16, 8, 2, 1
        if m == 4096 and n == 24576 and k == 8192:
            return 128, 128, 64, 16, 4, 3, 4
        if m == 2048 and n == 11008 and k == 4096:
            return 128, 128, 64, 16, 8, 2, 1
        if m == 2048 and n == 12288 and k == 4096:
            return 128, 128, 64, 16, 4, 3, 4
        if m == 8192 and n == 28672 and k == 8192:
            return 128, 128, 64, 16, 4, 4, 1
        if m == 8192 and n == 256 and k == 2048:
            return 128, 128, 64, 16, 2, 2, 1
        if m == 256 and n == 8192 and k == 2048:
            return 128, 128, 64, 16, 2, 2, 1
        if m == 4096 and n == 8192 and k == 28672:
            return 128, 128, 64, 16, 4, 6, 1
        if m == 32768 and n == 1024 and k == 1024:
            return 128, 128, 64, 16, 4, 3, 1
    if transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
        if m == 128 and n == 4096 and k == 1024:
            return 128, 128, 128, 16, 4, 2, 1
        if m == 256 and n == 8192 and k == 2048:
            return 128, 128, 64, 16, 8, 3, 1
    if transa == CUBLAS_OP_T and transb == CUBLAS_OP_T:
        if m == 128 and n == 4096 and k == 1024:
            return 128, 128, 128, 16, 8, 4, 1

    # ---- 128-ish narrow shapes ----
    if m == 128:
        return 64, 128, 64, 8, 8, 4, 1
    if n == 128:
        if transa == CUBLAS_OP_T:
            return 128, 64, 64, 8, 8, 4, 1
        return 64, 64, 128, 4, 8, 4, 1

    # ---- Medium / large shapes (max dim <= 2048, e.g. 1024^3 / 2048^3) ----
    if max(m, n, k) <= 2048:
        return 128, 128, 64, 16, 4, 3, 1

    # ---- Default large ----
    return 128, 128, 64, 16, 4, 3, 1


def _select_hgemm_tt_transpose_dot_config(m: int, n: int, k: int):
    if m == 256 and n == 8192 and k == 2048:
        return 128, 128, 64, 16, 16, 2
    if m == 512 and n == 16384 and k == 4096:
        return 128, 128, 64, 16, 8, 2
    if m == 16384 and n == 512 and k == 4096:
        return 128, 128, 64, 16, 4, 3
    if m == 4096 and n == 4096 and k == 4096:
        return 128, 128, 64, 16, 8, 2
    return None


def _select_hgemm_tn_transpose_dot_config(m: int, n: int, k: int):
    """Per-shape (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, group_m, num_stages)
    for the TN transposed-dot kernel, from a sweep on Iluvatar BI-V150.

    The transposed-dot kernel loads A fully coalesced (A is stored (k, m)
    for TN), which wins for most large shapes; it is disabled for shapes
    where the one-time fp32 accumulator transpose is more expensive than
    the strided-A savings (e.g. big tiles / extreme aspect ratios).
    """
    if m == 4096 and n == 4096 and k == 4096:
        return 128, 128, 64, 16, 8, 4
    if m == 8192 and n == 8192 and k == 8192:
        return 128, 128, 64, 16, 4, 6
    if m == 16384 and n == 16384 and k == 16384:
        return 128, 128, 64, 16, 8, 3
    if m == 16384 and n == 512 and k == 4096:
        return 128, 128, 64, 16, 2, 2
    if m == 2048 and n == 2048 and k == 16384:
        return 128, 128, 64, 16, 8, 2
    if m == 2048 and n == 12288 and k == 4096:
        return 128, 128, 64, 16, 4, 3
    if m == 8192 and n == 256 and k == 2048:
        return 128, 128, 64, 16, 2, 10
    if m == 256 and n == 8192 and k == 2048:
        return 128, 128, 64, 16, 2, 2
    if m == 4096 and n == 24576 and k == 8192:
        return 128, 128, 64, 16, 4, 4
    if m == 2048 and n == 11008 and k == 4096:
        return 128, 128, 64, 16, 8, 2
    if m == 8192 and n == 28672 and k == 8192:
        return 128, 128, 64, 16, 4, 4
    return None


def _can_use_fast_hgemm(m: int, n: int, k: int, block_m: int, block_n: int, block_k: int) -> bool:
    return (m % block_m == 0) and (n % block_n == 0) and (k % block_k == 0)


def _should_pretranspose_b(m: int, n: int, k: int) -> bool:
    """Transposing B is profitable only when B is large enough that the
    transposed-load software gather (per-element) dominates the one-time
    contiguous copy cost. Derived from sweeps on Iluvatar BI-V150: for
    transb == T with large K, converting to the transb == N load path yields
    up to ~35% speedup, while the copy is fully amortized over the K loop."""
    return k >= 8192 and min(m, n) >= 2048


def _launch_hgemm(
    transa: int,
    transb: int,
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    beta_is_zero: bool,
    check_bounds: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
    unroll: int,
) -> None:
    _hgemm_kernel[grid](
        A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
        transa == CUBLAS_OP_T, transb == CUBLAS_OP_T, check_bounds, False, 0, 0,
        ".cg",
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
        UNROLL=unroll, num_warps=num_warps, num_stages=num_stages,
    )


def _launch_hgemm_tt_transpose_dot(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    beta_is_zero: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
) -> None:
    _hgemm_tt_transpose_dot_kernel[grid](
        A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero, ".cg",
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
        num_warps=num_warps, num_stages=num_stages,
    )


def _launch_hgemm_tn_transpose_dot(
    grid,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float,
    m: int,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldc: int,
    beta_is_zero: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    group_m: int,
    num_stages: int,
) -> None:
    _hgemm_tn_transpose_dot_kernel[grid](
        A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero, ".cg",
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
        num_warps=num_warps, num_stages=num_stages,
    )


def hgemm(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    beta: ScalarType,
    C: torch.Tensor,
    ldc: int,
) -> None:
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()
    assert A.dtype == torch.float16
    assert B.dtype == torch.float16
    assert C.dtype == torch.float16
    assert A.device == B.device == C.device
    assert transa in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert transb in [CUBLAS_OP_N, CUBLAS_OP_T]

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if m == 0 or n == 0 or k == 0 or alpha == 0.0:
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)
        return

    # ---- B-transposed large-shape fast path ----
    # The transposed-B load path uses slow per-element software gathers. For
    # large shapes, transpose B once (a single coalesced copy) and reuse the
    # fast transb == N kernel; the copy is amortized over the long K loop.
    if transb == CUBLAS_OP_T and _should_pretranspose_b(m, n, k):
        B = B.t().contiguous()
        transb = CUBLAS_OP_N
        ldb = n

    beta_is_zero = beta == 0.0

    tt_transpose_dot_config = None
    if transa == CUBLAS_OP_T and transb == CUBLAS_OP_T:
        tt_transpose_dot_config = _select_hgemm_tt_transpose_dot_config(m, n, k)
    if tt_transpose_dot_config is not None:
        block_m, block_n, block_k, num_warps, group_m, num_stages = tt_transpose_dot_config
        if _can_use_fast_hgemm(m, n, k, block_m, block_n, block_k):
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            with torch_device_fn.device(A.device):
                _launch_hgemm_tt_transpose_dot(
                    grid, A, B, C, alpha, beta, m, n, k, lda, ldb, ldc,
                    beta_is_zero, block_m, block_n, block_k, num_warps,
                    group_m, num_stages,
                )
            return

    tn_transpose_dot_config = None
    if transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
        tn_transpose_dot_config = _select_hgemm_tn_transpose_dot_config(m, n, k)
    if tn_transpose_dot_config is not None:
        block_m, block_n, block_k, num_warps, group_m, num_stages = tn_transpose_dot_config
        if _can_use_fast_hgemm(m, n, k, block_m, block_n, block_k):
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            with torch_device_fn.device(A.device):
                _launch_hgemm_tn_transpose_dot(
                    grid, A, B, C, alpha, beta, m, n, k, lda, ldb, ldc,
                    beta_is_zero, block_m, block_n, block_k, num_warps,
                    group_m, num_stages,
                )
            return

    block_m, block_n, block_k, num_warps, group_m, num_stages, unroll = _select_hgemm_config(
        m, n, k, transa, transb
    )
    check_bounds = not _can_use_fast_hgemm(m, n, k, block_m, block_n, block_k)

    with torch_device_fn.device(A.device):
        # ---- Padding path: pad to block-aligned dims and run fast no-bounds kernel ----
        if check_bounds and max(m, n, k) >= 2048:
            padded_m = triton.cdiv(m, block_m) * block_m
            padded_n = triton.cdiv(n, block_n) * block_n
            padded_k = triton.cdiv(k, block_k) * block_k
            if transa == CUBLAS_OP_N:
                A_pad = F.pad(A, (0, padded_k - k, 0, padded_m - m))
                lda_pad = padded_k
            else:
                A_pad = F.pad(A, (0, padded_m - m, 0, padded_k - k))
                lda_pad = padded_m
            if transb == CUBLAS_OP_N:
                B_pad = F.pad(B, (0, padded_n - n, 0, padded_k - k))
                ldb_pad = padded_n
            else:
                B_pad = F.pad(B, (0, padded_k - k, 0, padded_n - n))
                ldb_pad = padded_k
            if beta_is_zero:
                C_pad = torch.empty((padded_m, padded_n), device=C.device, dtype=C.dtype)
            else:
                C_pad = F.pad(C, (0, padded_n - n, 0, padded_m - m))
            grid_pad = (triton.cdiv(padded_m, block_m) * triton.cdiv(padded_n, block_n),)
            _launch_hgemm(
                transa, transb, grid_pad, A_pad, B_pad, C_pad, alpha, beta,
                padded_m, padded_n, padded_k, lda_pad, ldb_pad, padded_n,
                beta_is_zero, False, block_m, block_n, block_k, num_warps,
                group_m, num_stages, unroll,
            )
            C.copy_(C_pad[:m, :n])
            return

        # ---- Simple path ----
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        _launch_hgemm(
            transa, transb, grid, A, B, C, alpha, beta, m, n, k, lda, ldb, ldc,
            beta_is_zero, check_bounds, block_m, block_n, block_k, num_warps,
            group_m, num_stages, unroll,
        )
