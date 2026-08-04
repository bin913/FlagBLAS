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
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                a = tl.load(a_ptrs, mask=offs_m[:, None] < m, other=0.0)
                b = tl.load(b_ptrs, mask=offs_n[None, :] < n, other=0.0)
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
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
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
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
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


def _select_hgemm_config(m: int, n: int, k: int):
    if max(m, n, k) >= 2048 and min(m, n) >= 128:
        return 128, 128, 32, 8, 8
    if m <= 64 and n >= 128:
        return 16, 128, 32, 4, 8
    if n <= 64 and m >= 128:
        return 128, 16, 32, 4, 8
    if max(m, n, k) <= 512:
        return 64, 32, 32, 4, 8
    return 128, 128, 32, 8, 8


def _can_use_fast_hgemm(m: int, n: int, k: int, block_m: int, block_n: int, block_k: int) -> bool:
    return (m % block_m == 0) and (n % block_n == 0) and (k % block_k == 0)


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

    block_m, block_n, block_k, num_warps, group_m = _select_hgemm_config(m, n, k)
    check_bounds = not _can_use_fast_hgemm(m, n, k, block_m, block_n, block_k)
    beta_is_zero = beta == 0.0
    trans_a = transa == CUBLAS_OP_T
    trans_b = transb == CUBLAS_OP_T

    with torch_device_fn.device(A.device):
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        _hgemm_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
            trans_a, trans_b, check_bounds, False, 0, 0,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
            num_warps=num_warps, num_stages=3,
        )
