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

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.dgemm import ScalarType, _validate_dgemm_args
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

logger = logging.getLogger(__name__)

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1


def _select_dgemm_config(transa: int, transb: int, m: int, n: int, k: int):
    """Select tile config optimized for Zhenwu PPU-ZW810E FP64 GEMM."""
    max_dim = max(m, n, k)
    min_dim = min(m, n, k)
    maxnreg = None
    num_stages = 3

    if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
        if max_dim <= 32:
            return 32, 16, 16, 4, 1, None, 3
        elif max_dim <= 128:
            return 16, 32, 32, 4, 1, None, 3
        elif max_dim <= 256:
            return 16, 32, 64, 4, 1, None, 3
        elif max_dim <= 512:
            if m % 32 == 0 and n % 32 == 0:
                return 32, 32, 32, 4, 1, None, 3
            else:
                return 64, 32, 32, 4, 1, None, 3
        elif max_dim <= 1024:
            if m % 64 == 0 and n % 64 == 0:
                return 64, 128, 64, 8, 1, 224, 3
            else:
                return 64, 128, 16, 4, 1, None, 3
        elif max_dim <= 1536:
            return 64, 32, 32, 4, 1, 128, 3
        elif max_dim <= 2048:
            return 64, 128, 32, 4, 8, None, 3
        elif max_dim == 4095:
            return 64, 128, 16, 4, 16, None, 3
        elif max_dim < 4096 and min_dim >= 2048:
            return 64, 128, 32, 4, 2, None, 3
        elif max_dim <= 4096:
            return 64, 128, 32, 4, 8, None, 3
        elif min_dim >= 8192:
            return 64, 128, 32, 4, 8, None, 3
        elif min_dim >= 6144:
            return 128, 64, 32, 4, 8, None, 3
        else:
            return 64, 128, 32, 4, 8, None, 3

    elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
        if max_dim == 4095:
            return 64, 128, 16, 4, 16, None, 3
        elif max_dim == 511:
            return 32, 32, 64, 4, 4, None, 3
        elif max_dim <= 32:
            return 16, 32, 16, 4, 4, None, 3
        elif max_dim <= 64:
            return 16, 16, 64, 4, 4, None, 3
        elif max_dim <= 128:
            return 16, 16, 64, 4, 1, None, 3
        elif max_dim == 256:
            return 32, 32, 16, 8, 4, None, 3
        elif max_dim <= 256:
            return 32, 16, 64, 4, 1, None, 3
        elif max_dim == 512:
            return 32, 32, 64, 4, 4, None, 3
        elif max_dim <= 512:
            return 64, 32, 64, 8, 1, None, 3
        elif max_dim <= 1024:
            return 64, 64, 32, 4, 8, None, 3
        elif max_dim <= 1536:
            return 128, 64, 32, 4, 8, None, 3
        elif max_dim <= 4096:
            return 128, 64, 32, 4, 8, None, 3
        elif min_dim >= 6144:
            return 128, 64, 16, 4, 8, None, 3
        else:
            return 64, 128, 32, 4, 8, None, 3

    elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
        if max_dim == 4095:
            return 64, 128, 16, 4, 16, None, 3
        elif max_dim == 511:
            return 32, 32, 64, 4, 4, None, 3
        elif max_dim <= 32:
            return 16, 16, 16, 4, 8, None, 3
        elif max_dim <= 64:
            return 16, 16, 64, 4, 1, None, 3
        elif max_dim <= 128:
            return 16, 16, 64, 4, 8, None, 3
        elif max_dim == 256:
            return 32, 32, 16, 8, 4, None, 3
        elif max_dim <= 256:
            return 16, 32, 64, 4, 8, None, 3
        elif max_dim == 512:
            return 32, 32, 64, 4, 4, None, 3
        elif max_dim == 1023:
            return 64, 64, 16, 4, 8, None, 3
        elif max_dim <= 512:
            return 32, 64, 64, 4, 8, None, 3
        elif max_dim <= 1024:
            return 32, 128, 16, 4, 1, 168, 3
        elif max_dim <= 1536:
            return 64, 64, 16, 4, 1, 168, 3
        elif max_dim <= 4096:
            return 128, 64, 32, 4, 8, None, 3
        elif min_dim >= 8192:
            return 128, 64, 32, 4, 8, None, 3
        elif min_dim >= 6144:
            return 128, 64, 32, 4, 4, None, 3
        else:
            return 64, 128, 32, 4, 8, None, 3

    else:  # TT
        if max_dim == 4095:
            return 64, 128, 16, 4, 16, None, 3
        elif max_dim == 511:
            return 32, 32, 64, 4, 4, None, 3
        elif max_dim <= 32:
            return 16, 16, 16, 4, 8, None, 3
        elif max_dim <= 64:
            return 16, 16, 64, 4, 4, None, 3
        elif max_dim <= 128:
            return 16, 32, 64, 4, 4, None, 3
        elif max_dim == 256:
            return 16, 32, 64, 4, 8, None, 3
        elif max_dim == 512:
            return 32, 32, 64, 4, 4, None, 3
        elif max_dim <= 512:
            return 32, 32, 64, 4, 1, None, 3
        elif max_dim <= 1536:
            return 64, 64, 16, 4, 1, None, 3
        elif max_dim <= 4096:
            return 128, 64, 32, 4, 8, None, 3
        elif min_dim >= 8192:
            return 128, 64, 32, 4, 8, None, 3
        elif min_dim >= 6144:
            return 128, 64, 32, 4, 4, None, 3
        else:
            return 64, 128, 32, 4, 8, None, 3


@libentry()
@triton.jit
def _dgemm_dot_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float64,
    beta: tl.float64,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
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
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for k_start in range(0, k, BLOCK_K):
        cur_k = k_start + offs_k
        mask_k = cur_k < k
        if TRANS_A == 0:
            a_ptrs = a_ptr + offs_m[:, None] * lda + cur_k[None, :]
        else:
            a_ptrs = a_ptr + cur_k[None, :] * lda + offs_m[:, None]

        if TRANS_B == 0:
            b_ptrs = b_ptr + cur_k[:, None] * ldb + offs_n[None, :]
        else:
            b_ptrs = b_ptr + offs_n[None, :] * ldb + cur_k[:, None]

        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float64)
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float64)
        acc += tl.dot(a, b, out_dtype=tl.float64, allow_tf32=False)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float64)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


def dgemm(
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
    _validate_dgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if m == 0 or n == 0 or k == 0 or alpha == 0.0:
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)
        return

    beta_is_zero = beta == 0.0

    block_m, block_n, block_k, num_warps, group_m, maxnreg, num_stages = (
        _select_dgemm_config(transa, transb, m, n, k)
    )

    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    launch_kwargs = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    if maxnreg is not None:
        launch_kwargs["maxnreg"] = maxnreg

    with torch_device_fn.device(A.device):
        _dgemm_dot_kernel[grid](
            A,
            B,
            C,
            alpha,
            beta,
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            transa,
            transb,
            beta_is_zero,
            **launch_kwargs,
        )


__all__ = ["dgemm"]
