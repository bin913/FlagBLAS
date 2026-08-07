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

from flag_blas.ops.level3.bfgemm import (
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
    _bfgemm_nn_kernel,
    _bfgemm_nt_kernel,
    _bfgemm_tn_kernel,
    _bfgemm_tt_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.backend._thead.ops.sgemm import _is_gemm_aligned
from flag_blas.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _thead_bfgemm_nn_impl(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
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

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= M
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= N
    k_full_iters = K // BLOCK_K
    k_remainder = K % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    result = alpha * acc
    if is_full_m and is_full_n:
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.bfloat16))
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.bfloat16), mask=c_mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_bfgemm_nn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    _thead_bfgemm_nn_impl(
        a_ptr,
        b_ptr,
        c_ptr,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        BETA_IS_ZERO,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    )


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_bfgemm_nn_bwd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    _thead_bfgemm_nn_impl(
        a_ptr,
        b_ptr,
        c_ptr,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        BETA_IS_ZERO,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    )


@libentry()
@triton.jit
def _thead_bfgemm_pad2d_kernel(
    src_ptr,
    dst_ptr,
    rows,
    cols,
    src_ld,
    dst_ld,
    dst_rows,
    dst_cols,
    BLOCK_SIZE: tl.constexpr,
):
    src_ptr = src_ptr.to(tl.pointer_type(tl.bfloat16))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.bfloat16))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dst_rows * dst_cols
    r = offsets // dst_cols
    c = offsets - r * dst_cols
    in_bounds = (r < rows) & (c < cols)
    vals = tl.load(src_ptr + r * src_ld + c, mask=mask & in_bounds, other=0.0)
    tl.store(dst_ptr + r * dst_ld + c, vals, mask=mask)


@libentry()
@triton.jit
def _thead_bfgemm_crop_c_kernel(
    src_ptr,
    dst_ptr,
    beta: tl.float32,
    rows,
    cols,
    src_ld,
    dst_ld,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    src_ptr = src_ptr.to(tl.pointer_type(tl.bfloat16))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.bfloat16))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < rows * cols
    r = offsets // cols
    c = offsets - r * cols
    vals = tl.load(src_ptr + r * src_ld + c, mask=mask, other=0.0).to(tl.float32)
    dst_offsets = r * dst_ld + c
    if not BETA_IS_ZERO:
        dst_vals = tl.load(dst_ptr + dst_offsets, mask=mask, other=0.0).to(tl.float32)
        vals += beta * dst_vals
    tl.store(dst_ptr + dst_offsets, vals.to(tl.bfloat16), mask=mask)


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_bfgemm_nn_desc_bwd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        b = b_desc.load([offs_k, offs_n])
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    if ALPHA_IS_ONE:
        result = acc
    else:
        result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.bfloat16))


@libentry()
@triton.jit
def _thead_bfgemm_nn_desc_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.bfloat16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.bfloat16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.bfloat16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = i * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        b = b_desc.load([offs_k, offs_n])
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        result += beta * c_vals
    c_desc.store([offs_m, offs_n], result.to(tl.bfloat16))


def _round_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


def _thead_bfgemm_nn_use_desc_bwd(m: int, n: int, k: int) -> bool:
    """Mainline: desc_bwd for all but tiny shapes.

    Measured on Zhenwu (bf16): desc is the fastest path once the shape is
    large enough for the tensor-descriptor pipeline to amortize launch cost.
    """
    return max(m, n, k) >= 128


def _thead_bfgemm_nn_use_impl(m: int, n: int, k: int) -> bool:
    """Tiny shapes are launch-bound; use the lightweight masked kernel."""
    return max(m, n, k) < 128


def _thead_bfgemm_nn_config(m: int, n: int, k: int):
    max_mnk = max(m, n, k)
    min_mn = min(m, n)

    # 64^3: impl (16,16,64) = 1.28
    if max_mnk <= 64:
        return 16, 16, 64, 1, 3, 128

    # 256^3: desc (64,64,64) = 1.30 (vs impl 16,16,64 = 0.95)
    if max_mnk <= 256:
        return 64, 64, 64, 4, 3, 128

    # 511^3 (non-aligned small): desc (64,64,64) = 1.17 (vs 128,256,64 = 0.69)
    if max_mnk <= 512 and (m % 64 != 0 or n % 64 != 0 or k % 64 != 0):
        return 64, 64, 64, 4, 3, 128

    if min_mn <= 64:
        return 64, 64, 64, 4, 3, 128

    if min_mn <= 128:
        return 64, 128, 64, 4, 3, 128

    if m == n == k == 512:
        return 128, 128, 128, 8, 3, 128

    return 128, 256, 64, 8, 3, 160


def _thead_bfgemm_nn_should_pad(m: int, n: int, k: int) -> bool:
    """Pad big non-aligned shapes only when the aligned GEMM + pad/crop
    beats the direct desc (which pays a boundary-check penalty when both M
    and N are non-divisible, e.g. 8191^3)."""
    if m % 64 == 0 and n % 64 == 0 and k % 64 == 0:
        return False
    # 8191^3 measured: padded 8.69ms < desc 8.85ms; 4095^3 measured:
    # padded 1.21ms > desc 1.16ms.  Threshold on total size.
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    extra = m_pad * k_pad + k_pad * n_pad + m_pad * n_pad
    original = m * k + k * n + m * n
    return (
        m * n * k >= 4096 * 4096 * 4096
        and extra <= original * 1.08
        and m_pad * n_pad * k_pad >= 4096 * 4096 * 4096
    )


def _can_use_thead_bfgemm_nn(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    return (
        lda == k
        and ldb == n
        and ldc == n
        and m >= 16
        and n >= 16
        and k >= 16
    )


def _run_thead_bfgemm_nn(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, aligned
):
    block_m, block_n, block_k, num_warps, num_stages, maxnreg = _thead_bfgemm_nn_config(
        m, n, k
    )
    kwargs = dict(
        M=m,
        N=n,
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )
    if _thead_bfgemm_nn_use_desc_bwd(m, n, k):
        kernel = _thead_bfgemm_nn_desc_bwd_kernel
        kwargs["ALPHA_IS_ONE"] = alpha == 1.0
    else:
        kernel = _thead_bfgemm_nn_kernel
    kernel[(triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)](
        A,
        B,
        C,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        **kwargs,
    )


def _run_thead_bfgemm_nn_padded(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    A_pad = torch.empty((m_pad, k_pad), dtype=A.dtype, device=A.device)
    B_pad = torch.empty((k_pad, n_pad), dtype=B.dtype, device=B.device)
    C_pad = torch.empty((m_pad, n_pad), dtype=C.dtype, device=C.device)

    pad_block = 1024
    _thead_bfgemm_pad2d_kernel[(triton.cdiv(m_pad * k_pad, pad_block),)](
        A, A_pad, m, k, lda, k_pad, m_pad, k_pad, BLOCK_SIZE=pad_block
    )
    _thead_bfgemm_pad2d_kernel[(triton.cdiv(k_pad * n_pad, pad_block),)](
        B, B_pad, k, n, ldb, n_pad, k_pad, n_pad, BLOCK_SIZE=pad_block
    )
    _run_thead_bfgemm_nn(
        A_pad,
        k_pad,
        B_pad,
        n_pad,
        C_pad,
        n_pad,
        m_pad,
        n_pad,
        k_pad,
        alpha,
        0.0,
        True,
        True,
    )
    _thead_bfgemm_crop_c_kernel[(triton.cdiv(m * n, pad_block),)](
        C_pad,
        C,
        beta,
        m,
        n,
        n_pad,
        ldc,
        beta_is_zero,
        BLOCK_SIZE=pad_block,
    )


def bfgemm(
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
    assert A.dtype == torch.bfloat16
    assert B.dtype == torch.bfloat16
    assert C.dtype == torch.bfloat16
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

    beta_is_zero = beta == 0.0
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )

    with torch_device_fn.device(A.device):
        if (
            transa == CUBLAS_OP_N
            and transb == CUBLAS_OP_N
            and _can_use_thead_bfgemm_nn(m, n, k, lda, ldb, ldc, alpha, beta)
        ):
            aligned = _is_gemm_aligned(A, lda, B, ldb, C, ldc)
            if _thead_bfgemm_nn_should_pad(m, n, k):
                _run_thead_bfgemm_nn_padded(
                    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                )
            else:
                _run_thead_bfgemm_nn(
                    A,
                    lda,
                    B,
                    ldb,
                    C,
                    ldc,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    beta_is_zero,
                    aligned,
                )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            _bfgemm_nn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            _bfgemm_tn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            _bfgemm_nt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        else:
            _bfgemm_tt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )


bgemm = bfgemm

__all__ = ["bfgemm", "bgemm"]
