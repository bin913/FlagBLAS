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

from flag_blas.ops.level3.hgemm import (
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
    _hgemm_nn_kernel,
    _hgemm_nt_kernel,
    _hgemm_tn_kernel,
    _hgemm_tt_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.backend._thead.ops.sgemm import _is_gemm_aligned
from flag_blas.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _thead_hgemm_nn_impl(
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
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

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
        tl.store(c_ptrs, result.to(tl.float16))
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if not BETA_IS_ZERO:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            result += beta * c_vals
        tl.store(c_ptrs, result.to(tl.float16), mask=c_mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_hgemm_nn_kernel(
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
    _thead_hgemm_nn_impl(
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
def _thead_hgemm_pad2d_kernel(
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
    src_ptr = src_ptr.to(tl.pointer_type(tl.float16))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float16))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dst_rows * dst_cols
    r = offsets // dst_cols
    c = offsets - r * dst_cols
    in_bounds = (r < rows) & (c < cols)
    vals = tl.load(src_ptr + r * src_ld + c, mask=mask & in_bounds, other=0.0)
    tl.store(dst_ptr + r * dst_ld + c, vals, mask=mask)


@libentry()
@triton.jit
def _thead_hgemm_crop_c_kernel(
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
    src_ptr = src_ptr.to(tl.pointer_type(tl.float16))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float16))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < rows * cols
    r = offsets // cols
    c = offsets - r * cols
    vals = tl.load(src_ptr + r * src_ld + c, mask=mask, other=0.0).to(tl.float32)
    dst_offsets = r * dst_ld + c
    if not BETA_IS_ZERO:
        dst_vals = tl.load(dst_ptr + dst_offsets, mask=mask, other=0.0).to(tl.float32)
        vals += beta * dst_vals
    tl.store(dst_ptr + dst_offsets, vals.to(tl.float16), mask=mask)


@libentry()
@triton.jit
def _thead_hgemm_zero_f32_kernel(
    ptr,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    ptr = ptr.to(tl.pointer_type(tl.float32))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    tl.store(ptr + offsets, tl.zeros((BLOCK_SIZE,), dtype=tl.float32), mask=mask)


@libentry()
@triton.jit
def _thead_hgemm_f32_to_h_kernel(
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
    src_ptr = src_ptr.to(tl.pointer_type(tl.float32))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float16))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < rows * cols
    r = offsets // cols
    c = offsets - r * cols
    vals = tl.load(src_ptr + r * src_ld + c, mask=mask, other=0.0)
    dst_offsets = r * dst_ld + c
    if not BETA_IS_ZERO:
        dst_vals = tl.load(dst_ptr + dst_offsets, mask=mask, other=0.0).to(tl.float32)
        vals += beta * dst_vals
    tl.store(dst_ptr + dst_offsets, vals.to(tl.float16), mask=mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_hgemm_nn_splitk_kernel(
    a_ptr,
    b_ptr,
    tmp_ptr,
    alpha: tl.float32,
    lda,
    ldb,
    tmp_ld,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    tmp_ptr = tmp_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    chunk_k = tl.cdiv(K, SPLIT_K)
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, K)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * lda + (k_begin + offs_k)[None, :]
    b_ptrs = b_ptr + (k_begin + offs_k)[:, None] * ldb + offs_n[None, :]

    mask_m = offs_m < M
    mask_n = offs_n < N
    k_remain = k_end - k_begin
    full_iters = k_remain // BLOCK_K
    remainder = k_remain % BLOCK_K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, full_iters):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb

    if remainder > 0:
        mask_k = offs_k < remainder
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    tmp_ptrs = tmp_ptr + offs_m[:, None] * tmp_ld + offs_n[None, :]
    tmp_mask = mask_m[:, None] & mask_n[None, :]
    tl.atomic_add(tmp_ptrs, alpha * acc, mask=tmp_mask, sem="relaxed")


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_hgemm_nn_bwd_kernel(
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
    _thead_hgemm_nn_impl(
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
@triton.jit(ppu_hint="fwd")
def _thead_hgemm_nn_trans_kernel(
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
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

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
    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc_t = tl.dot(tl.trans(b), tl.trans(a), acc_t, out_dtype=tl.float32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb
        offs_k += BLOCK_K

    acc = tl.trans(acc_t)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        result += beta * c_vals
    tl.store(c_ptrs, result.to(tl.float16), mask=c_mask)


@libentry()
@triton.jit(ppu_hint="fwd")
def _thead_hgemm_nn_blockptr_kernel(
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
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(lda, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(ldb, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    result = alpha * acc
    if not BETA_IS_ZERO:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result += beta * c_vals
    tl.store(c_block_ptr, result.to(tl.float16), boundary_check=(0, 1))


@libentry()
@triton.jit
def _thead_hgemm_nn_desc_kernel(
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
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

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
    c_desc.store([offs_m, offs_n], result.to(tl.float16))


@libentry()
@triton.jit(ppu_hint="bwd")
def _thead_hgemm_nn_desc_bwd_kernel(
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
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

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
    c_desc.store([offs_m, offs_n], result.to(tl.float16))


def _thead_hgemm_nn_config(m: int, n: int, k: int):
    min_mn = min(m, n)

    if max(m, n, k) <= 512 and (m % 64 != 0 or n % 64 != 0 or k % 64 != 0):
        return 64, 64, 64, 4, 3, 128

    if max(m, n, k) <= 256:
        return 64, 64, 64, 4, 3, 128

    if _thead_hgemm_nn_use_desc_bwd_narrow(m, n, k):
        return 128, 256, 64, 8, 3, 160

    if _thead_hgemm_nn_use_desc_bwd(m, n, k):
        return 128, 256, 64, 8, 3, 160

    if _thead_hgemm_nn_use_large_bwd(m, n, k):
        return 128, 128, 64, 4, 3, 128

    if min_mn <= 64:
        return 64, 64, 128, 4, 3, 128

    if min_mn == 128 and max(m, n) >= 4096 and k <= 1024:
        return 128, 128, 64, 8, 3, 128

    if min_mn == 256 and max(m, n) >= 4096:
        if m <= n:
            return 128, 64, 64, 4, 3, 128
        return 128, 128, 128, 8, 3, 128

    if min_mn >= 2048:
        return 128, 128, 128, 8, 3, 128

    if min_mn >= 512:
        if max(m, n) >= 4096:
            return 128, 128, 128, 8, 3, 128
        return 128, 64, 64, 4, 4, 128

    return 128, 128, 64, 8, 3, 128


def _thead_hgemm_nn_use_large_bwd(m: int, n: int, k: int) -> bool:
    return k <= 8192 and max(m, n) >= 8192


def _thead_hgemm_nn_use_bwd(m: int, n: int, k: int) -> bool:
    return _thead_hgemm_nn_use_large_bwd(m, n, k) and not _thead_hgemm_nn_use_desc_bwd_narrow(
        m, n, k
    )


def _thead_hgemm_nn_use_desc_bwd_narrow(m: int, n: int, k: int) -> bool:
    return 256 <= min(m, n) < 1024 and max(m, n) >= 8192 and k <= 8192


def _thead_hgemm_nn_use_desc_bwd(m: int, n: int, k: int) -> bool:
    return _thead_hgemm_nn_use_desc_bwd_narrow(m, n, k) or min(m, n) >= 1024 or (
        min(m, n) >= 512 and 512 < max(m, n, k) <= 1024
    )


def _thead_hgemm_nn_use_trans(m: int, n: int, k: int) -> bool:
    return False


def _thead_hgemm_nn_use_blockptr(m: int, n: int, k: int) -> bool:
    return False


def _thead_hgemm_nn_use_desc(m: int, n: int, k: int, aligned: bool) -> bool:
    return not _thead_hgemm_nn_use_bwd(m, n, k) and not _thead_hgemm_nn_use_desc_bwd(
        m, n, k
    )


def _round_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


def _thead_hgemm_nn_should_pad(m: int, n: int, k: int) -> bool:
    if m % 64 == 0 and n % 64 == 0 and k % 64 == 0:
        return False
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    extra = m_pad * k_pad + k_pad * n_pad + m_pad * n_pad
    original = m * k + k * n + m * n
    return m * n * k >= 2048 * 2048 * 2048 and extra <= original * 1.08


def _thead_hgemm_nn_use_splitk(m: int, n: int, k: int) -> bool:
    return False


def _thead_hgemm_nn_splitk_config(m: int, n: int, k: int):
    if min(m, n) <= 64:
        if m <= n:
            return 64, 64, 64, 4, 3, min(triton.cdiv(k, 256), 8)
        return 64, 64, 64, 4, 3, min(triton.cdiv(k, 256), 8)
    if min(m, n) <= 128:
        return 128, 128, 64, 8, 3, min(triton.cdiv(k, 256), 8)
    return 128, 128, 64, 8, 3, min(triton.cdiv(k, 512), 4)


def _can_use_thead_hgemm_nn(
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


def _run_thead_hgemm_nn(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, aligned
):
    block_m, block_n, block_k, num_warps, num_stages, maxnreg = _thead_hgemm_nn_config(
        m, n, k
    )
    kernel = _thead_hgemm_nn_kernel
    if _thead_hgemm_nn_use_desc_bwd(m, n, k):
        kernel = _thead_hgemm_nn_desc_bwd_kernel
    elif _thead_hgemm_nn_use_bwd(m, n, k):
        kernel = _thead_hgemm_nn_bwd_kernel
    elif _thead_hgemm_nn_use_desc(m, n, k, aligned):
        kernel = _thead_hgemm_nn_desc_kernel
    elif _thead_hgemm_nn_use_trans(m, n, k):
        kernel = _thead_hgemm_nn_trans_kernel
    elif _thead_hgemm_nn_use_blockptr(m, n, k):
        kernel = _thead_hgemm_nn_blockptr_kernel
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


def _run_thead_hgemm_nn_padded(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    m_pad = _round_up(m, 64)
    n_pad = _round_up(n, 64)
    k_pad = _round_up(k, 64)
    A_pad = torch.empty((m_pad, k_pad), dtype=A.dtype, device=A.device)
    B_pad = torch.empty((k_pad, n_pad), dtype=B.dtype, device=B.device)
    C_pad = torch.empty((m_pad, n_pad), dtype=C.dtype, device=C.device)

    pad_block = 1024
    _thead_hgemm_pad2d_kernel[(triton.cdiv(m_pad * k_pad, pad_block),)](
        A, A_pad, m, k, lda, k_pad, m_pad, k_pad, BLOCK_SIZE=pad_block
    )
    _thead_hgemm_pad2d_kernel[(triton.cdiv(k_pad * n_pad, pad_block),)](
        B, B_pad, k, n, ldb, n_pad, k_pad, n_pad, BLOCK_SIZE=pad_block
    )
    _run_thead_hgemm_nn(
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
    _thead_hgemm_crop_c_kernel[(triton.cdiv(m * n, pad_block),)](
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


def _run_thead_hgemm_nn_splitk(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    block_m, block_n, block_k, num_warps, num_stages, split_k = (
        _thead_hgemm_nn_splitk_config(m, n, k)
    )
    tmp = torch.empty((m, n), dtype=torch.float32, device=C.device)
    block = 1024
    _thead_hgemm_zero_f32_kernel[(triton.cdiv(m * n, block),)](
        tmp, m * n, BLOCK_SIZE=block
    )
    _thead_hgemm_nn_splitk_kernel[
        (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), split_k)
    ](
        A,
        B,
        tmp,
        alpha,
        lda,
        ldb,
        n,
        M=m,
        N=n,
        K=k,
        SPLIT_K=split_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _thead_hgemm_f32_to_h_kernel[(triton.cdiv(m * n, block),)](
        tmp, C, beta, m, n, n, ldc, beta_is_zero, BLOCK_SIZE=block
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

    beta_is_zero = beta == 0.0
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )

    aligned = _is_gemm_aligned(A, lda, B, ldb, C, ldc)
    with torch_device_fn.device(A.device):
        if (
            transa == CUBLAS_OP_N
            and transb == CUBLAS_OP_N
            and _can_use_thead_hgemm_nn(m, n, k, lda, ldb, ldc, alpha, beta)
        ):
            if _thead_hgemm_nn_should_pad(m, n, k):
                _run_thead_hgemm_nn_padded(
                    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                )
            elif _thead_hgemm_nn_use_splitk(m, n, k):
                _run_thead_hgemm_nn_splitk(
                    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                )
            else:
                _run_thead_hgemm_nn(
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
            _hgemm_nn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            _hgemm_tn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            _hgemm_nt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        else:
            _hgemm_tt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )


__all__ = ["hgemm"]
