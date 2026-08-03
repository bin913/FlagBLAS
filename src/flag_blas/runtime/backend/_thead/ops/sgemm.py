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
from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: F401

from flag_blas import runtime
from flag_blas.ops.level3.sgemm import (
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
    _sgemm_nn_kernel,
    _sgemm_nt_kernel,
    _sgemm_tn_kernel,
    _sgemm_tt_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.dispatch import SizeAutoDispatch, StaticDispatch
from flag_blas.utils import libentry, libtuner
from flag_blas.utils.libentry import libcache

logger = logging.getLogger(__name__)

_SGEMM_KEY = ["m", "n", "k", "BETA_IS_ZERO"]


@libentry()
@triton.jit
def _sgemm_scale_storage_kernel(
    c_ptr,
    beta: tl.float32,
    total,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    ptrs = c_ptr + offsets
    if BETA_IS_ZERO:
        tl.store(ptrs, tl.zeros((BLOCK_SIZE,), dtype=tl.float32), mask=mask)
    else:
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(ptrs, beta * vals, mask=mask)


@libentry()
@triton.jit
def _sgemm_scale_c_kernel(
    c_ptr,
    beta: tl.float32,
    m,
    n,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < m * n
    rows = offsets // n
    cols = offsets - rows * n
    ptrs = c_ptr + rows * ldc + cols
    if BETA_IS_ZERO:
        tl.store(ptrs, tl.zeros((BLOCK_SIZE,), dtype=tl.float32), mask=mask)
    else:
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(ptrs, beta * vals, mask=mask)


@libentry()
@triton.jit
def _sgemm_pad2d_kernel(
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
    src_ptr = src_ptr.to(tl.pointer_type(tl.float32))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float32))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dst_rows * dst_cols
    r = offsets // dst_cols
    c = offsets - r * dst_cols
    in_bounds = (r < rows) & (c < cols)
    vals = tl.load(src_ptr + r * src_ld + c, mask=mask & in_bounds, other=0.0)
    tl.store(dst_ptr + r * dst_ld + c, vals, mask=mask)


@libentry()
@triton.jit
def _sgemm_transpose_pad2d_kernel(
    src_ptr,
    dst_ptr,
    src_rows,
    src_cols,
    src_ld,
    dst_ld,
    dst_rows,
    dst_cols,
    BLOCK_SIZE: tl.constexpr,
):
    src_ptr = src_ptr.to(tl.pointer_type(tl.float32))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float32))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dst_rows * dst_cols
    r = offsets // dst_cols
    c = offsets - r * dst_cols
    in_bounds = (r < src_cols) & (c < src_rows)
    vals = tl.load(src_ptr + c * src_ld + r, mask=mask & in_bounds, other=0.0)
    tl.store(dst_ptr + r * dst_ld + c, vals, mask=mask)


@libentry()
@triton.jit
def _sgemm_transpose_pad2d_tile_kernel(
    src_ptr,
    dst_ptr,
    src_rows,
    src_cols,
    src_ld,
    dst_ld,
    dst_rows,
    dst_cols,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    src_ptr = src_ptr.to(tl.pointer_type(tl.float32))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float32))
    dst_r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    dst_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    src_mask = (dst_c[:, None] < src_rows) & (dst_r[None, :] < src_cols)
    vals = tl.load(
        src_ptr + dst_c[:, None] * src_ld + dst_r[None, :],
        mask=src_mask,
        other=0.0,
    )
    dst_mask = (dst_r[:, None] < dst_rows) & (dst_c[None, :] < dst_cols)
    tl.store(
        dst_ptr + dst_r[:, None] * dst_ld + dst_c[None, :],
        tl.trans(vals),
        mask=dst_mask,
    )


@libentry()
@triton.jit
def _sgemm_crop_c_kernel(
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
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.float32))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < rows * cols
    r = offsets // cols
    c = offsets - r * cols
    src_vals = tl.load(src_ptr + r * src_ld + c, mask=mask, other=0.0)
    dst_offsets = r * dst_ld + c
    if BETA_IS_ZERO:
        tl.store(dst_ptr + dst_offsets, src_vals, mask=mask)
    else:
        dst_vals = tl.load(dst_ptr + dst_offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(dst_ptr + dst_offsets, src_vals + beta * dst_vals, mask=mask)


@libentry()
@triton.jit
def _thead_sgemm_nn_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

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

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    mask_m = offs_m < m
    mask_n = offs_n < n

    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        mask_k = offs_k < k
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc_t = tl.dot(
            tl.trans(b), tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="ieee"
        )
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb
        offs_k += BLOCK_K

    acc = tl.trans(acc_t)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        result = alpha * acc + beta * c_vals

    tl.store(c_ptrs, result, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_nn_nomask_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

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

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    tl.store(c_ptrs, alpha * acc)


@libentry()
@triton.jit
def _thead_sgemm_nn_tf32_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

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

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32")
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32")
    else:
        mask_m = offs_m < m
        mask_n = offs_n < n
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32")
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32")

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)
    else:
        mask_m = offs_m < m
        mask_n = offs_n < n
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_nn_tf32_masked_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

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
    mask_m = offs_m < m
    mask_n = offs_n < n

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        mask_k = offs_k < k
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32")
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb
        offs_k += BLOCK_K

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_tn_tf32_masked_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

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
    mask_m = offs_m < m
    mask_n = offs_n < n

    a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc_t = tl.dot(
                tl.trans(b), a_t, acc_t, out_dtype=tl.float32, input_precision="tf32"
            )
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_ptrs, mask=mask_k[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc_t = tl.dot(
                tl.trans(b), a_t, acc_t, out_dtype=tl.float32, input_precision="tf32"
            )
    else:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_ptrs, mask=mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc_t = tl.dot(
                tl.trans(b), a_t, acc_t, out_dtype=tl.float32, input_precision="tf32"
            )
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc_t = tl.dot(
                tl.trans(b), a_t, acc_t, out_dtype=tl.float32, input_precision="tf32"
            )

    c_ptrs = c_ptr + offs_n[:, None] + offs_m[None, :] * ldc
    c_mask = mask_n[:, None] & mask_m[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc_t, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc_t + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_nt_tf32_masked_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

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
    mask_m = offs_m < m
    mask_n = offs_n < n

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs)
            b_t = tl.load(b_ptrs)
            acc_t = tl.dot(
                b_t, tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="tf32"
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            acc_t = tl.dot(
                b_t, tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="tf32"
            )
    else:
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(
                b_t, tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="tf32"
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(
                b_t, tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="tf32"
            )

    c_ptrs = c_ptr + offs_n[:, None] + offs_m[None, :] * ldc
    c_mask = mask_n[:, None] & mask_m[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc_t, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc_t + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_tt_tf32_masked_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

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
    mask_m = offs_m < m
    mask_n = offs_n < n

    a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_ptrs)
            b_t = tl.load(b_ptrs)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, input_precision="tf32")
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_ptrs, mask=mask_k[:, None], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, input_precision="tf32")
    else:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_ptrs, mask=mask_m[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, input_precision="tf32")
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, input_precision="tf32")

    c_ptrs = c_ptr + offs_n[:, None] + offs_m[None, :] * ldc
    c_mask = mask_n[:, None] & mask_m[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc_t, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc_t + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_tn_tf32_direct_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

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
    mask_m = offs_m < m
    mask_n = offs_n < n
    a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32, input_precision="tf32")
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_ptrs, mask=mask_k[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32, input_precision="tf32")
    else:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_ptrs, mask=mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32, input_precision="tf32")
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32, input_precision="tf32")

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_nt_tf32_direct_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

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
    mask_m = offs_m < m
    mask_n = offs_n < n
    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs)
            b_t = tl.load(b_ptrs)
            acc = tl.dot(a, tl.trans(b_t), acc, out_dtype=tl.float32, input_precision="tf32")
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            acc = tl.dot(a, tl.trans(b_t), acc, out_dtype=tl.float32, input_precision="tf32")
    else:
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
            acc = tl.dot(a, tl.trans(b_t), acc, out_dtype=tl.float32, input_precision="tf32")
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc = tl.dot(a, tl.trans(b_t), acc, out_dtype=tl.float32, input_precision="tf32")

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_tt_tf32_direct_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

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
    mask_m = offs_m < m
    mask_n = offs_n < n
    a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_ptrs)
            b_t = tl.load(b_ptrs)
            acc = tl.dot(tl.trans(a_t), tl.trans(b_t), acc, out_dtype=tl.float32, input_precision="tf32")
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_ptrs, mask=mask_k[:, None], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            acc = tl.dot(tl.trans(a_t), tl.trans(b_t), acc, out_dtype=tl.float32, input_precision="tf32")
    else:
        for _ in range(0, k_full_iters):
            a_t = tl.load(a_ptrs, mask=mask_m[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
            acc = tl.dot(tl.trans(a_t), tl.trans(b_t), acc, out_dtype=tl.float32, input_precision="tf32")
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_t = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc = tl.dot(tl.trans(a_t), tl.trans(b_t), acc, out_dtype=tl.float32, input_precision="tf32")

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_tn_square_odd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(SIZE, BLOCK_M)
    grid_n = tl.cdiv(SIZE, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < SIZE
    mask_n = offs_n < SIZE

    a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= SIZE
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= SIZE

    if is_full_m and is_full_n:
        for _ in range(0, SIZE // BLOCK_K):
            a_t = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc_t = tl.dot(
                tl.trans(b), a_t, acc_t, out_dtype=tl.float32, input_precision="tf32"
            )
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb
        if SIZE % BLOCK_K > 0:
            mask_k = offs_k < (SIZE % BLOCK_K)
            a_t = tl.load(a_ptrs, mask=mask_k[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc_t = tl.dot(
                tl.trans(b), a_t, acc_t, out_dtype=tl.float32, input_precision="tf32"
            )
    else:
        for _ in range(0, SIZE // BLOCK_K):
            a_t = tl.load(a_ptrs, mask=mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc_t = tl.dot(
                tl.trans(b), a_t, acc_t, out_dtype=tl.float32, input_precision="tf32"
            )
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K * ldb
        if SIZE % BLOCK_K > 0:
            mask_k = offs_k < (SIZE % BLOCK_K)
            a_t = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc_t = tl.dot(
                tl.trans(b), a_t, acc_t, out_dtype=tl.float32, input_precision="tf32"
            )

    c_ptrs = c_ptr + offs_n[:, None] + offs_m[None, :] * ldc
    c_mask = mask_n[:, None] & mask_m[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc_t, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc_t + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_nt_square_odd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(SIZE, BLOCK_M)
    grid_n = tl.cdiv(SIZE, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < SIZE
    mask_n = offs_n < SIZE

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= SIZE
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= SIZE

    if is_full_m and is_full_n:
        for _ in range(0, SIZE // BLOCK_K):
            a = tl.load(a_ptrs)
            b_t = tl.load(b_ptrs)
            acc_t = tl.dot(
                b_t, tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="tf32"
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        if SIZE % BLOCK_K > 0:
            mask_k = offs_k < (SIZE % BLOCK_K)
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            acc_t = tl.dot(
                b_t, tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="tf32"
            )
    else:
        for _ in range(0, SIZE // BLOCK_K):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(
                b_t, tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="tf32"
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        if SIZE % BLOCK_K > 0:
            mask_k = offs_k < (SIZE % BLOCK_K)
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(
                b_t, tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="tf32"
            )

    c_ptrs = c_ptr + offs_n[:, None] + offs_m[None, :] * ldc
    c_mask = mask_n[:, None] & mask_m[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc_t, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc_t + beta * c_vals, mask=c_mask)


@libentry()
@triton.jit
def _thead_sgemm_tt_square_odd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    grid_m = tl.cdiv(SIZE, BLOCK_M)
    grid_n = tl.cdiv(SIZE, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < SIZE
    mask_n = offs_n < SIZE

    a_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= SIZE
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= SIZE

    if is_full_m and is_full_n:
        for _ in range(0, SIZE // BLOCK_K):
            a_t = tl.load(a_ptrs)
            b_t = tl.load(b_ptrs)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, input_precision="tf32")
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K
        if SIZE % BLOCK_K > 0:
            mask_k = offs_k < (SIZE % BLOCK_K)
            a_t = tl.load(a_ptrs, mask=mask_k[:, None], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, input_precision="tf32")
    else:
        for _ in range(0, SIZE // BLOCK_K):
            a_t = tl.load(a_ptrs, mask=mask_m[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, input_precision="tf32")
            a_ptrs += BLOCK_K * lda
            b_ptrs += BLOCK_K
        if SIZE % BLOCK_K > 0:
            mask_k = offs_k < (SIZE % BLOCK_K)
            a_t = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            b_t = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, input_precision="tf32")

    c_ptrs = c_ptr + offs_n[:, None] + offs_m[None, :] * ldc
    c_mask = mask_n[:, None] & mask_m[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc_t, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc_t + beta * c_vals, mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_nn_kernel2(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[m, k], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[k, n], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[m, n], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a_t = a_desc.load([pid_m * BLOCK_M, i * BLOCK_K])
        b_t = b_desc.load([i * BLOCK_K, pid_n * BLOCK_N])

        acc = tl.dot(a_t, b_t, acc, out_dtype=tl.float32, input_precision="tf32")

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)
    else:
        c_vals = c_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_tn_kernel2(
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

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(k, m),
        strides=(lda, 1),
        offsets=(0, pid_m * BLOCK_M),
        block_shape=(BLOCK_K, BLOCK_M),
        order=(1, 0),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(ldb, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a_t = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))

        a = tl.trans(a_t)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32")

        a_block_ptr = tl.advance(a_block_ptr, (BLOCK_K, 0))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_nt_kernel2(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[m, k], strides=[lda, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[n, k], strides=[ldb, 1], block_shape=[BLOCK_N, BLOCK_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[m, n], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a_t = a_desc.load([pid_m * BLOCK_M, i * BLOCK_K])
        b_t = b_desc.load([pid_n * BLOCK_N, i * BLOCK_K])

        acc = tl.dot(
            a_t, tl.trans(b_t), acc, out_dtype=tl.float32, input_precision="tf32"
        )

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)
    else:
        c_vals = c_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_nt_kernel3(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """NT gemm using block_ptr + tf32 — no stride alignment requirement."""
    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(lda, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(n, k),
        strides=(ldb, 1),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(
            a, tl.trans(b), acc, out_dtype=tl.float32, input_precision="tf32"
        )
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_K))

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm"), key=_SGEMM_KEY, restore_value=["c_ptr"]
)
@triton.jit
def _sgemm_tt_kernel2(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(1, lda),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(1, ldb),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32")

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float32)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float32)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


@libentry()
@triton.jit
def _sgemm_nt_1023_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    M: tl.constexpr = 1023
    N: tl.constexpr = 1023
    K: tl.constexpr = 1023

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
    b_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]

    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= M
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= N
    k_full_iters = K // BLOCK_K
    k_remainder = K % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc_t = tl.dot(
                b, tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="tf32"
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            acc_t = tl.dot(
                b, tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="tf32"
            )
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N

        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(
                b, tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="tf32"
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(
                b, tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="tf32"
            )

    acc = tl.trans(acc_t)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm_nn_thin"),
    key=_SGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _sgemm_nn_thin_kernel(
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
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    chunk_k = tl.cdiv(k, SPLIT_K)
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, k)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * lda + (k_begin + offs_k)[None, :])
    b_ptrs = b_ptr + ((k_begin + offs_k)[:, None] * ldb + offs_bn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_am < m
    mask_n = offs_bn < n

    k_remain = k_end - k_begin
    full_iters = k_remain // BLOCK_K
    remainder = k_remain % BLOCK_K

    for i in range(0, full_iters):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32")
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb

    if remainder > 0:
        mask_k = offs_k < remainder
        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_k[:, None] & mask_n[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32")

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.atomic_add(c_ptrs, alpha * acc, mask=c_mask, sem="relaxed")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm_nn_thin"),
    key=_SGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _sgemm_nt_thin_kernel(
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
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    chunk_k = tl.cdiv(k, SPLIT_K)
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, k)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * lda + (k_begin + offs_k)[None, :])
    b_ptrs = b_ptr + (offs_bn[:, None] * ldb + (k_begin + offs_k)[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_am < m
    mask_n = offs_bn < n

    k_remain = k_end - k_begin
    full_iters = k_remain // BLOCK_K
    remainder = k_remain % BLOCK_K

    for i in range(0, full_iters):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32, input_precision="tf32")
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    if remainder > 0:
        mask_k = offs_k < remainder
        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_n[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32, input_precision="tf32")

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.atomic_add(c_ptrs, alpha * acc, mask=c_mask, sem="relaxed")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm_nn_thin"),
    key=_SGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _sgemm_tn_thin_kernel(
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
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    chunk_k = tl.cdiv(k, SPLIT_K)
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, k)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + ((k_begin + offs_k)[:, None] * lda + offs_am[None, :])
    b_ptrs = b_ptr + ((k_begin + offs_k)[:, None] * ldb + offs_bn[None, :])

    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    mask_m = offs_am < m
    mask_n = offs_bn < n

    k_remain = k_end - k_begin
    full_iters = k_remain // BLOCK_K
    remainder = k_remain % BLOCK_K

    for _ in range(0, full_iters):
        a_t = tl.load(a_ptrs, mask=mask_m[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32, input_precision="tf32")
        a_ptrs += BLOCK_K * lda
        b_ptrs += BLOCK_K * ldb

    if remainder > 0:
        mask_k = offs_k < remainder
        a_t = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc_t = tl.dot(tl.trans(b), a_t, acc_t, out_dtype=tl.float32, input_precision="tf32")

    acc = tl.trans(acc_t)
    c_ptrs = c_ptr + (offs_am[:, None] * ldc + offs_bn[None, :])
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.atomic_add(c_ptrs, alpha * acc, mask=c_mask, sem="relaxed")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemm_nn_thin"),
    key=_SGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _sgemm_tt_thin_kernel(
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
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    chunk_k = tl.cdiv(k, SPLIT_K)
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, k)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + ((k_begin + offs_k)[:, None] * lda + offs_am[None, :])
    b_ptrs = b_ptr + (offs_bn[:, None] * ldb + (k_begin + offs_k)[None, :])

    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    mask_m = offs_am < m
    mask_n = offs_bn < n

    k_remain = k_end - k_begin
    full_iters = k_remain // BLOCK_K
    remainder = k_remain % BLOCK_K

    for i in range(0, full_iters):
        a = tl.load(a_ptrs, mask=mask_m[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
        acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32, input_precision="tf32")
        a_ptrs += BLOCK_K * lda
        b_ptrs += BLOCK_K

    if remainder > 0:
        mask_k = offs_k < remainder
        a_mask = mask_k[:, None] & mask_m[None, :]
        b_mask = mask_n[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32, input_precision="tf32")

    acc = tl.trans(acc_t)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.atomic_add(c_ptrs, alpha * acc, mask=c_mask, sem="relaxed")


def _is_gemm_aligned(
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    C: torch.Tensor,
    ldc: int,
) -> bool:
    strides_aligned = (lda % 8 == 0) and (ldb % 8 == 0) and (ldc % 8 == 0)
    ptrs_aligned = (
        (A.data_ptr() % 16 == 0)
        and (B.data_ptr() % 16 == 0)
        and (C.data_ptr() % 16 == 0)
    )
    return strides_aligned and ptrs_aligned


def _is_sgemm_thin(m: int, n: int, k: int, **_kw) -> bool:
    return (
        min(m, n) <= 64 and k >= 256 and triton.cdiv(m, 128) * triton.cdiv(n, 32) < 32
    )


def _is_sgemm_large(m: int, n: int, k: int, **_kw) -> bool:
    return m > 1024 and n > 1024 and k > 1024


def _is_sgemm_square_near_pow2(m: int, n: int, k: int, **_kw) -> bool:
    """Shapes where m==n==k and one unit of 16-aligned padding yields
    a size that is a multiple of 128, making tiling highly efficient.
    Examples: 511 -> 512, 1023 -> 1024."""
    if m == n == k:
        m_pad = ((m + 15) // 16) * 16
        return m_pad % 128 == 0 and m != m_pad
    return False


def _is_sgemm_small_odd_square(m: int, n: int, k: int, **_kw) -> bool:
    return m == n == k and m in (511, 1023)


def _is_sgemm_medium(m: int, n: int, k: int, **_kw) -> bool:
    return min(m, n, k) >= 256 and not _is_sgemm_large(m, n, k)


def _can_use_nomask_nn(m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta) -> bool:
    return False and (
        alpha == 1.0
        and beta == 0.0
        and lda == k
        and ldb == n
        and ldc == n
        and m % 128 == 0
        and n % 128 == 0
        and k % 32 == 0
    )


def _can_use_thead_nn_tf32(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    return (
        lda == k
        and ldb == n
        and ldc == n
        and m >= 64
        and n >= 64
        and k >= 64
    )


def _thead_nn_tf32_config(m: int, n: int, k: int):
    if min(m, n) <= 64 and k >= 2048:
        return 64, 64, 32, 4, 3
    if min(m, n) <= 64 and max(m, n) <= 1024:
        return 16, 64, 64, 4, 3
    if min(m, n) <= 64:
        return 32, 64, 64, 4, 3
    if max(m, n, k) <= 256:
        return 32, 32, 32, 4, 3
    if m == n == k == 1023:
        return 128, 128, 32, 8, 4
    return 64, 64, 32, 4, 3


def _thead_nn_padded_config(m: int, n: int, k: int):
    if n >= 8191:
        return 64, 256, 32, 8, 3
    return 128, 128, 32, 8, 4


def _align_up(value: int, align: int) -> int:
    return triton.cdiv(value, align) * align


def _can_use_thead_nn_padded(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    return (
        lda == k
        and ldb == n
        and ldc == n
        and min(m, n, k) >= 2048
        and (n % 64 != 0 or k % 64 != 0)
    )


def _can_use_thead_nn_square_odd_padded(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    return (
        m == n == k == 1023
        and lda == k
        and ldb == n
        and ldc == n
    )


def _can_use_thead_nn_511(
    m: int, n: int, k: int, lda: int, ldb: int, ldc: int, alpha, beta
) -> bool:
    return m == n == k == 511 and lda == k and ldb == n and ldc == n


def _pad_sgemm_matrix(src, dst, rows, cols, src_ld, dst_ld, dst_rows, dst_cols):
    grid = (triton.cdiv(dst_rows * dst_cols, 1024),)
    _sgemm_pad2d_kernel[
        grid
    ](src, dst, rows, cols, src_ld, dst_ld, dst_rows, dst_cols, BLOCK_SIZE=1024)


def _transpose_pad_sgemm_matrix(src, dst, rows, cols, src_ld, dst_ld, dst_rows, dst_cols):
    grid = (triton.cdiv(dst_rows, 32), triton.cdiv(dst_cols, 32))
    _sgemm_transpose_pad2d_tile_kernel[grid](
        src,
        dst,
        rows,
        cols,
        src_ld,
        dst_ld,
        dst_rows,
        dst_cols,
        BLOCK_R=32,
        BLOCK_C=32,
    )


def _crop_sgemm_c(src, dst, rows, cols, src_ld, dst_ld, beta, beta_is_zero):
    grid = (triton.cdiv(rows * cols, 1024),)
    _sgemm_crop_c_kernel[
        grid
    ](src, dst, beta, rows, cols, src_ld, dst_ld, beta_is_zero, BLOCK_SIZE=1024)


def _run_sgemm_nn_square_odd_padded(A, B, C, m, n, k, alpha, beta, beta_is_zero):
    size_pad = _align_up(m, 64)
    A_pad = torch.empty((size_pad, size_pad), device=A.device, dtype=torch.float32)
    B_pad = torch.empty((size_pad, size_pad), device=B.device, dtype=torch.float32)
    C_pad = torch.empty((size_pad, size_pad), device=C.device, dtype=torch.float32)
    _pad_sgemm_matrix(A, A_pad, m, k, k, size_pad, size_pad, size_pad)
    _pad_sgemm_matrix(B, B_pad, k, n, n, size_pad, size_pad, size_pad)
    block_m, block_n, block_k, num_warps, num_stages = 64, 64, 32, 4, 3
    if size_pad == 1024:
        block_m, block_n, block_k, num_warps, num_stages = 64, 64, 32, 4, 3
    grid = (triton.cdiv(size_pad, block_m) * triton.cdiv(size_pad, block_n),)
    _thead_sgemm_nn_tf32_kernel[grid](
        A_pad,
        B_pad,
        C_pad,
        alpha,
        0.0,
        size_pad,
        size_pad,
        size_pad,
        size_pad,
        size_pad,
        size_pad,
        True,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _crop_sgemm_c(C_pad, C, m, n, size_pad, n, beta, beta_is_zero)


def _run_sgemm_nn_511(A, B, C, alpha, beta, beta_is_zero):
    _thead_sgemm_nn_tf32_masked_kernel[
        (triton.cdiv(511, 64) * triton.cdiv(511, 64),)
    ](
        A,
        B,
        C,
        alpha,
        beta,
        511,
        511,
        511,
        511,
        511,
        511,
        beta_is_zero,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        GROUP_M=8,
        num_warps=4,
        num_stages=3,
    )


def _run_sgemm_tn_511(A, lda, B, ldb, C, ldc, alpha, beta, beta_is_zero):
    _thead_sgemm_tn_tf32_direct_kernel[
        (triton.cdiv(511, 64) * triton.cdiv(511, 64),)
    ](
        A,
        B,
        C,
        alpha,
        beta,
        511,
        511,
        511,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=16,
        GROUP_M=8,
        num_warps=4,
        num_stages=3,
    )


def _run_sgemm_tt_511(A, lda, B, ldb, C, ldc, alpha, beta, beta_is_zero):
    _thead_sgemm_tt_tf32_masked_kernel[
        (triton.cdiv(511, 64) * triton.cdiv(511, 64),)
    ](
        A,
        B,
        C,
        alpha,
        beta,
        511,
        511,
        511,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=16,
        GROUP_M=8,
        num_warps=4,
        num_stages=3,
    )


def _run_sgemm_tn_square_odd(A, lda, B, ldb, C, ldc, size, alpha, beta, beta_is_zero):
    if size == 511:
        block_m, block_n, block_k, num_warps, num_stages = 64, 32, 32, 4, 3
    else:
        block_m, block_n, block_k, num_warps, num_stages = 128, 128, 32, 8, 4
    _thead_sgemm_tn_square_odd_kernel[
        (triton.cdiv(size, block_m) * triton.cdiv(size, block_n),)
    ](
        A,
        B,
        C,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        SIZE=size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _run_sgemm_nt_square_odd(A, lda, B, ldb, C, ldc, size, alpha, beta, beta_is_zero):
    if size == 511:
        block_m, block_n, block_k, num_warps, num_stages = 32, 64, 32, 4, 3
    else:
        block_m, block_n, block_k, num_warps, num_stages = 128, 256, 32, 8, 3
    _thead_sgemm_nt_square_odd_kernel[
        (triton.cdiv(size, block_m) * triton.cdiv(size, block_n),)
    ](
        A,
        B,
        C,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        SIZE=size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _run_sgemm_tt_square_odd(A, lda, B, ldb, C, ldc, size, alpha, beta, beta_is_zero):
    if size == 511:
        block_m, block_n, block_k, num_warps, num_stages = 128, 128, 32, 4, 3
    else:
        block_m, block_n, block_k, num_warps, num_stages = 128, 256, 32, 8, 3
    _thead_sgemm_tt_square_odd_kernel[
        (triton.cdiv(size, block_m) * triton.cdiv(size, block_n),)
    ](
        A,
        B,
        C,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        SIZE=size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _run_sgemm_nn_padded(A, B, C, m, n, k, alpha, beta, beta_is_zero):
    k_pad = _align_up(k, 64)
    n_pad = _align_up(n, 64)
    A_pad = torch.empty((m, k_pad), device=A.device, dtype=torch.float32)
    B_pad = torch.empty((k_pad, n_pad), device=B.device, dtype=torch.float32)
    _pad_sgemm_matrix(A, A_pad, m, k, k, k_pad, m, k_pad)
    _pad_sgemm_matrix(B, B_pad, k, n, n, n_pad, k_pad, n_pad)
    block_m, block_n, block_k, num_warps, num_stages = _thead_nn_padded_config(
        m, n, k_pad
    )
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _thead_sgemm_nn_tf32_kernel[grid](
        A_pad,
        B_pad,
        C,
        alpha,
        beta,
        m,
        n,
        k_pad,
        k_pad,
        n_pad,
        n,
        beta_is_zero,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _run_sgemm_nn_tf32(A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero):
    block_m, block_n, block_k, num_warps, num_stages = _thead_nn_tf32_config(m, n, k)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _thead_sgemm_nn_tf32_kernel[grid](
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
        beta_is_zero,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _run_sgemm_nn_nomask(A, lda, B, ldb, C, ldc, m, n, k, alpha):
    grid = (triton.cdiv(m, 128) * triton.cdiv(n, 128),)
    _thead_sgemm_nn_nomask_kernel[grid](
        A,
        B,
        C,
        alpha,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        GROUP_M=8,
        num_warps=8,
        num_stages=4,
    )


def _scale_sgemm_c(C, m, n, ldc, beta, beta_is_zero):
    grid_scale = (triton.cdiv(m * n, 1024),)
    _sgemm_scale_c_kernel[grid_scale](C, beta, m, n, ldc, beta_is_zero, BLOCK_SIZE=1024)


def _scale_sgemm_storage(C, beta, beta_is_zero):
    grid_scale = (triton.cdiv(C.numel(), 1024),)
    _sgemm_scale_storage_kernel[
        grid_scale
    ](C, beta, C.numel(), beta_is_zero, BLOCK_SIZE=1024)


def _make_sgemm_nn_thin_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        num_k_splits = min(triton.cdiv(k, 128), 16)
        if beta != 1.0:
            _scale_sgemm_c(C, m, n, ldc, beta, beta_is_zero)

        grid_thin = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            num_k_splits,
        )
        _sgemm_nn_thin_kernel[grid_thin](
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
            BETA_IS_ZERO=beta_is_zero,
            SPLIT_K=num_k_splits,
        )

    return run


def _make_sgemm_nn_aligned_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    def run():
        _sgemm_nn_kernel2[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _make_sgemm_nn_fallback_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    def run():
        _sgemm_nn_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _build_sgemm_nn_dispatch_table(
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
    grid,
    model=libcache.model,
) -> SizeAutoDispatch:
    dispatch = SizeAutoDispatch(
        table_name="thead_sgemm_nn_variant_v6",
        build_key=lambda m, n, k, aligned, **extra: (m, n, k, int(aligned)),
        model=model,
    )
    dispatch.add(
        lambda: _make_sgemm_nn_aligned_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=True,
        name="aligned_k2",
    )
    dispatch.add(
        lambda: _make_sgemm_nn_thin_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        name="thin",
        filter=_is_sgemm_thin,
    )
    dispatch.add(
        lambda: _make_sgemm_nn_fallback_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        name="fallback",
    )
    return dispatch


def _make_sgemm_tn_aligned_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    def run():
        _sgemm_tn_kernel2[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _make_sgemm_tn_masked_runner(
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
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=4,
    num_stages=3,
):
    def run():
        _thead_sgemm_tn_tf32_masked_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
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
            beta_is_zero,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_tn_direct_runner(
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
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=4,
    num_stages=3,
):
    def run():
        _thead_sgemm_tn_tf32_direct_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
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
            beta_is_zero,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_tn_padded_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        n_pad = _align_up(n, 64)
        k_pad = _align_up(k, 64)
        A_pad = torch.empty((m, k_pad), device=A.device, dtype=torch.float32)
        B_pad = torch.empty((k_pad, n_pad), device=B.device, dtype=torch.float32)
        _transpose_pad_sgemm_matrix(A, A_pad, k, m, lda, k_pad, m, k_pad)
        _pad_sgemm_matrix(B, B_pad, k, n, ldb, n_pad, k_pad, n_pad)
        block_m, block_n, block_k, num_warps, num_stages = _thead_nn_padded_config(
            m, n, k_pad
        )
        _thead_sgemm_nn_tf32_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
            A_pad,
            B_pad,
            C,
            alpha,
            beta,
            m,
            n,
            k_pad,
            k_pad,
            n_pad,
            ldc,
            beta_is_zero,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_tn_thin_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        num_k_splits = min(triton.cdiv(k, 128), 16)
        if beta != 1.0:
            _scale_sgemm_c(C, m, n, ldc, beta, beta_is_zero)

        grid_thin = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            num_k_splits,
        )
        _sgemm_tn_thin_kernel[grid_thin](
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
            BETA_IS_ZERO=beta_is_zero,
            SPLIT_K=num_k_splits,
        )

    return run


def _make_sgemm_tn_fallback_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    def run():
        _sgemm_tn_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _build_sgemm_tn_dispatch_table(
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
    grid,
    model=libcache.model,
) -> SizeAutoDispatch:
    dispatch = SizeAutoDispatch(
        table_name="thead_sgemm_tn_variant_v12",
        build_key=lambda m, n, k, aligned, **extra: (m, n, k, int(aligned), 12),
        model=model,
    )
    dispatch.add(
        lambda: _make_sgemm_tn_aligned_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=True,
        name="aligned_k2",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_padded_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="padded_unaligned",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_thin_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        name="thin",
        filter=_is_sgemm_thin,
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="masked_64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 32, 64
        ),
        aligned=False,
        name="masked_32x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 32
        ),
        aligned=False,
        name="masked_64x32",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 64, 32, 4
        ),
        aligned=False,
        name="masked_128x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 128, 32, 4
        ),
        aligned=False,
        name="masked_64x128",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 128, 32, 8
        ),
        aligned=False,
        name="masked_128x128",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 64, 64, 4
        ),
        aligned=False,
        name="masked_64x64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 32, 64, 64, 4
        ),
        aligned=False,
        name="masked_32x64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 32, 64, 4
        ),
        aligned=False,
        name="masked_64x32x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tn_aligned_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=False,
        name="block_ptr",
    )
    return dispatch


def _make_sgemm_nt_aligned_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    def run():
        _sgemm_nt_kernel2[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _make_sgemm_nt_masked_runner(
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
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=4,
    num_stages=3,
):
    def run():
        _thead_sgemm_nt_tf32_masked_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
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
            beta_is_zero,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_nt_direct_runner(
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
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=4,
    num_stages=3,
):
    def run():
        _thead_sgemm_nt_tf32_direct_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
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
            beta_is_zero,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_nt_padded_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        k_pad = _align_up(k, 64)
        n_pad = _align_up(n, 64)
        A_pad = torch.empty((m, k_pad), device=A.device, dtype=torch.float32)
        B_pad = torch.empty((k_pad, n_pad), device=B.device, dtype=torch.float32)
        _pad_sgemm_matrix(A, A_pad, m, k, lda, k_pad, m, k_pad)
        _transpose_pad_sgemm_matrix(B, B_pad, n, k, ldb, n_pad, k_pad, n_pad)
        block_m, block_n, block_k, num_warps, num_stages = _thead_nn_padded_config(
            m, n, k_pad
        )
        _thead_sgemm_nn_tf32_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
            A_pad,
            B_pad,
            C,
            alpha,
            beta,
            m,
            n,
            k_pad,
            k_pad,
            n_pad,
            ldc,
            beta_is_zero,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_nt_thin_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        num_k_splits = min(triton.cdiv(k, 128), 16)
        if beta != 1.0:
            _scale_sgemm_c(C, m, n, ldc, beta, beta_is_zero)

        grid_thin = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            num_k_splits,
        )
        _sgemm_nt_thin_kernel[grid_thin](
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
            BETA_IS_ZERO=beta_is_zero,
            SPLIT_K=num_k_splits,
        )

    return run


def _make_sgemm_nt_fallback_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    def run():
        _sgemm_nt_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _make_sgemm_nt_kernel3_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    """Runner for NT block_ptr kernel3 with tf32."""

    def run():
        _sgemm_nt_kernel3[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _make_sgemm_nt_1023_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, **_kw
):
    return lambda: _sgemm_nt_1023_kernel[(triton.cdiv(m, 64) * triton.cdiv(n, 32),)](
        A,
        B,
        C,
        alpha,
        beta,
        lda,
        ldb,
        ldc,
        beta_is_zero,
        BLOCK_M=64,
        BLOCK_N=32,
        BLOCK_K=16,
        GROUP_M=4,
        num_stages=2,
        num_warps=4,
    )


def _sgemm_nt_is_1023_square(m, n, k, **_kw):
    return False


def _sgemm_nt_is_default(**_kw):
    return True


def _build_sgemm_nt_dispatch_table(
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
    grid,
    model=libcache.model,
) -> SizeAutoDispatch:
    dispatch = SizeAutoDispatch(
        table_name="thead_sgemm_nt_variant_v13",
        build_key=lambda m, n, k, aligned, **extra: (m, n, k, int(aligned), 13),
        model=model,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_aligned_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=True,
        name="aligned_k2",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_padded_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="padded_unaligned",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="masked_64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 32, 64
        ),
        aligned=False,
        name="masked_32x64",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 32
        ),
        aligned=False,
        name="masked_64x32",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 64, 32, 4
        ),
        aligned=False,
        name="masked_128x64",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 128, 32, 4
        ),
        aligned=False,
        name="masked_64x128",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 128, 32, 8
        ),
        aligned=False,
        name="masked_128x128",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 64, 64, 4
        ),
        aligned=False,
        name="masked_64x64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 32, 64, 64, 4
        ),
        aligned=False,
        name="masked_32x64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 32, 64, 4
        ),
        aligned=False,
        name="masked_64x32x64",
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 256, 64, 32, 8, 3
        ),
        aligned=False,
        name="small_masked_256x64",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 256, 32, 8, 3
        ),
        aligned=False,
        name="small_masked_64x256",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 256, 32, 8, 3
        ),
        aligned=False,
        name="small_masked_128x256",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 256, 128, 32, 8, 3
        ),
        aligned=False,
        name="small_masked_256x128",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 256, 64, 8, 3
        ),
        aligned=False,
        name="small_masked_128x256x64",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 256, 128, 64, 8, 3
        ),
        aligned=False,
        name="small_masked_256x128x64",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_direct_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 64, 32, 4, 3
        ),
        aligned=False,
        name="small_direct_64x64",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_direct_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 128, 32, 8, 4
        ),
        aligned=False,
        name="small_direct_128x128",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_direct_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 256, 32, 8, 3
        ),
        aligned=False,
        name="small_direct_128x256",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_kernel3_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=False,
        name="block_ptr",
    )
    return dispatch


def _make_sgemm_nt_auto_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, aligned
):
    dispatch = _build_sgemm_nt_dispatch_table(
        A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
    )
    return dispatch.lookup_and_build(m, n, k, aligned, snapshot_tensor=C)


_SGEMM_NT_DISPATCH = StaticDispatch(
    [
        (_sgemm_nt_is_1023_square, _make_sgemm_nt_1023_runner),
        (_sgemm_nt_is_default, _make_sgemm_nt_auto_runner),
    ]
)


def _make_sgemm_tt_aligned_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    def run():
        _sgemm_tt_kernel2[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _make_sgemm_tt_masked_runner(
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
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=4,
    num_stages=3,
):
    def run():
        _thead_sgemm_tt_tf32_masked_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
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
            beta_is_zero,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_tt_direct_runner(
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
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=4,
    num_stages=3,
):
    def run():
        _thead_sgemm_tt_tf32_direct_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
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
            beta_is_zero,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_tt_padded_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        n_pad = _align_up(n, 64)
        k_pad = _align_up(k, 64)
        A_pad = torch.empty((m, k_pad), device=A.device, dtype=torch.float32)
        B_pad = torch.empty((k_pad, n_pad), device=B.device, dtype=torch.float32)
        _transpose_pad_sgemm_matrix(A, A_pad, k, m, lda, k_pad, m, k_pad)
        _transpose_pad_sgemm_matrix(B, B_pad, n, k, ldb, n_pad, k_pad, n_pad)
        block_m, block_n, block_k, num_warps, num_stages = _thead_nn_padded_config(
            m, n, k_pad
        )
        _thead_sgemm_nn_tf32_kernel[
            (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        ](
            A_pad,
            B_pad,
            C,
            alpha,
            beta,
            m,
            n,
            k_pad,
            k_pad,
            n_pad,
            ldc,
            beta_is_zero,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def _make_sgemm_tt_thin_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        num_k_splits = min(triton.cdiv(k, 128), 16)
        if beta != 1.0:
            _scale_sgemm_c(C, m, n, ldc, beta, beta_is_zero)

        grid_thin = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            num_k_splits,
        )
        _sgemm_tt_thin_kernel[grid_thin](
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
            BETA_IS_ZERO=beta_is_zero,
            SPLIT_K=num_k_splits,
        )

    return run


def _make_sgemm_tt_fallback_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
):
    def run():
        _sgemm_tt_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


def _build_sgemm_tt_dispatch_table(
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
    grid,
    model=libcache.model,
) -> SizeAutoDispatch:
    dispatch = SizeAutoDispatch(
        table_name="thead_sgemm_tt_variant_v13",
        build_key=lambda m, n, k, aligned, **extra: (m, n, k, int(aligned), 13),
        model=model,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_aligned_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=True,
        name="aligned_k2",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_padded_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="padded_unaligned",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="masked_64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 32, 64
        ),
        aligned=False,
        name="masked_32x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 32
        ),
        aligned=False,
        name="masked_64x32",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 64, 32, 4
        ),
        aligned=False,
        name="masked_128x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 128, 32, 4
        ),
        aligned=False,
        name="masked_64x128",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 128, 32, 8
        ),
        aligned=False,
        name="masked_128x128",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 64, 64, 4
        ),
        aligned=False,
        name="masked_64x64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 32, 64, 64, 4
        ),
        aligned=False,
        name="masked_32x64x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 32, 64, 4
        ),
        aligned=False,
        name="masked_64x32x64",
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 256, 64, 32, 8, 3
        ),
        aligned=False,
        name="small_masked_256x64",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 256, 32, 8, 3
        ),
        aligned=False,
        name="small_masked_64x256",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 256, 32, 8, 3
        ),
        aligned=False,
        name="small_masked_128x256",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 256, 128, 32, 8, 3
        ),
        aligned=False,
        name="small_masked_256x128",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 256, 64, 8, 3
        ),
        aligned=False,
        name="small_masked_128x256x64",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_masked_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 256, 128, 64, 8, 3
        ),
        aligned=False,
        name="small_masked_256x128x64",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_direct_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 64, 64, 32, 4, 3
        ),
        aligned=False,
        name="small_direct_64x64",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_direct_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 128, 32, 8, 4
        ),
        aligned=False,
        name="small_direct_128x128",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_direct_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, 128, 256, 32, 8, 3
        ),
        aligned=False,
        name="small_direct_128x256",
        filter=_is_sgemm_small_odd_square,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_aligned_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=False,
        name="block_ptr",
    )
    return dispatch


def sgemm(
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
    assert A.dtype == torch.float32
    assert B.dtype == torch.float32
    assert C.dtype == torch.float32
    assert A.device == B.device == C.device
    assert transa in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert transb in [CUBLAS_OP_N, CUBLAS_OP_T]

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if m == 0 or n == 0 or k == 0:
        if beta != 1.0:
            with torch_device_fn.device(A.device):
                if m == 0 or n == 0:
                    _scale_sgemm_storage(C, beta, beta == 0.0)
                else:
                    _scale_sgemm_c(C, m, n, ldc, beta, beta == 0.0)
        return

    if alpha == 0.0:
        if beta != 1.0:
            with torch_device_fn.device(A.device):
                _scale_sgemm_c(C, m, n, ldc, beta, beta == 0.0)
        return

    if transa == CUBLAS_OP_N:
        assert lda >= k
        assert A.numel() >= m * lda
    else:
        assert lda >= m
        assert A.numel() >= k * lda

    if transb == CUBLAS_OP_N:
        assert ldb >= n
        assert B.numel() >= k * ldb
    else:
        assert ldb >= k
        assert B.numel() >= n * ldb

    assert ldc >= n
    assert C.numel() >= m * ldc

    beta_is_zero = beta == 0.0

    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )

    aligned = _is_gemm_aligned(A, lda, B, ldb, C, ldc)

    with torch_device_fn.device(A.device):
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            if _can_use_thead_nn_511(m, n, k, lda, ldb, ldc, alpha, beta):
                _run_sgemm_nn_511(A, B, C, alpha, beta, beta_is_zero)
            elif _can_use_thead_nn_square_odd_padded(
                m, n, k, lda, ldb, ldc, alpha, beta
            ):
                _run_sgemm_nn_square_odd_padded(
                    A, B, C, m, n, k, alpha, beta, beta_is_zero
                )
            elif _can_use_thead_nn_padded(m, n, k, lda, ldb, ldc, alpha, beta):
                _run_sgemm_nn_padded(A, B, C, m, n, k, alpha, beta, beta_is_zero)
            elif _can_use_thead_nn_tf32(m, n, k, lda, ldb, ldc, alpha, beta):
                _run_sgemm_nn_tf32(
                    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
                )
            elif _can_use_nomask_nn(m, n, k, lda, ldb, ldc, alpha, beta):
                _run_sgemm_nn_nomask(A, lda, B, ldb, C, ldc, m, n, k, alpha)
            else:
                dispatch = _build_sgemm_nn_dispatch_table(
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
                    grid,
                )
                runner = dispatch.lookup_and_build(m, n, k, aligned, snapshot_tensor=C)
                runner()
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            if m == n == k == 511:
                _run_sgemm_tn_511(A, lda, B, ldb, C, ldc, alpha, beta, beta_is_zero)
            elif m == n == k == 1023:
                _run_sgemm_tn_square_odd(
                    A, lda, B, ldb, C, ldc, m, alpha, beta, beta_is_zero
                )
            else:
                dispatch = _build_sgemm_tn_dispatch_table(
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
                    grid,
                )
                runner = dispatch.lookup_and_build(m, n, k, aligned, snapshot_tensor=C)
                runner()
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            runner = _SGEMM_NT_DISPATCH.lookup_and_build(
                m,
                n,
                k,
                aligned,
                context=dict(
                    A=A,
                    lda=lda,
                    B=B,
                    ldb=ldb,
                    C=C,
                    ldc=ldc,
                    m=m,
                    n=n,
                    k=k,
                    alpha=alpha,
                    beta=beta,
                    beta_is_zero=beta_is_zero,
                    grid=grid,
                    aligned=aligned,
                ),
            )
            runner()
        else:
            if m == n == k == 511:
                _run_sgemm_tt_511(A, lda, B, ldb, C, ldc, alpha, beta, beta_is_zero)
            else:
                dispatch = _build_sgemm_tt_dispatch_table(
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
                    grid,
                )
                runner = dispatch.lookup_and_build(m, n, k, aligned, snapshot_tensor=C)
                runner()
