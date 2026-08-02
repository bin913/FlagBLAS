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

        acc = tl.dot(a_t, b_t, acc, out_dtype=tl.float32, input_precision="tf32x3")

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

        acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32x3")

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
            a_t, tl.trans(b_t), acc, out_dtype=tl.float32, input_precision="tf32x3"
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
    """NT gemm using block_ptr + tf32x3 — no stride alignment requirement."""
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
            a, tl.trans(b), acc, out_dtype=tl.float32, input_precision="tf32x3"
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
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32x3")

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
                b, tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="tf32x3"
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)
            acc_t = tl.dot(
                b, tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="tf32x3"
            )
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N

        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(
                b, tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="tf32x3"
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(
                b, tl.trans(a), acc_t, out_dtype=tl.float32, input_precision="tf32x3"
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
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb

    if remainder > 0:
        mask_k = offs_k < remainder
        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_k[:, None] & mask_n[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

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
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    if remainder > 0:
        mask_k = offs_k < remainder
        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_n[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32, allow_tf32=False)

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

    a_ptrs = a_ptr + (offs_k[:, None] * lda + offs_am[None, :])
    b_ptrs = b_ptr + (offs_bn[:, None] * ldb + offs_k[None, :])

    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    mask_m = offs_am < m
    mask_n = offs_bn < n

    k_remain = k_end - k_begin
    full_iters = k_remain // BLOCK_K
    remainder = k_remain % BLOCK_K

    for i in range(0, full_iters):
        a = tl.load(a_ptrs, mask=mask_m[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[:, None], other=0.0)
        acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32, input_precision="tf32x3")
        a_ptrs += BLOCK_K * lda
        b_ptrs += BLOCK_K

    if remainder > 0:
        mask_k = offs_k < remainder
        a_mask = mask_k[:, None] & mask_m[None, :]
        b_mask = mask_n[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc_t = tl.dot(b, a, acc_t, out_dtype=tl.float32, input_precision="tf32x3")

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


def _sgemm_pad_tensors(
    A: torch.Tensor,
    lda: int,
    transa: int,
    B: torch.Tensor,
    ldb: int,
    transb: int,
    C: torch.Tensor,
    ldc: int,
    m: int,
    n: int,
    k: int,
):
    pad_m = (16 - (m % 16)) % 16
    pad_n = (16 - (n % 16)) % 16
    pad_k = (16 - (k % 16)) % 16

    A_2d = A.view(-1, lda)
    B_2d = B.view(-1, ldb)
    C_2d = C.view(-1, ldc)

    if transa == CUBLAS_OP_N:
        A_padded = torch.nn.functional.pad(A_2d[:m, :k], (0, pad_k, 0, pad_m))
        lda_pad = k + pad_k
    else:
        A_padded = torch.nn.functional.pad(A_2d[:k, :m], (0, pad_m, 0, pad_k))
        lda_pad = m + pad_m

    if transb == CUBLAS_OP_N:
        B_padded = torch.nn.functional.pad(B_2d[:k, :n], (0, pad_n, 0, pad_k))
        ldb_pad = n + pad_n
    else:
        B_padded = torch.nn.functional.pad(B_2d[:n, :k], (0, pad_k, 0, pad_n))
        ldb_pad = k + pad_k

    C_padded = torch.nn.functional.pad(C_2d[:m, :n], (0, pad_n, 0, pad_m))
    ldc_pad = n + pad_n

    m_pad = m + pad_m
    n_pad = n + pad_n
    k_pad = k + pad_k

    return (
        A_padded,
        B_padded,
        C_padded,
        C_2d,
        lda_pad,
        ldb_pad,
        ldc_pad,
        m_pad,
        n_pad,
        k_pad,
    )


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


def _is_sgemm_medium(m: int, n: int, k: int, **_kw) -> bool:
    return min(m, n, k) >= 256 and not _is_sgemm_large(m, n, k)


def _make_sgemm_nn_thin_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        num_k_splits = min(triton.cdiv(k, 128), 16)
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)

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


def _make_sgemm_nn_padded_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        (
            A_padded,
            B_padded,
            C_padded,
            C_2d,
            lda_pad,
            ldb_pad,
            ldc_pad,
            m_pad,
            n_pad,
            k_pad,
        ) = _sgemm_pad_tensors(
            A, lda, CUBLAS_OP_N, B, ldb, CUBLAS_OP_N, C, ldc, m, n, k
        )

        grid_pad = lambda meta: (
            triton.cdiv(m_pad, meta["BLOCK_M"]) * triton.cdiv(n_pad, meta["BLOCK_N"]),
        )

        _sgemm_nn_kernel2[grid_pad](
            A_padded,
            B_padded,
            C_padded,
            alpha,
            beta,
            m_pad,
            n_pad,
            k_pad,
            lda_pad,
            ldb_pad,
            ldc_pad,
            beta_is_zero,
        )

        C_2d[:m, :n] = C_padded[:m, :n]

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
        table_name="sgemm_nn_variant",
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
        lambda: _make_sgemm_nn_padded_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="padded_k2",
        filter=_is_sgemm_large,
    )
    dispatch.add(
        lambda: _make_sgemm_nn_thin_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="thin",
        filter=lambda m, n, k, **kw: not _is_sgemm_large(m, n, k),
    )
    dispatch.add(
        lambda: _make_sgemm_nn_fallback_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=False,
        name="fallback",
        filter=lambda m, n, k, **kw: not _is_sgemm_large(m, n, k),
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


def _make_sgemm_tn_padded_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        (
            A_padded,
            B_padded,
            C_padded,
            C_2d,
            lda_pad,
            ldb_pad,
            ldc_pad,
            m_pad,
            n_pad,
            k_pad,
        ) = _sgemm_pad_tensors(
            A, lda, CUBLAS_OP_T, B, ldb, CUBLAS_OP_N, C, ldc, m, n, k
        )

        grid_pad = lambda meta: (
            triton.cdiv(m_pad, meta["BLOCK_M"]) * triton.cdiv(n_pad, meta["BLOCK_N"]),
        )

        _sgemm_tn_kernel2[grid_pad](
            A_padded,
            B_padded,
            C_padded,
            alpha,
            beta,
            m_pad,
            n_pad,
            k_pad,
            lda_pad,
            ldb_pad,
            ldc_pad,
            beta_is_zero,
        )

        C_2d[:m, :n] = C_padded[:m, :n]

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
        table_name="sgemm_tn_variant",
        build_key=lambda m, n, k, aligned, **extra: (m, n, k, int(aligned)),
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
        name="padded_k2",
        filter=_is_sgemm_large,
    )
    dispatch.add(
        lambda: _make_sgemm_tn_fallback_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=False,
        name="fallback",
        filter=lambda m, n, k, **kw: not _is_sgemm_large(m, n, k),
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


def _make_sgemm_nt_padded_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        (
            A_padded,
            B_padded,
            C_padded,
            C_2d,
            lda_pad,
            ldb_pad,
            ldc_pad,
            m_pad,
            n_pad,
            k_pad,
        ) = _sgemm_pad_tensors(
            A, lda, CUBLAS_OP_N, B, ldb, CUBLAS_OP_T, C, ldc, m, n, k
        )

        grid_pad = lambda meta: (
            triton.cdiv(m_pad, meta["BLOCK_M"]) * triton.cdiv(n_pad, meta["BLOCK_N"]),
        )

        _sgemm_nt_kernel2[grid_pad](
            A_padded,
            B_padded,
            C_padded,
            alpha,
            beta,
            m_pad,
            n_pad,
            k_pad,
            lda_pad,
            ldb_pad,
            ldc_pad,
            beta_is_zero,
        )

        C_2d[:m, :n] = C_padded[:m, :n]

    return run


def _make_sgemm_nt_thin_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        num_k_splits = min(triton.cdiv(k, 128), 16)
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)

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
    """Runner for NT block_ptr kernel3 with tf32x3."""

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
    return m == 1023 and n == 1023 and k == 1023


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
        table_name="sgemm_nt_variant",
        build_key=lambda m, n, k, aligned, **extra: (m, n, k, int(aligned), 5),
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
        name="padded_k2",
        filter=_is_sgemm_large,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_kernel3_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=False,
        name="kernel3_square_near_pow2",
        filter=_is_sgemm_square_near_pow2,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_padded_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="padded_k2_square_near_pow2_large",
        filter=lambda m, n, k, **kw: _is_sgemm_square_near_pow2(m, n, k) and m >= 768,
    )
    dispatch.add(
        lambda: _make_sgemm_nt_thin_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="thin",
        filter=lambda m, n, k, **kw: not _is_sgemm_large(m, n, k)
        and not _is_sgemm_square_near_pow2(m, n, k),
    )
    dispatch.add(
        lambda: _make_sgemm_nt_fallback_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=False,
        name="fallback",
        filter=lambda m, n, k, **kw: not _is_sgemm_large(m, n, k)
        and not _is_sgemm_square_near_pow2(m, n, k),
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


def _make_sgemm_tt_padded_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        (
            A_padded,
            B_padded,
            C_padded,
            C_2d,
            lda_pad,
            ldb_pad,
            ldc_pad,
            m_pad,
            n_pad,
            k_pad,
        ) = _sgemm_pad_tensors(
            A, lda, CUBLAS_OP_T, B, ldb, CUBLAS_OP_T, C, ldc, m, n, k
        )

        grid_pad = lambda meta: (
            triton.cdiv(m_pad, meta["BLOCK_M"]) * triton.cdiv(n_pad, meta["BLOCK_N"]),
        )

        _sgemm_tt_kernel2[grid_pad](
            A_padded,
            B_padded,
            C_padded,
            alpha,
            beta,
            m_pad,
            n_pad,
            k_pad,
            lda_pad,
            ldb_pad,
            ldc_pad,
            beta_is_zero,
        )

        C_2d[:m, :n] = C_padded[:m, :n]

    return run


def _make_sgemm_tt_thin_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
):
    def run():
        num_k_splits = min(triton.cdiv(k, 128), 16)
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)

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
        table_name="sgemm_tt_variant",
        build_key=lambda m, n, k, aligned, **extra: (m, n, k, int(aligned)),
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
        name="padded_k2",
        filter=_is_sgemm_large,
    )
    dispatch.add(
        lambda: _make_sgemm_tt_thin_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero
        ),
        aligned=False,
        name="thin",
        filter=lambda m, n, k, **kw: not _is_sgemm_large(m, n, k),
    )
    dispatch.add(
        lambda: _make_sgemm_tt_fallback_runner(
            A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid
        ),
        aligned=False,
        name="fallback",
        filter=lambda m, n, k, **kw: not _is_sgemm_large(m, n, k),
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
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)
        return

    if alpha == 0.0:
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)
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
            if m == n == k == 511 and lda == k and ldb == n and ldc == n:
                _sgemm_nn_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
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
