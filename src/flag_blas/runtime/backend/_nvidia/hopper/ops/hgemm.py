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
from flag_blas.ops.level3.hgemm import (
    _HGEMM_KEY,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
    _hgemm_nn_kernel,
    _hgemm_nt_kernel,
    _hgemm_tn_kernel,
    _hgemm_tt_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.backend._nvidia.hopper.ops.sgemm import _is_gemm_aligned
from flag_blas.runtime.dispatch import StaticDispatch
from flag_blas.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hgemm_nn"),
    key=_HGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _hgemm_nn_kernel2(
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
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

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
        shape=(k, n),
        strides=(ldb, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    if BETA_IS_ZERO:
        result = acc * alpha
        tl.store(c_block_ptr, result.to(tl.float16), boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = acc * alpha + beta * c_vals
        tl.store(c_block_ptr, result.to(tl.float16), boundary_check=(0, 1))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hgemm_nn2"),
    key=_HGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _hgemm_nn_kernel3(
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
    a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float16))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float16))

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
        a = a_desc.load([pid_m * BLOCK_M, i * BLOCK_K])
        b = b_desc.load([i * BLOCK_K, pid_n * BLOCK_N])

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float16)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)
    else:
        c_vals = c_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float16)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)


@libentry()
@triton.jit
def _hgemm_nn_kernel4(
    desc_a,
    desc_b,
    desc_c,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
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

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        offs_k = i * BLOCK_K
        a = desc_a.load([offs_m, offs_k])
        b = desc_b.load([offs_k, offs_n])
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float16)
        desc_c.store([offs_m, offs_n], result)
    else:
        c_vals = desc_c.load([offs_m, offs_n]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float16)
        desc_c.store([offs_m, offs_n], result)


@libentry()
@triton.jit
def _hgemm_nn_kernel5(
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        mask_k = offs_k < k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < m) & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & (offs_n[None, :] < n),
            other=0.0,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb
        offs_k += BLOCK_K

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    if BETA_IS_ZERO:
        tl.store(c_ptrs, (alpha * acc).to(tl.float16), mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
        tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float16), mask=c_mask)


@libentry()
@triton.jit
def _hgemm_tn_kernel3(
    desc_a,
    desc_b,
    desc_c,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
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

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        offs_k = i * BLOCK_K

        a_t = desc_a.load([offs_k, offs_m])
        a = tl.trans(a_t)

        b = desc_b.load([offs_k, offs_n])

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float16)
        desc_c.store([offs_m, offs_n], result)
    else:
        c_vals = desc_c.load([offs_m, offs_n]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float16)
        desc_c.store([offs_m, offs_n], result)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hgemm_tn2"),
    key=_HGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _hgemm_tn_kernel2(
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

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_block_ptr = tl.advance(a_block_ptr, (BLOCK_K, 0))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float16)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float16)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


@libentry()
@triton.jit
def _hgemm_tn_kernel4(
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_t_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        mask_k = offs_k < k
        a_t = tl.load(
            a_t_ptrs,
            mask=mask_k[:, None] & (offs_m[None, :] < m),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & (offs_n[None, :] < n),
            other=0.0,
        )
        acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32)
        a_t_ptrs += BLOCK_K * lda
        b_ptrs += BLOCK_K * ldb
        offs_k += BLOCK_K

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    if BETA_IS_ZERO:
        tl.store(c_ptrs, (alpha * acc).to(tl.float16), mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
        tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float16), mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hgemm_nt"),
    key=_HGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _hgemm_nt_kernel2(
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

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    offset_n = (pid_n * BLOCK_N).to(tl.int32)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(lda, 1),
        offsets=(offset_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(1, ldb),
        offsets=(0, offset_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(m, n),
        strides=(ldc, 1),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float16)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float16)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


@libentry()
@triton.jit
def _hgemm_nt_kernel3(
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_t_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        mask_k = offs_k < k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < m) & mask_k[None, :],
            other=0.0,
        )
        b_t = tl.load(
            b_t_ptrs,
            mask=(offs_n[:, None] < n) & mask_k[None, :],
            other=0.0,
        )
        acc = tl.dot(a, tl.trans(b_t), acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K
        b_t_ptrs += BLOCK_K
        offs_k += BLOCK_K

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    if BETA_IS_ZERO:
        tl.store(c_ptrs, (alpha * acc).to(tl.float16), mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
        tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float16), mask=c_mask)


@libentry()
@triton.jit
def _hgemm_nt_kernel4(
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

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * lda + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] + offs_n[None, :] * ldb

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    is_full_m = (pid_m * BLOCK_M + BLOCK_M) <= m
    is_full_n = (pid_n * BLOCK_N + BLOCK_N) <= n
    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K

    if is_full_m and is_full_n:
        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    else:
        mask_m = offs_m < m
        mask_n = offs_n < n
        a_mask_base = mask_m[:, None]
        b_mask_base = mask_n[None, :]

        for _ in range(0, k_full_iters):
            a = tl.load(a_ptrs, mask=a_mask_base, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_base, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            a_mask_tail = mask_m[:, None] & mask_k[None, :]
            b_mask_tail = mask_k[:, None] & mask_n[None, :]
            a = tl.load(a_ptrs, mask=a_mask_tail, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask_tail, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]

    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.float16))
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float16))
    else:
        mask_m = offs_m < m
        mask_n = offs_n < n
        c_mask = mask_m[:, None] & mask_n[None, :]

        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * acc).to(tl.float16), mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
            tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float16), mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hgemm_nn"),
    key=_HGEMM_KEY,
    restore_value=["c_ptr"],
)
@triton.jit
def _hgemm_tt_kernel2(
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
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float16)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))
    else:
        c_vals = tl.load(c_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float16)
        tl.store(c_block_ptr, result, boundary_check=(0, 1))


@libentry()
@triton.jit
def _hgemm_tt_kernel3(
    desc_a,
    desc_b,
    desc_c,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
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

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        offs_k = i * BLOCK_K
        a_t = desc_a.load([offs_k, offs_m])
        a = tl.trans(a_t)
        b_t = desc_b.load([offs_n, offs_k])
        b = tl.trans(b_t)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    if BETA_IS_ZERO:
        result = (alpha * acc).to(tl.float16)
        desc_c.store([offs_m, offs_n], result)
    else:
        c_vals = desc_c.load([offs_m, offs_n]).to(tl.float32)
        result = (alpha * acc + beta * c_vals).to(tl.float16)
        desc_c.store([offs_m, offs_n], result)


@libentry()
@triton.jit
def _hgemm_tt_kernel4(
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_t_ptrs = a_ptr + offs_k[:, None] * lda + offs_m[None, :]
    b_t_ptrs = b_ptr + offs_n[:, None] * ldb + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k, BLOCK_K)):
        mask_k = offs_k < k
        a_t = tl.load(
            a_t_ptrs,
            mask=mask_k[:, None] & (offs_m[None, :] < m),
            other=0.0,
        )
        b_t = tl.load(
            b_t_ptrs,
            mask=(offs_n[:, None] < n) & mask_k[None, :],
            other=0.0,
        )
        acc = tl.dot(tl.trans(a_t), tl.trans(b_t), acc, out_dtype=tl.float32)
        a_t_ptrs += BLOCK_K * lda
        b_t_ptrs += BLOCK_K
        offs_k += BLOCK_K

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    if BETA_IS_ZERO:
        tl.store(c_ptrs, (alpha * acc).to(tl.float16), mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
        tl.store(c_ptrs, (alpha * acc + beta * c_vals).to(tl.float16), mask=c_mask)


@libentry()
@triton.jit
def _hgemm_tn_kernel6(
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

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[k, m], strides=[lda, 1], block_shape=[BLOCK_K, BLOCK_M]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[k, n], strides=[ldb, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[m, n], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        offs_k = i * BLOCK_K
        a_t = a_desc.load([offs_k, offs_m])
        b = b_desc.load([offs_k, offs_n])
        acc = tl.dot(tl.trans(a_t), b, acc, out_dtype=tl.float32)

    if BETA_IS_ZERO:
        c_desc.store([offs_m, offs_n], (alpha * acc).to(tl.float16))
    else:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        c_desc.store([offs_m, offs_n], (alpha * acc + beta * c_vals).to(tl.float16))


@libentry()
@triton.jit
def _hgemm_nt_kernel5(
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

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

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
        offs_k = i * BLOCK_K
        a = a_desc.load([offs_m, offs_k])
        b_t = b_desc.load([offs_n, offs_k])
        acc = tl.dot(a, tl.trans(b_t), acc, out_dtype=tl.float32)

    if BETA_IS_ZERO:
        c_desc.store([offs_m, offs_n], (alpha * acc).to(tl.float16))
    else:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        c_desc.store([offs_m, offs_n], (alpha * acc + beta * c_vals).to(tl.float16))


@libentry()
@triton.jit
def _hgemm_tt_kernel6(
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

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[k, m], strides=[lda, 1], block_shape=[BLOCK_K, BLOCK_M]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[n, k], strides=[ldb, 1], block_shape=[BLOCK_N, BLOCK_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[m, n], strides=[ldc, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_K)):
        offs_k = i * BLOCK_K
        a_t = a_desc.load([offs_k, offs_m])
        b_t = b_desc.load([offs_n, offs_k])
        acc = tl.dot(tl.trans(a_t), tl.trans(b_t), acc, out_dtype=tl.float32)

    if BETA_IS_ZERO:
        c_desc.store([offs_m, offs_n], (alpha * acc).to(tl.float16))
    else:
        c_vals = c_desc.load([offs_m, offs_n]).to(tl.float32)
        c_desc.store([offs_m, offs_n], (alpha * acc + beta * c_vals).to(tl.float16))


# ---------------------------------------------------------------------------
# Module-level condition predicates for hgemm StaticDispatch
# ---------------------------------------------------------------------------


def _hgemm_nn_is_skinny_aligned_large(m, n, k, aligned, **_kw):
    return (
        aligned
        and (m * n > 2048 * 2048)
        and min(m, n) >= 64
        and ((m >= 16384 and max(n, k) <= 2048) or (n >= 16384 and max(m, k) <= 2048))
    )


def _hgemm_nn_is_m64_mid(m, n, k, aligned, **_kw):
    return aligned and m == 64 and n >= 512 and k >= 512


def _hgemm_nn_is_n64_mid(m, n, k, aligned, **_kw):
    return aligned and n == 64 and m >= 512 and k >= 512


def _hgemm_nn_is_128_mid(m, n, k, aligned, **_kw):
    return aligned and k <= 1024 and max(m, n) >= 4096 and min(m, n) == 128


def _hgemm_nn_is_2048_wide_k4096(m, n, k, aligned, **_kw):
    return aligned and m == 2048 and k == 4096 and 11008 <= n <= 12288


def _hgemm_nn_is_aligned_large(m, n, k, aligned, **_kw):
    return aligned and (m * n > 2048 * 2048) and min(m, n) >= 64


def _hgemm_nn_is_aligned_tiny(m, n, k, aligned, **_kw):
    return aligned and max(m, n, k) <= 512


def _hgemm_nn_is_aligned_small(m, n, k, aligned, **_kw):
    return aligned and max(m, n) <= 1024


def _hgemm_nn_is_default(**_kw):
    return True


# ---------------------------------------------------------------------------
# Module-level factory functions for hgemm StaticDispatch
# ---------------------------------------------------------------------------
def _hgemm_nn_build_kernel5_64x128(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_nn_kernel5[(triton.cdiv(m, 64) * triton.cdiv(n, 128),)](
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
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=128,
        GROUP_M=8,
        num_stages=4,
        num_warps=4,
        num_ctas=1,
    )


def _hgemm_nn_build_kernel5_128x64(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_nn_kernel5[(triton.cdiv(m, 128) * triton.cdiv(n, 64),)](
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
        BLOCK_M=128,
        BLOCK_N=64,
        BLOCK_K=128,
        GROUP_M=8,
        num_stages=4,
        num_warps=4,
        num_ctas=1,
    )


def _hgemm_nn_build_kernel4(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_nn_kernel4[(triton.cdiv(m, 128) * triton.cdiv(n, 256),)](
        TensorDescriptor(base=A, shape=[m, k], strides=[lda, 1], block_shape=[128, 64]),
        TensorDescriptor(base=B, shape=[k, n], strides=[ldb, 1], block_shape=[64, 256]),
        TensorDescriptor(
            base=C, shape=[m, n], strides=[ldc, 1], block_shape=[128, 256]
        ),
        alpha,
        beta,
        m,
        n,
        k,
        beta_is_zero,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=64,
        GROUP_M=8,
        num_stages=4,
        num_warps=8,
        num_ctas=1,
    )


def _hgemm_nn_build_kernel3(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return lambda: _hgemm_nn_kernel3[grid](
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
    )


def _hgemm_nn_build_kernel2(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return lambda: _hgemm_nn_kernel2[grid](
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
    )


def _hgemm_nn_build_kernel(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return lambda: _hgemm_nn_kernel[grid](
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
    )


_HGEMM_NN_DISPATCH = StaticDispatch(
    [
        (_hgemm_nn_is_aligned_tiny, _hgemm_nn_build_kernel),
        (_hgemm_nn_is_m64_mid, _hgemm_nn_build_kernel5_64x128),
        (_hgemm_nn_is_n64_mid, _hgemm_nn_build_kernel5_64x128),
        (_hgemm_nn_is_128_mid, _hgemm_nn_build_kernel5_64x128),
        (_hgemm_nn_is_2048_wide_k4096, _hgemm_nn_build_kernel4),
        (_hgemm_nn_is_skinny_aligned_large, _hgemm_nn_build_kernel4),
        (_hgemm_nn_is_aligned_large, _hgemm_nn_build_kernel3),
        (_hgemm_nn_is_aligned_small, _hgemm_nn_build_kernel2),
        (_hgemm_nn_is_default, _hgemm_nn_build_kernel),
    ]
)


# ---------------------------------------------------------------------------
# Module-level condition predicates for hgemm_tn StaticDispatch
# ---------------------------------------------------------------------------
def _hgemm_tn_is_aligned_tiny(m, n, k, aligned, **_kw):
    return aligned and max(m, n, k) <= 512


def _hgemm_tn_is_m64_mid(m, n, k, aligned, **_kw):
    return aligned and m == 64 and n >= 1024 and k >= 512


def _hgemm_tn_is_m64_n1024(m, n, k, aligned, **_kw):
    return aligned and m == 64 and n == 1024 and k == 512


def _hgemm_tn_is_1024_square(m, n, k, aligned, **_kw):
    return aligned and m == 1024 and n == 1024 and k == 1024


def _hgemm_tn_is_128_mid(m, n, k, aligned, **_kw):
    return aligned and k <= 1024 and max(m, n) >= 4096 and min(m, n) == 128


def _hgemm_tn_is_tma_large(m, n, k, aligned, **_kw):
    return aligned and k >= 2048 and min(m, n) >= 512 and m * n >= 2048 * 2048


def _hgemm_tn_is_default(**_kw):
    return True


# ---------------------------------------------------------------------------
# Module-level condition predicates for hgemm_nt StaticDispatch
# ---------------------------------------------------------------------------
def _hgemm_nt_is_aligned_tiny(m, n, k, aligned, **_kw):
    return aligned and max(m, n, k) <= 512


def _hgemm_nt_is_m64_mid(m, n, k, aligned, **_kw):
    return aligned and m == 64 and n >= 1024 and k >= 512


def _hgemm_nt_is_m64_n2048_k1024(m, n, k, aligned, **_kw):
    return aligned and m == 64 and n == 2048 and k == 1024


def _hgemm_nt_is_1023_square(m, n, k, aligned, **_kw):
    return m == 1023 and n == 1023 and k == 1023


def _hgemm_nt_is_tma_huge_k(m, n, k, aligned, **_kw):
    return aligned and k > 8192 and m * n >= 2048 * 4096


def _hgemm_nt_is_2048_square(m, n, k, aligned, **_kw):
    return aligned and m == 2048 and n == 2048 and k == 2048


def _hgemm_nt_is_tma_large_wide_tile(m, n, k, aligned, **_kw):
    return aligned and k >= 2048 and min(m, n) >= 512 and m * n >= 2048 * 2048


def _hgemm_nt_is_aligned(m, n, k, aligned, **_kw):
    return aligned


def _hgemm_nt_is_default(**_kw):
    return True


# ---------------------------------------------------------------------------
# Module-level condition predicates for hgemm_tt StaticDispatch
# ---------------------------------------------------------------------------
def _hgemm_tt_is_aligned_tiny(m, n, k, aligned, **_kw):
    return aligned and max(m, n, k) <= 512


def _hgemm_tt_is_m64_mid(m, n, k, aligned, **_kw):
    return aligned and m == 64 and n >= 1024 and k >= 512


def _hgemm_tt_is_1024_square(m, n, k, aligned, **_kw):
    return aligned and m == 1024 and n == 1024 and k == 1024


def _hgemm_tt_is_128_mid_wide(m, n, k, aligned, **_kw):
    return aligned and m == 128 and n >= 4096 and k <= 1024


def _hgemm_tt_is_128_mid_tall(m, n, k, aligned, **_kw):
    return aligned and n == 128 and m >= 4096 and k <= 1024


def _hgemm_tt_is_tma_large_square(m, n, k, aligned, **_kw):
    return aligned and k >= 2048 and m == n and m >= 8192


def _hgemm_tt_is_tall_1024(m, n, k, aligned, **_kw):
    return aligned and m >= 32768 and n == 1024 and k == 1024


def _hgemm_tt_is_tma_large_wide_aspect(m, n, k, aligned, **_kw):
    return aligned and k >= 2048 and n > m and m >= 8192 and n >= 8192


def _hgemm_tt_is_tma_large_tall_tile(m, n, k, aligned, **_kw):
    return aligned and k >= 2048 and m >= 8192 and n >= 8192


def _hgemm_tt_is_tma_large_wide_tile(m, n, k, aligned, **_kw):
    return aligned and k >= 2048 and min(m, n) >= 512 and m * n >= 2048 * 2048


def _hgemm_tt_is_default(**_kw):
    return True


# ---------------------------------------------------------------------------
# Module-level factory functions for hgemm_tn StaticDispatch
# ---------------------------------------------------------------------------
def _hgemm_tn_build_kernel3(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_tn_kernel3[(triton.cdiv(m, 128) * triton.cdiv(n, 256),)](
        TensorDescriptor(base=A, shape=[k, m], strides=[lda, 1], block_shape=[64, 128]),
        TensorDescriptor(base=B, shape=[k, n], strides=[ldb, 1], block_shape=[64, 256]),
        TensorDescriptor(
            base=C, shape=[m, n], strides=[ldc, 1], block_shape=[128, 256]
        ),
        alpha,
        beta,
        m,
        n,
        k,
        beta_is_zero,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=64,
        GROUP_M=8,
        num_stages=4,
        num_warps=8,
        num_ctas=1,
    )


def _hgemm_tn_build_kernel2(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return lambda: _hgemm_tn_kernel2[grid](
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
    )


def _hgemm_tn_build_kernel(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return lambda: _hgemm_tn_kernel[grid](
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
    )


def _hgemm_tn_build_kernel4_64x64(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_tn_kernel4[(triton.cdiv(m, 64) * triton.cdiv(n, 64),)](
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
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=128,
        GROUP_M=8,
        num_stages=3,
        num_warps=4,
        num_ctas=1,
    )


def _hgemm_tn_build_kernel4_m64(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_tn_kernel4[(triton.cdiv(m, 64) * triton.cdiv(n, 128),)](
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
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=128,
        GROUP_M=8,
        num_stages=4,
        num_warps=4,
        num_ctas=1,
    )


def _hgemm_tn_build_kernel6_128x256(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_tn_kernel6[(triton.cdiv(m, 128) * triton.cdiv(n, 256),)](
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
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=64,
        GROUP_M=8,
        num_stages=3,
        num_warps=8,
        num_ctas=1,
    )


# ---------------------------------------------------------------------------
# Module-level factory functions for hgemm_nt StaticDispatch
# ---------------------------------------------------------------------------
def _hgemm_nt_build_kernel2(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return lambda: _hgemm_nt_kernel2[grid](
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
    )


def _hgemm_nt_build_kernel(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return lambda: _hgemm_nt_kernel[grid](
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
    )


def _hgemm_nt_build_kernel4_32x32(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_nt_kernel4[(triton.cdiv(m, 32) * triton.cdiv(n, 32),)](
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
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=128,
        GROUP_M=8,
        num_stages=3,
        num_warps=8,
        num_ctas=1,
    )


def _hgemm_nt_build_kernel4_64x64_k32(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_nt_kernel4[(triton.cdiv(m, 64) * triton.cdiv(n, 64),)](
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
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        GROUP_M=8,
        num_stages=3,
        num_warps=8,
        num_ctas=1,
    )


def _hgemm_nt_build_kernel3_64x64(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_nt_kernel3[(triton.cdiv(m, 64) * triton.cdiv(n, 64),)](
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
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=128,
        GROUP_M=8,
        num_stages=3,
        num_warps=4,
        num_ctas=1,
    )


def _hgemm_nt_build_kernel5_128x256(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_nt_kernel5[(triton.cdiv(m, 128) * triton.cdiv(n, 256),)](
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
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=64,
        GROUP_M=8,
        num_stages=3,
        num_warps=8,
        num_ctas=1,
    )


def _hgemm_nt_build_kernel5_256x128(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_nt_kernel5[(triton.cdiv(m, 256) * triton.cdiv(n, 128),)](
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
        BLOCK_M=256,
        BLOCK_N=128,
        BLOCK_K=64,
        GROUP_M=8,
        num_stages=3,
        num_warps=8,
        num_ctas=1,
    )


# ---------------------------------------------------------------------------
# Module-level factory functions for hgemm_tt StaticDispatch
# ---------------------------------------------------------------------------
def _hgemm_tt_build_kernel3(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_tt_kernel3[(triton.cdiv(m, 128) * triton.cdiv(n, 256),)](
        TensorDescriptor(base=A, shape=[k, m], strides=[lda, 1], block_shape=[64, 128]),
        TensorDescriptor(base=B, shape=[n, k], strides=[ldb, 1], block_shape=[256, 64]),
        TensorDescriptor(
            base=C, shape=[m, n], strides=[ldc, 1], block_shape=[128, 256]
        ),
        alpha,
        beta,
        m,
        n,
        k,
        beta_is_zero,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=64,
        GROUP_M=8,
        num_stages=4,
        num_warps=8,
        num_ctas=1,
    )


def _hgemm_tt_build_kernel2(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return lambda: _hgemm_tt_kernel2[grid](
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
    )


def _hgemm_tt_build_kernel(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return lambda: _hgemm_tt_kernel[grid](
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
    )


def _hgemm_tt_build_kernel4_64x64(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_tt_kernel4[(triton.cdiv(m, 64) * triton.cdiv(n, 64),)](
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
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=128,
        GROUP_M=8,
        num_stages=3,
        num_warps=4,
        num_ctas=1,
    )


def _hgemm_tt_build_kernel4_64x128(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_tt_kernel4[(triton.cdiv(m, 64) * triton.cdiv(n, 128),)](
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
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=128,
        GROUP_M=8,
        num_stages=4,
        num_warps=4,
        num_ctas=1,
    )


def _hgemm_tt_build_kernel6_128x256(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_tt_kernel6[(triton.cdiv(m, 128) * triton.cdiv(n, 256),)](
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
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=64,
        GROUP_M=8,
        num_stages=3,
        num_warps=8,
        num_ctas=1,
    )


def _hgemm_tt_build_kernel6_256x128(
    A,
    B,
    C,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    alpha,
    beta,
    beta_is_zero,
):
    return lambda: _hgemm_tt_kernel6[(triton.cdiv(m, 256) * triton.cdiv(n, 128),)](
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
        BLOCK_M=256,
        BLOCK_N=128,
        BLOCK_K=64,
        GROUP_M=8,
        num_stages=3,
        num_warps=8,
        num_ctas=1,
    )


_HGEMM_TN_DISPATCH = StaticDispatch(
    [
        (_hgemm_tn_is_aligned_tiny, _hgemm_tn_build_kernel),
        (_hgemm_tn_is_m64_n1024, _hgemm_tn_build_kernel4_64x64),
        (_hgemm_tn_is_m64_mid, _hgemm_tn_build_kernel4_m64),
        (_hgemm_tn_is_128_mid, _hgemm_tn_build_kernel4_m64),
        (_hgemm_tn_is_1024_square, _hgemm_tn_build_kernel4_m64),
        (_hgemm_tn_is_tma_large, _hgemm_tn_build_kernel6_128x256),
        (_hgemm_tn_is_default, _hgemm_tn_build_kernel),
    ]
)

_HGEMM_NT_DISPATCH = StaticDispatch(
    [
        (_hgemm_nt_is_aligned_tiny, _hgemm_nt_build_kernel),
        (_hgemm_nt_is_m64_n2048_k1024, _hgemm_nt_build_kernel4_32x32),
        (_hgemm_nt_is_m64_mid, _hgemm_nt_build_kernel3_64x64),
        (_hgemm_nt_is_1023_square, _hgemm_nt_build_kernel4_64x64_k32),
        (_hgemm_nt_is_tma_huge_k, _hgemm_nt_build_kernel5_128x256),
        (_hgemm_nt_is_2048_square, _hgemm_nt_build_kernel5_128x256),
        (_hgemm_nt_is_tma_large_wide_tile, _hgemm_nt_build_kernel5_128x256),
        (_hgemm_nt_is_aligned, _hgemm_nt_build_kernel2),
        (_hgemm_nt_is_default, _hgemm_nt_build_kernel),
    ]
)

_HGEMM_TT_DISPATCH = StaticDispatch(
    [
        (_hgemm_tt_is_aligned_tiny, _hgemm_tt_build_kernel4_64x64),
        (_hgemm_tt_is_m64_mid, _hgemm_tt_build_kernel4_64x64),
        (_hgemm_tt_is_128_mid_wide, _hgemm_tt_build_kernel4_64x64),
        (_hgemm_tt_is_128_mid_tall, _hgemm_tt_build_kernel4_64x128),
        (_hgemm_tt_is_1024_square, _hgemm_tt_build_kernel4_64x128),
        (_hgemm_tt_is_tall_1024, _hgemm_tt_build_kernel6_128x256),
        (_hgemm_tt_is_tma_large_square, _hgemm_tt_build_kernel6_128x256),
        (_hgemm_tt_is_tma_large_wide_aspect, _hgemm_tt_build_kernel6_128x256),
        (_hgemm_tt_is_tma_large_tall_tile, _hgemm_tt_build_kernel6_256x128),
        (_hgemm_tt_is_tma_large_wide_tile, _hgemm_tt_build_kernel6_128x256),
        (_hgemm_tt_is_default, _hgemm_tt_build_kernel),
    ]
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

    aligned = _is_gemm_aligned(A, lda, B, ldb, C, ldc)
    with torch_device_fn.device(A.device):
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            runner = _HGEMM_NN_DISPATCH.lookup_and_build(
                m,
                n,
                k,
                aligned,
                context=dict(
                    A=A,
                    B=B,
                    C=C,
                    m=m,
                    n=n,
                    k=k,
                    lda=lda,
                    ldb=ldb,
                    ldc=ldc,
                    alpha=alpha,
                    beta=beta,
                    beta_is_zero=beta_is_zero,
                ),
            )
            runner()
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            runner = _HGEMM_TN_DISPATCH.lookup_and_build(
                m,
                n,
                k,
                aligned,
                context=dict(
                    A=A,
                    B=B,
                    C=C,
                    m=m,
                    n=n,
                    k=k,
                    lda=lda,
                    ldb=ldb,
                    ldc=ldc,
                    alpha=alpha,
                    beta=beta,
                    beta_is_zero=beta_is_zero,
                ),
            )
            runner()
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            runner = _HGEMM_NT_DISPATCH.lookup_and_build(
                m,
                n,
                k,
                aligned,
                context=dict(
                    A=A,
                    B=B,
                    C=C,
                    m=m,
                    n=n,
                    k=k,
                    lda=lda,
                    ldb=ldb,
                    ldc=ldc,
                    alpha=alpha,
                    beta=beta,
                    beta_is_zero=beta_is_zero,
                ),
            )
            runner()
        else:
            runner = _HGEMM_TT_DISPATCH.lookup_and_build(
                m,
                n,
                k,
                aligned,
                context=dict(
                    A=A,
                    B=B,
                    C=C,
                    m=m,
                    n=n,
                    k=k,
                    lda=lda,
                    ldb=ldb,
                    ldc=ldc,
                    alpha=alpha,
                    beta=beta,
                    beta_is_zero=beta_is_zero,
                ),
            )
            runner()
