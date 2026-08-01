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
from flag_blas.ops.level3.fp8gemm import (
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    FP8_DTYPES,
    ScalarType,
    _fp8gemm_nn_kernel,
    _fp8gemm_nt_kernel,
    _fp8gemm_tn_kernel,
    _fp8gemm_tt_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.dispatch import SizeAutoDispatch, StaticDispatch
from flag_blas.utils import libentry, libtuner
from flag_blas.utils.libentry import libcache

logger = logging.getLogger(__name__)

def fp8gemm(
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
    assert A.dtype in FP8_DTYPES
    assert B.dtype in FP8_DTYPES
    assert C.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert A.device == B.device == C.device
    assert transa in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert transb in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert m % 16 == 0
    assert n % 16 == 0
    assert k % 16 == 0
    assert A.data_ptr() % 16 == 0
    assert B.data_ptr() % 16 == 0
    assert C.data_ptr() % 16 == 0

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

    with torch_device_fn.device(A.device):
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            _fp8gemm_nn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            _fp8gemm_tn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            _fp8gemm_nt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        else:
            _fp8gemm_tt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
