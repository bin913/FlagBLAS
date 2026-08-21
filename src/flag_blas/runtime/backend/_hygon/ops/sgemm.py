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

import torch
import triton

from flag_blas import runtime
from flag_blas.ops.level3.sgemm import CUBLAS_OP_N, CUBLAS_OP_T, ScalarType
from flag_blas.ops.level3.sgemm import (
    _sgemm_nn_kernel as common_sgemm_nn_kernel,
    _sgemm_nt_kernel as common_sgemm_nt_kernel,
    _sgemm_tn_kernel as common_sgemm_tn_kernel,
    _sgemm_tt_kernel as common_sgemm_tt_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

_SGEMM_KEY = ["m", "n", "k", "BETA_IS_ZERO"]

sgemm_nn_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemm_hygon"),
        key=_SGEMM_KEY,
        restore_value=["c_ptr"],
    )(common_sgemm_nn_kernel.fn.fn)
)

sgemm_tn_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemm_hygon"),
        key=_SGEMM_KEY,
        restore_value=["c_ptr"],
    )(common_sgemm_tn_kernel.fn.fn)
)

sgemm_nt_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemm_hygon"),
        key=_SGEMM_KEY,
        restore_value=["c_ptr"],
    )(common_sgemm_nt_kernel.fn.fn)
)

sgemm_tt_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemm_hygon"),
        key=_SGEMM_KEY,
        restore_value=["c_ptr"],
    )(common_sgemm_tt_kernel.fn.fn)
)


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
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            sgemm_nn_hygon_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            sgemm_tn_hygon_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            sgemm_nt_hygon_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        else:
            sgemm_tt_hygon_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )


__all__ = ["sgemm"]
