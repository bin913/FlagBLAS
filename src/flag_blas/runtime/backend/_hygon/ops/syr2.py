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
from flag_blas.ops.level2._constants import CUBLAS_FILL_MODE_UPPER
from flag_blas.ops.level2.syr2 import (
    ScalarType,
    _check_ssyr2_args,
    _check_syr2_args,
    _f64_to_i64,
    _row_major_uplo,
)
from flag_blas.ops.level2.syr2 import dsyr2_kernel as common_dsyr2_kernel
from flag_blas.ops.level2.syr2 import ssyr2_kernel as common_ssyr2_kernel
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

ssyr2_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("ssyr2_hygon"),
        key=["n", "LDA", "INCX", "INCY", "UPLO"],
        restore_value=["a_ptr"],
    )(common_ssyr2_kernel.fn.fn)
)

ssyr2_large_lower_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("ssyr2_large_lower_hygon"),
        key=["n", "LDA", "INCX", "INCY", "UPLO"],
        restore_value=["a_ptr"],
    )(common_ssyr2_kernel.fn.fn)
)

dsyr2_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("dsyr2_hygon"),
        key=["n", "LDA", "INCX", "INCY", "UPLO"],
        restore_value=["a_ptr"],
    )(common_dsyr2_kernel.fn.fn)
)


def ssyr2(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
) -> None:
    _check_ssyr2_args(uplo, n, x, incx, y, incy, A, lda)
    if n == 0:
        return
    uplo = _row_major_uplo(uplo)
    alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_val == 0.0:
        return

    def grid(meta):
        blocks = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (blocks, blocks)

    kernel = (
        ssyr2_large_lower_hygon_kernel
        if n >= 4095 and uplo == CUBLAS_FILL_MODE_UPPER
        else ssyr2_hygon_kernel
    )
    with torch_device_fn.device(A.device):
        kernel[grid](
            A,
            x,
            y,
            alpha_val,
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
        )


def dsyr2(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
) -> None:
    _check_syr2_args(torch.float64, n, x, incx, y, incy, A, lda, uplo)
    if n == 0:
        return
    uplo = _row_major_uplo(uplo)
    alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_val == 0.0:
        return

    def grid(meta):
        blocks = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (blocks, blocks)

    with torch_device_fn.device(A.device):
        dsyr2_hygon_kernel[grid](
            A,
            x,
            y,
            _f64_to_i64(alpha_val),
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
        )


__all__ = ["dsyr2", "ssyr2"]
