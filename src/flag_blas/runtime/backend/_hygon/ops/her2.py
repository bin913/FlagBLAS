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
from flag_blas.ops.level2.her2 import (
    ScalarType,
    _check_her2_args,
    _complex_scalar,
    _f64_to_i64,
    _row_major_uplo,
)
from flag_blas.ops.level2.her2 import cher2_kernel as common_cher2_kernel
from flag_blas.ops.level2.her2 import zher2_kernel as common_zher2_kernel
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

_HER2_KEY = ["n", "LDA", "INCX", "INCY", "UPLO"]
_HER2_RESTORE = ["a_ptr"]


cher2_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cher2_hygon"),
        key=_HER2_KEY,
        restore_value=_HER2_RESTORE,
    )(common_cher2_kernel.fn.fn)
)

cher2_mid_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cher2_mid_hygon"),
        key=_HER2_KEY,
        restore_value=_HER2_RESTORE,
    )(common_cher2_kernel.fn.fn)
)

cher2_large_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cher2_large_hygon"),
        key=_HER2_KEY,
        restore_value=_HER2_RESTORE,
    )(common_cher2_kernel.fn.fn)
)

zher2_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("zher2_hygon"),
        key=_HER2_KEY,
        restore_value=_HER2_RESTORE,
    )(common_zher2_kernel.fn.fn)
)


def cher2(
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
    _check_her2_args(torch.complex64, uplo, n, x, incx, y, incy, A, lda)
    if n == 0:
        return
    uplo = _row_major_uplo(uplo)
    ar, ai = _complex_scalar(alpha)
    if ar == 0.0 and ai == 0.0:
        return

    def grid(meta):
        blocks = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (blocks, blocks)

    if 1024 <= n <= 1536:
        kernel = cher2_mid_hygon_kernel
    elif n >= 3900:
        kernel = cher2_large_hygon_kernel
    else:
        kernel = cher2_hygon_kernel
    with torch_device_fn.device(A.device):
        kernel[grid](
            torch.view_as_real(A),
            torch.view_as_real(x),
            torch.view_as_real(y),
            ar,
            ai,
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
        )


def zher2(
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
    _check_her2_args(torch.complex128, uplo, n, x, incx, y, incy, A, lda)
    if n == 0:
        return
    uplo = _row_major_uplo(uplo)
    ar, ai = _complex_scalar(alpha)
    if ar == 0.0 and ai == 0.0:
        return

    def grid(meta):
        blocks = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (blocks, blocks)

    with torch_device_fn.device(A.device):
        zher2_hygon_kernel[grid](
            torch.view_as_real(A),
            torch.view_as_real(x),
            torch.view_as_real(y),
            _f64_to_i64(ar),
            _f64_to_i64(ai),
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
        )


__all__ = ["cher2", "zher2"]
