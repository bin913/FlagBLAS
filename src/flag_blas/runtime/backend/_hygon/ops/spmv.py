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
from flag_blas.ops.level2.spmv import (
    ScalarType,
    _check_common,
    _f64_to_i64,
    _row_major_uplo,
    _strided_y,
)
from flag_blas.ops.level2.spmv import dspmv_kernel as common_dspmv_kernel
from flag_blas.ops.level2.spmv import sspmv_kernel as common_sspmv_kernel
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

sspmv_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sspmv_hygon"),
        key=["n", "uplo_key"],
        restore_value=["y_ptr"],
    )(common_sspmv_kernel.fn)
)

dspmv_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("dspmv_hygon"),
        key=["n", "uplo_key"],
        restore_value=["y_ptr"],
    )(common_dspmv_kernel.fn)
)


def sspmv(
    uplo: int,
    n: int,
    alpha: ScalarType,
    AP: torch.Tensor,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert AP.dtype == torch.float32 == x.dtype == y.dtype
    _check_common(AP, x, y, uplo, n, incx, incy)
    if n == 0:
        return
    uplo = _row_major_uplo(uplo)

    alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta_val = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    y_view = _strided_y(y, n, incy)
    if alpha_val == 0.0:
        if beta_val == 0.0:
            y_view.zero_()
        elif beta_val != 1.0:
            y_view.mul_(beta_val)
        return

    with torch_device_fn.device(AP.device):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        sspmv_hygon_kernel[grid](
            AP,
            x,
            y,
            alpha_val,
            beta_val,
            n,
            incx,
            incy,
            uplo,
            UPLO=uplo,
            BETA_IS_ZERO=beta_val == 0.0,
        )


def dspmv(
    uplo: int,
    n: int,
    alpha: ScalarType,
    AP: torch.Tensor,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert AP.dtype == torch.float64 == x.dtype == y.dtype
    _check_common(AP, x, y, uplo, n, incx, incy)
    if n == 0:
        return
    uplo = _row_major_uplo(uplo)

    alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta_val = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    y_view = _strided_y(y, n, incy)
    if alpha_val == 0.0:
        if beta_val == 0.0:
            y_view.zero_()
        elif beta_val != 1.0:
            y_view.mul_(beta_val)
        return

    with torch_device_fn.device(AP.device):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        dspmv_hygon_kernel[grid](
            AP,
            x,
            y,
            _f64_to_i64(alpha_val),
            _f64_to_i64(beta_val),
            n,
            incx,
            incy,
            uplo,
            UPLO=uplo,
            BETA_IS_ZERO=beta_val == 0.0,
        )


__all__ = ["dspmv", "sspmv"]
