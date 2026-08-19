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
from flag_blas.ops.level2.hpmv import (
    ScalarType,
    _check_common,
    _complex_scalars,
    _f64_to_i64,
    _row_major_uplo,
    _strided_y,
)
from flag_blas.ops.level2.hpmv import chpmv_kernel as common_chpmv_kernel
from flag_blas.ops.level2.hpmv import zhpmv_kernel as common_zhpmv_kernel
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

chpmv_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("chpmv_hygon"),
        key=["n", "uplo_key"],
        restore_value=["y_ptr"],
    )(common_chpmv_kernel.fn)
)

zhpmv_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("zhpmv_hygon"),
        key=["n", "uplo_key"],
        restore_value=["y_ptr"],
    )(common_zhpmv_kernel.fn)
)


def chpmv(
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
    assert AP.dtype == torch.complex64 == x.dtype == y.dtype
    _check_common(AP, x, y, uplo, n, incx, incy)
    if n == 0:
        return
    uplo = _row_major_uplo(uplo)

    ar, ai, br, bi = _complex_scalars(alpha, beta)
    y_view = _strided_y(y, n, incy)
    if ar == 0.0 and ai == 0.0:
        if br == 0.0 and bi == 0.0:
            y_view.zero_()
        elif br != 1.0 or bi != 0.0:
            y_view.mul_(complex(br, bi))
        return

    AP_real = torch.view_as_real(AP)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    with torch_device_fn.device(AP.device):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        chpmv_hygon_kernel[grid](
            AP_real,
            x_real,
            y_real,
            ar,
            ai,
            br,
            bi,
            n,
            incx,
            incy,
            uplo,
            UPLO=uplo,
            BETA_IS_ZERO=br == 0.0 and bi == 0.0,
        )


def zhpmv(
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
    assert AP.dtype == torch.complex128 == x.dtype == y.dtype
    _check_common(AP, x, y, uplo, n, incx, incy)
    if n == 0:
        return
    uplo = _row_major_uplo(uplo)

    ar, ai, br, bi = _complex_scalars(alpha, beta)
    y_view = _strided_y(y, n, incy)
    if ar == 0.0 and ai == 0.0:
        if br == 0.0 and bi == 0.0:
            y_view.zero_()
        elif br != 1.0 or bi != 0.0:
            y_view.mul_(complex(br, bi))
        return

    AP_real = torch.view_as_real(AP)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    with torch_device_fn.device(AP.device):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        zhpmv_hygon_kernel[grid](
            AP_real,
            x_real,
            y_real,
            _f64_to_i64(ar),
            _f64_to_i64(ai),
            _f64_to_i64(br),
            _f64_to_i64(bi),
            n,
            incx,
            incy,
            uplo,
            UPLO=uplo,
            BETA_IS_ZERO=br == 0.0 and bi == 0.0,
        )


__all__ = ["chpmv", "zhpmv"]
