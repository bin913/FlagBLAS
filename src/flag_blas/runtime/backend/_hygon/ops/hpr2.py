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

from flag_blas import runtime
from flag_blas.ops.level2.hpr2 import (
    ScalarType,
    _check_hpr2_args,
    _complex_scalar,
    _f64_to_i64,
    _hpr2_grid,
    zhpr2_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

_HPR2_KEY = ["n", "INCX", "INCY", "UPLO"]
_HPR2_RESTORE = ["ap_ptr"]


zhpr2_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("zhpr2_hygon"),
        key=_HPR2_KEY,
        restore_value=_HPR2_RESTORE,
    )(zhpr2_kernel.jit_function)
)


def zhpr2(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    AP: torch.Tensor,
) -> None:
    _check_hpr2_args(torch.complex128, uplo, n, x, incx, y, incy, AP)
    if n == 0:
        return
    ar, ai = _complex_scalar(alpha)
    if ar == 0.0 and ai == 0.0:
        return

    with torch_device_fn.device(AP.device):
        zhpr2_hygon_kernel[_hpr2_grid(n)](
            torch.view_as_real(AP),
            torch.view_as_real(x),
            torch.view_as_real(y),
            _f64_to_i64(ar),
            _f64_to_i64(ai),
            n,
            incx,
            incy,
            UPLO=uplo,
        )


__all__ = ["zhpr2"]
