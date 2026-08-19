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
from flag_blas.ops.level2.hpr import (
    ScalarType,
    _check_hpr_args,
    _f64_to_i64,
    _hpr_grid,
    chpr_kernel,
    zhpr_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

_HPR_KEY = ["n", "INCX", "UPLO"]
_HPR_RESTORE = ["ap_ptr"]


def _make_hygon_hpr_kernel(public_kernel, config_name):
    tuned_kernel = libtuner(
        configs=runtime.get_tuned_config(config_name),
        key=_HPR_KEY,
        restore_value=_HPR_RESTORE,
    )(public_kernel.jit_function)
    return libentry()(tuned_kernel)


chpr_hygon_kernel = _make_hygon_hpr_kernel(chpr_kernel, "chpr_hygon")
zhpr_hygon_kernel = _make_hygon_hpr_kernel(zhpr_kernel, "zhpr_hygon")


def chpr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    AP: torch.Tensor,
) -> None:
    _check_hpr_args(torch.complex64, uplo, n, x, incx, AP)
    if n == 0:
        return
    alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_val == 0.0:
        return

    with torch_device_fn.device(AP.device):
        chpr_hygon_kernel[_hpr_grid(n)](
            torch.view_as_real(AP),
            torch.view_as_real(x),
            alpha_val,
            n,
            incx,
            UPLO=uplo,
        )


def zhpr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    AP: torch.Tensor,
) -> None:
    _check_hpr_args(torch.complex128, uplo, n, x, incx, AP)
    if n == 0:
        return
    alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_val == 0.0:
        return

    with torch_device_fn.device(AP.device):
        zhpr_hygon_kernel[_hpr_grid(n)](
            torch.view_as_real(AP),
            torch.view_as_real(x),
            _f64_to_i64(alpha_val),
            n,
            incx,
            UPLO=uplo,
        )


__all__ = ["chpr", "zhpr"]
