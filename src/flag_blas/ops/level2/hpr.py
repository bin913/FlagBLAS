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
import struct
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2._constants import (
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]

_HPR_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
]
_HPR_KEY = ["n", "INCX", "UPLO"]
_HPR_RESTORE = ["ap_ptr"]


def _f64_to_i64(v: float) -> int:
    return struct.unpack("<q", struct.pack("<d", v))[0]


@triton.jit
def _hpr_row_col(packed_off, packed_size, n, UPLO: tl.constexpr):
    if UPLO == 0:
        triangle_off = packed_off
    else:
        triangle_off = packed_size - 1 - packed_off

    major = ((tl.sqrt(8.0 * triangle_off.to(tl.float32) + 1.0) - 1.0) * 0.5).to(
        tl.int64
    )
    major_base = major * (major + 1) // 2
    major = tl.where(major_base > triangle_off, major - 1, major)
    next_base = (major + 1) * (major + 2) // 2
    major = tl.where(next_base <= triangle_off, major + 1, major)
    major_base = major * (major + 1) // 2
    minor = triangle_off - major_base

    if UPLO == 0:
        return major, minor
    n64 = tl.full((), n, tl.int64)
    return n64 - 1 - major, n64 - 1 - minor


def _hpr_grid(n):
    def grid(meta):
        packed_size = n * (n + 1) // 2
        return (triton.cdiv(packed_size, meta["BLOCK_SIZE"]),)

    return grid


@libentry()
@triton.autotune(configs=_HPR_CONFIGS, key=_HPR_KEY, restore_value=_HPR_RESTORE)
@triton.jit
def chpr_kernel(
    ap_ptr,
    x_ptr,
    alpha: tl.float32,
    n,
    INCX,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    packed_size = n * (n + 1) // 2
    packed_off = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = packed_off < packed_size
    n64 = tl.full((), n, tl.int64)
    safe_off = tl.where(mask, packed_off, 0).to(tl.int64)
    rows, cols = _hpr_row_col(safe_off, n64 * (n64 + 1) // 2, n, UPLO)

    xr = tl.load(x_ptr + rows * INCX * 2, mask=mask, other=0.0)
    xi = tl.load(x_ptr + rows * INCX * 2 + 1, mask=mask, other=0.0)
    xcr = tl.load(x_ptr + cols * INCX * 2, mask=mask, other=0.0)
    xci = tl.load(x_ptr + cols * INCX * 2 + 1, mask=mask, other=0.0)
    update_r = alpha * (xr * xcr + xi * xci)
    update_i = alpha * (xi * xcr - xr * xci)

    ap_off = safe_off * 2
    ar = tl.load(ap_ptr + ap_off, mask=mask, other=0.0)
    ai = tl.load(ap_ptr + ap_off + 1, mask=mask, other=0.0)
    diag = rows == cols
    out_i = tl.where(diag, 0.0, ai + update_i)
    tl.store(ap_ptr + ap_off, ar + update_r, mask=mask)
    tl.store(ap_ptr + ap_off + 1, out_i, mask=mask)


@libentry()
@triton.autotune(configs=_HPR_CONFIGS, key=_HPR_KEY, restore_value=_HPR_RESTORE)
@triton.jit
def zhpr_kernel(
    ap_ptr,
    x_ptr,
    alpha_int: tl.int64,
    n,
    INCX,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    alpha = alpha_int.to(tl.float64, bitcast=True)
    packed_size = n * (n + 1) // 2
    packed_off = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = packed_off < packed_size
    n64 = tl.full((), n, tl.int64)
    safe_off = tl.where(mask, packed_off, 0).to(tl.int64)
    rows, cols = _hpr_row_col(safe_off, n64 * (n64 + 1) // 2, n, UPLO)

    xr = tl.load(x_ptr + rows * INCX * 2, mask=mask, other=0.0)
    xi = tl.load(x_ptr + rows * INCX * 2 + 1, mask=mask, other=0.0)
    xcr = tl.load(x_ptr + cols * INCX * 2, mask=mask, other=0.0)
    xci = tl.load(x_ptr + cols * INCX * 2 + 1, mask=mask, other=0.0)
    update_r = alpha * (xr * xcr + xi * xci)
    update_i = alpha * (xi * xcr - xr * xci)

    ap_off = safe_off * 2
    ar = tl.load(ap_ptr + ap_off, mask=mask, other=0.0)
    ai = tl.load(ap_ptr + ap_off + 1, mask=mask, other=0.0)
    diag = rows == cols
    out_i = tl.where(diag, 0.0, ai + update_i)
    tl.store(ap_ptr + ap_off, ar + update_r, mask=mask)
    tl.store(ap_ptr + ap_off + 1, out_i, mask=mask)


def _check_hpr_args(dtype, uplo, n, x, incx, AP) -> None:
    assert AP.dtype == dtype == x.dtype
    assert AP.is_contiguous() and x.is_contiguous()
    assert AP.device == x.device
    assert uplo in (CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER)
    assert n >= 0
    assert incx > 0
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert AP.numel() >= n * (n + 1) // 2


def chpr(
    uplo: int, n: int, alpha: ScalarType, x: torch.Tensor, incx: int, AP: torch.Tensor
) -> None:
    _check_hpr_args(torch.complex64, uplo, n, x, incx, AP)
    if n == 0:
        return
    alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha == 0.0:
        return

    with torch_device_fn.device(AP.device):
        chpr_kernel[_hpr_grid(n)](
            torch.view_as_real(AP),
            torch.view_as_real(x),
            alpha,
            n,
            incx,
            UPLO=uplo,
        )


def zhpr(
    uplo: int, n: int, alpha: ScalarType, x: torch.Tensor, incx: int, AP: torch.Tensor
) -> None:
    _check_hpr_args(torch.complex128, uplo, n, x, incx, AP)
    if n == 0:
        return
    alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_val == 0.0:
        return

    with torch_device_fn.device(AP.device):
        zhpr_kernel[_hpr_grid(n)](
            torch.view_as_real(AP),
            torch.view_as_real(x),
            _f64_to_i64(alpha_val),
            n,
            incx,
            UPLO=uplo,
        )
