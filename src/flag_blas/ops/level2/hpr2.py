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

_HPR2_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
]
_HPR2_KEY = ["n", "INCX", "INCY", "UPLO"]
_HPR2_RESTORE = ["ap_ptr"]


def _f64_to_i64(v: float) -> int:
    return struct.unpack("<q", struct.pack("<d", v))[0]


@triton.jit
def _hpr2_row_col(packed_off, packed_size, n, UPLO: tl.constexpr):
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


def _hpr2_grid(n):
    def grid(meta):
        packed_size = n * (n + 1) // 2
        return (triton.cdiv(packed_size, meta["BLOCK_SIZE"]),)

    return grid


@libentry()
@triton.autotune(configs=_HPR2_CONFIGS, key=_HPR2_KEY, restore_value=_HPR2_RESTORE)
@triton.jit
def chpr2_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    n,
    INCX,
    INCY,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    packed_size = n * (n + 1) // 2
    packed_off = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = packed_off < packed_size
    n64 = tl.full((), n, tl.int64)
    safe_off = tl.where(mask, packed_off, 0).to(tl.int64)
    rows, cols = _hpr2_row_col(safe_off, n64 * (n64 + 1) // 2, n, UPLO)

    xrr = tl.load(x_ptr + rows * INCX * 2, mask=mask, other=0.0)
    xri = tl.load(x_ptr + rows * INCX * 2 + 1, mask=mask, other=0.0)
    yrr = tl.load(y_ptr + rows * INCY * 2, mask=mask, other=0.0)
    yri = tl.load(y_ptr + rows * INCY * 2 + 1, mask=mask, other=0.0)
    xcr = tl.load(x_ptr + cols * INCX * 2, mask=mask, other=0.0)
    xci = tl.load(x_ptr + cols * INCX * 2 + 1, mask=mask, other=0.0)
    ycr = tl.load(y_ptr + cols * INCY * 2, mask=mask, other=0.0)
    yci = tl.load(y_ptr + cols * INCY * 2 + 1, mask=mask, other=0.0)

    p1r = xrr * ycr + xri * yci
    p1i = xri * ycr - xrr * yci
    p2r = yrr * xcr + yri * xci
    p2i = yri * xcr - yrr * xci
    update_r = alpha_r * p1r - alpha_i * p1i + alpha_r * p2r + alpha_i * p2i
    update_i = alpha_r * p1i + alpha_i * p1r + alpha_r * p2i - alpha_i * p2r

    ap_off = safe_off * 2
    ar = tl.load(ap_ptr + ap_off, mask=mask, other=0.0)
    ai = tl.load(ap_ptr + ap_off + 1, mask=mask, other=0.0)
    diag = rows == cols
    out_i = tl.where(diag, 0.0, ai + update_i)
    tl.store(ap_ptr + ap_off, ar + update_r, mask=mask)
    tl.store(ap_ptr + ap_off + 1, out_i, mask=mask)


@libentry()
@triton.autotune(configs=_HPR2_CONFIGS, key=_HPR2_KEY, restore_value=_HPR2_RESTORE)
@triton.jit
def zhpr2_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha_r_int: tl.int64,
    alpha_i_int: tl.int64,
    n,
    INCX,
    INCY,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    alpha_r = alpha_r_int.to(tl.float64, bitcast=True)
    alpha_i = alpha_i_int.to(tl.float64, bitcast=True)
    packed_size = n * (n + 1) // 2
    packed_off = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = packed_off < packed_size
    n64 = tl.full((), n, tl.int64)
    safe_off = tl.where(mask, packed_off, 0).to(tl.int64)
    rows, cols = _hpr2_row_col(safe_off, n64 * (n64 + 1) // 2, n, UPLO)

    xrr = tl.load(x_ptr + rows * INCX * 2, mask=mask, other=0.0)
    xri = tl.load(x_ptr + rows * INCX * 2 + 1, mask=mask, other=0.0)
    yrr = tl.load(y_ptr + rows * INCY * 2, mask=mask, other=0.0)
    yri = tl.load(y_ptr + rows * INCY * 2 + 1, mask=mask, other=0.0)
    xcr = tl.load(x_ptr + cols * INCX * 2, mask=mask, other=0.0)
    xci = tl.load(x_ptr + cols * INCX * 2 + 1, mask=mask, other=0.0)
    ycr = tl.load(y_ptr + cols * INCY * 2, mask=mask, other=0.0)
    yci = tl.load(y_ptr + cols * INCY * 2 + 1, mask=mask, other=0.0)

    p1r = xrr * ycr + xri * yci
    p1i = xri * ycr - xrr * yci
    p2r = yrr * xcr + yri * xci
    p2i = yri * xcr - yrr * xci
    update_r = alpha_r * p1r - alpha_i * p1i + alpha_r * p2r + alpha_i * p2i
    update_i = alpha_r * p1i + alpha_i * p1r + alpha_r * p2i - alpha_i * p2r

    ap_off = safe_off * 2
    ar = tl.load(ap_ptr + ap_off, mask=mask, other=0.0)
    ai = tl.load(ap_ptr + ap_off + 1, mask=mask, other=0.0)
    diag = rows == cols
    out_i = tl.where(diag, 0.0, ai + update_i)
    tl.store(ap_ptr + ap_off, ar + update_r, mask=mask)
    tl.store(ap_ptr + ap_off + 1, out_i, mask=mask)


def _check_hpr2_args(dtype, uplo, n, x, incx, y, incy, AP) -> None:
    assert AP.dtype == dtype == x.dtype == y.dtype
    assert AP.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert AP.device == x.device == y.device
    assert uplo in (CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER)
    assert n >= 0
    assert incx > 0 and incy > 0
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert y.numel() >= 1 + (n - 1) * incy
        assert AP.numel() >= n * (n + 1) // 2


def _complex_scalar(alpha: ScalarType):
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    ar = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    ai = float(alpha.imag) if isinstance(alpha, complex) else 0.0
    return ar, ai


def chpr2(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    AP: torch.Tensor,
) -> None:
    _check_hpr2_args(torch.complex64, uplo, n, x, incx, y, incy, AP)
    if n == 0:
        return
    ar, ai = _complex_scalar(alpha)
    if ar == 0.0 and ai == 0.0:
        return

    with torch_device_fn.device(AP.device):
        chpr2_kernel[_hpr2_grid(n)](
            torch.view_as_real(AP),
            torch.view_as_real(x),
            torch.view_as_real(y),
            ar,
            ai,
            n,
            incx,
            incy,
            UPLO=uplo,
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
        zhpr2_kernel[_hpr2_grid(n)](
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
