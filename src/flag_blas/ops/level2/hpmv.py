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

from flag_blas.runtime import torch_device_fn

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]

CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_FILL_MODE_UPPER = 1


_CHPMV_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_K": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_K": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
]

_ZHPMV_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_K": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_K": 16}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
]

_HPMV_KEY = ["n", "uplo_key"]
_RESTORE = ["y_ptr"]


def _f64_to_i64(v: float) -> int:
    return struct.unpack("<q", struct.pack("<d", v))[0]


def _row_major_uplo(uplo: int) -> int:
    return (
        CUBLAS_FILL_MODE_LOWER
        if uplo == CUBLAS_FILL_MODE_UPPER
        else CUBLAS_FILL_MODE_UPPER
    )


@triton.autotune(configs=_CHPMV_CONFIGS, key=_HPMV_KEY, restore_value=_RESTORE)
@triton.jit
def chpmv_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    beta_r: tl.float32,
    beta_i: tl.float32,
    n,
    INCX,
    INCY,
    uplo_key,
    UPLO: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)

    for kb in tl.range(0, n, BLOCK_K):
        j = kb + offs_k
        j_mask = j < n
        j64 = j.to(tl.int64)

        i_ge_j = rows[:, None] >= j[None, :]
        a_lo64 = tl.where(i_ge_j, j64[None, :], rows64[:, None])
        b_hi64 = tl.where(i_ge_j, rows64[:, None], j64[None, :])
        diag = rows[:, None] == j[None, :]

        if UPLO == 1:
            off = b_hi64 * (b_hi64 + 1) // 2 + a_lo64
            use_conj = rows[:, None] > j[None, :]
        else:
            off = b_hi64 + a_lo64 * (2 * n64 - a_lo64 - 1) // 2
            use_conj = rows[:, None] < j[None, :]

        mask = row_mask[:, None] & j_mask[None, :]
        safe_off = tl.where(mask, off, 0)
        a_off = safe_off * 2
        x_off = j * INCX * 2
        ar = tl.load(ap_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(ap_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + x_off, mask=j_mask, other=0.0)
        xi = tl.load(x_ptr + x_off + 1, mask=j_mask, other=0.0)
        ai = tl.where(use_conj, -ai, ai)
        ai = -ai
        ai = tl.where(diag, 0.0, ai)
        acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
        acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    y_off = rows * INCY * 2
    res_r = alpha_r * acc_r - alpha_i * acc_i
    res_i = alpha_r * acc_i + alpha_i * acc_r
    if not BETA_IS_ZERO:
        yr = tl.load(y_ptr + y_off, mask=row_mask, other=0.0)
        yi = tl.load(y_ptr + y_off + 1, mask=row_mask, other=0.0)
        res_r += beta_r * yr - beta_i * yi
        res_i += beta_r * yi + beta_i * yr
    tl.store(y_ptr + y_off, res_r, mask=row_mask)
    tl.store(y_ptr + y_off + 1, res_i, mask=row_mask)


@triton.autotune(configs=_ZHPMV_CONFIGS, key=_HPMV_KEY, restore_value=_RESTORE)
@triton.jit
def zhpmv_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha_r_int: tl.int64,
    alpha_i_int: tl.int64,
    beta_r_int: tl.int64,
    beta_i_int: tl.int64,
    n,
    INCX,
    INCY,
    uplo_key,
    UPLO: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n
    alpha_r = alpha_r_int.to(tl.float64, bitcast=True)
    alpha_i = alpha_i_int.to(tl.float64, bitcast=True)
    beta_r = beta_r_int.to(tl.float64, bitcast=True)
    beta_i = beta_i_int.to(tl.float64, bitcast=True)
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)

    for kb in tl.range(0, n, BLOCK_K):
        j = kb + offs_k
        j_mask = j < n
        j64 = j.to(tl.int64)

        i_ge_j = rows[:, None] >= j[None, :]
        a_lo64 = tl.where(i_ge_j, j64[None, :], rows64[:, None])
        b_hi64 = tl.where(i_ge_j, rows64[:, None], j64[None, :])
        diag = rows[:, None] == j[None, :]

        if UPLO == 1:
            off = b_hi64 * (b_hi64 + 1) // 2 + a_lo64
            use_conj = rows[:, None] > j[None, :]
        else:
            off = b_hi64 + a_lo64 * (2 * n64 - a_lo64 - 1) // 2
            use_conj = rows[:, None] < j[None, :]

        mask = row_mask[:, None] & j_mask[None, :]
        safe_off = tl.where(mask, off, 0)
        a_off = safe_off * 2
        x_off = j * INCX * 2
        ar = tl.load(ap_ptr + a_off, mask=mask, other=0.0)
        ai = tl.load(ap_ptr + a_off + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + x_off, mask=j_mask, other=0.0)
        xi = tl.load(x_ptr + x_off + 1, mask=j_mask, other=0.0)
        ai = tl.where(use_conj, -ai, ai)
        ai = -ai
        ai = tl.where(diag, 0.0, ai)
        acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
        acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    y_off = rows * INCY * 2
    res_r = alpha_r * acc_r - alpha_i * acc_i
    res_i = alpha_r * acc_i + alpha_i * acc_r
    if not BETA_IS_ZERO:
        yr = tl.load(y_ptr + y_off, mask=row_mask, other=0.0)
        yi = tl.load(y_ptr + y_off + 1, mask=row_mask, other=0.0)
        res_r += beta_r * yr - beta_i * yi
        res_i += beta_r * yi + beta_i * yr
    tl.store(y_ptr + y_off, res_r, mask=row_mask)
    tl.store(y_ptr + y_off + 1, res_i, mask=row_mask)


def _check_common(AP, x, y, uplo, n, incx, incy):
    assert AP.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert AP.device == x.device == y.device
    assert uplo in (CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER)
    assert incx > 0 and incy > 0
    assert n >= 0
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert y.numel() >= 1 + (n - 1) * incy
        assert AP.numel() >= n * (n + 1) // 2


def _complex_scalars(alpha, beta):
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    beta = beta.item() if isinstance(beta, torch.Tensor) else beta
    ar = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    ai = float(alpha.imag) if isinstance(alpha, complex) else 0.0
    br = float(beta.real) if isinstance(beta, complex) else float(beta)
    bi = float(beta.imag) if isinstance(beta, complex) else 0.0
    return ar, ai, br, bi


def _strided_y(y: torch.Tensor, n: int, incy: int) -> torch.Tensor:
    return y[: (n - 1) * incy + 1 : incy]


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

    beta_is_zero = br == 0.0 and bi == 0.0
    AP_real = torch.view_as_real(AP)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    with torch_device_fn.device(AP.device):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        chpmv_kernel[grid](
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
            BETA_IS_ZERO=beta_is_zero,
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

    ar_i = _f64_to_i64(ar)
    ai_i = _f64_to_i64(ai)
    br_i = _f64_to_i64(br)
    bi_i = _f64_to_i64(bi)
    beta_is_zero = br == 0.0 and bi == 0.0
    AP_real = torch.view_as_real(AP)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)

    with torch_device_fn.device(AP.device):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        zhpmv_kernel[grid](
            AP_real,
            x_real,
            y_real,
            ar_i,
            ai_i,
            br_i,
            bi_i,
            n,
            incx,
            incy,
            uplo,
            UPLO=uplo,
            BETA_IS_ZERO=beta_is_zero,
        )
