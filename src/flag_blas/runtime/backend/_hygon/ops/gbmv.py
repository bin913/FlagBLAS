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

import struct

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2._constants import CUBLAS_OP_C, CUBLAS_OP_N, CUBLAS_OP_T
from flag_blas.ops.level2.gbmv import (
    ScalarType,
    _band_bucket,
    _check_common,
    _complex_scalars,
    _f64_to_i64,
    _pick_split_band,
    _row_major_gbmv_args,
)
from flag_blas.ops.level2.gbmv import cgbmv as common_cgbmv
from flag_blas.ops.level2.gbmv import cgbmv_n_kernel as common_cgbmv_n_kernel
from flag_blas.ops.level2.gbmv import cgbmv_t_kernel as common_cgbmv_t_kernel
from flag_blas.ops.level2.gbmv import dgbmv as common_dgbmv
from flag_blas.ops.level2.gbmv import dgbmv_n_kernel as common_dgbmv_n_kernel
from flag_blas.ops.level2.gbmv import (
    dgbmv_n_split_band_kernel as common_dgbmv_n_split_band_kernel,
)
from flag_blas.ops.level2.gbmv import (
    dgbmv_t_split_band_kernel as common_dgbmv_t_split_band_kernel,
)
from flag_blas.ops.level2.gbmv import zgbmv as common_zgbmv
from flag_blas.ops.level2.gbmv import zgbmv_t_kernel as common_zgbmv_t_kernel
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

_DGBMV_KEY = [
    "m",
    "n",
    "BAND",
    "num_band_splits",
    "out_len",
    "band_bucket",
]


def _f32_to_i32(value: float) -> int:
    return struct.unpack("<i", struct.pack("<f", value))[0]


@triton.jit
def complex_gbmv_n_active_hygon_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r_arg: tl.int64,
    alpha_i_arg: tl.int64,
    beta_r_arg: tl.int64,
    beta_i_arg: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    CONJ: tl.constexpr,
    FP64: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    if FP64:
        alpha_r = alpha_r_arg.to(tl.float64, bitcast=True)
        alpha_i = alpha_i_arg.to(tl.float64, bitcast=True)
        beta_r = beta_r_arg.to(tl.float64, bitcast=True)
        beta_i = beta_i_arg.to(tl.float64, bitcast=True)
        acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
        acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    else:
        alpha_r = alpha_r_arg.to(tl.int32).to(tl.float32, bitcast=True)
        alpha_i = alpha_i_arg.to(tl.int32).to(tl.float32, bitcast=True)
        beta_r = beta_r_arg.to(tl.int32).to(tl.float32, bitcast=True)
        beta_i = beta_i_arg.to(tl.int32).to(tl.float32, bitcast=True)
        acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    active_end = tl.minimum(m, n + KL)
    if pid * BLOCK_SIZE_M < active_end:
        offsets = tl.arange(0, BAND_TILE)
        for base in tl.range(0, BAND, BAND_TILE):
            d_idx = base + offsets
            d = d_idx - KL
            cols = rows[:, None] + d[None, :]
            band_rows = KU - d
            mask = (
                row_mask[:, None] & (d_idx[None, :] < BAND) & (cols >= 0) & (cols < n)
            )
            safe_cols = tl.where(mask, cols, 0)
            a_offsets = (band_rows[None, :] + safe_cols * LDA) * 2
            x_offsets = safe_cols * INCX * 2
            ar = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
            ai = tl.load(a_ptr + a_offsets + 1, mask=mask, other=0.0)
            xr = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
            xi = tl.load(x_ptr + x_offsets + 1, mask=mask, other=0.0)
            if CONJ:
                ai = -ai
            acc_r += tl.sum(ar * xr - ai * xi, axis=1)
            acc_i += tl.sum(ar * xi + ai * xr, axis=1)

    y_offsets = rows * INCY * 2
    result_r = alpha_r * acc_r - alpha_i * acc_i
    result_i = alpha_r * acc_i + alpha_i * acc_r
    if not BETA_IS_ZERO:
        yr = tl.load(y_ptr + y_offsets, mask=row_mask, other=0.0)
        yi = tl.load(y_ptr + y_offsets + 1, mask=row_mask, other=0.0)
        result_r += beta_r * yr - beta_i * yi
        result_i += beta_r * yi + beta_i * yr
    tl.store(y_ptr + y_offsets, result_r, mask=row_mask)
    tl.store(y_ptr + y_offsets + 1, result_i, mask=row_mask)


@triton.jit
def zgbmv_n_vector_load_hygon_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r_arg: tl.int64,
    alpha_i_arg: tl.int64,
    beta_r_arg: tl.int64,
    beta_i_arg: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    alpha_r = alpha_r_arg.to(tl.float64, bitcast=True)
    alpha_i = alpha_i_arg.to(tl.float64, bitcast=True)
    beta_r = beta_r_arg.to(tl.float64, bitcast=True)
    beta_i = beta_i_arg.to(tl.float64, bitcast=True)
    rows = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    offsets = tl.arange(0, BAND_TILE)
    components = tl.arange(0, 2)
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    for base in tl.range(0, BAND, BAND_TILE):
        d_idx = base + offsets
        d = d_idx - KL
        cols = rows[:, None] + d[None, :]
        band_rows = KU - d
        mask = row_mask[:, None] & (d_idx[None, :] < BAND) & (cols >= 0) & (cols < n)
        safe_cols = tl.where(mask, cols, 0)
        a_offsets = (band_rows[None, :] + safe_cols * LDA) * 2
        x_offsets = safe_cols * INCX * 2
        a_values = tl.load(
            a_ptr + a_offsets[:, :, None] + components[None, None, :],
            mask=mask[:, :, None],
            other=0.0,
        )
        x_values = tl.load(
            x_ptr + x_offsets[:, :, None] + components[None, None, :],
            mask=mask[:, :, None],
            other=0.0,
        )
        ar, ai = tl.split(a_values)
        xr, xi = tl.split(x_values)
        if CONJ:
            ai = -ai
        acc_r += tl.sum(ar * xr - ai * xi, axis=1)
        acc_i += tl.sum(ar * xi + ai * xr, axis=1)
    y_offsets = rows * INCY * 2
    result_r = alpha_r * acc_r - alpha_i * acc_i
    result_i = alpha_r * acc_i + alpha_i * acc_r
    if not BETA_IS_ZERO:
        yr = tl.load(y_ptr + y_offsets, mask=row_mask, other=0.0)
        yi = tl.load(y_ptr + y_offsets + 1, mask=row_mask, other=0.0)
        result_r += beta_r * yr - beta_i * yi
        result_i += beta_r * yi + beta_i * yr
    tl.store(y_ptr + y_offsets, result_r, mask=row_mask)
    tl.store(y_ptr + y_offsets + 1, result_i, mask=row_mask)


@triton.jit
def zgbmv_scale_y_hygon_kernel(
    y_ptr,
    beta_r_arg: tl.int64,
    beta_i_arg: tl.int64,
    out_len,
    INCY,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_len
    beta_r = beta_r_arg.to(tl.float64, bitcast=True)
    beta_i = beta_i_arg.to(tl.float64, bitcast=True)
    y_offsets = offsets * INCY * 2
    yr = tl.load(y_ptr + y_offsets, mask=mask, other=0.0)
    yi = tl.load(y_ptr + y_offsets + 1, mask=mask, other=0.0)
    tl.store(y_ptr + y_offsets, beta_r * yr - beta_i * yi, mask=mask)
    tl.store(y_ptr + y_offsets + 1, beta_r * yi + beta_i * yr, mask=mask)


@triton.jit
def zgbmv_grouped_scatter_hygon_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r_arg: tl.int64,
    alpha_i_arg: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    BAND,
    CONJ: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    OUTPUT_TILE: tl.constexpr,
):
    row_local = tl.arange(0, BLOCK_ROWS)
    out_local = tl.arange(0, OUTPUT_TILE)
    row_base = tl.program_id(0) * BLOCK_ROWS
    q = tl.program_id(1) * OUTPUT_TILE + out_local
    rows = row_base + row_local
    cols = row_base - KL + q
    band_cols = q[None, :] - row_local[:, None]
    mask = (
        (rows[:, None] < m)
        & (q[None, :] < BAND + BLOCK_ROWS - 1)
        & (band_cols >= 0)
        & (band_cols < BAND)
        & (cols[None, :] >= 0)
        & (cols[None, :] < n)
    )
    components = tl.arange(0, 2)
    a_offsets = (rows[:, None] * LDA + band_cols) * 2
    a_values = tl.load(
        a_ptr + a_offsets[:, :, None] + components[None, None, :],
        mask=mask[:, :, None],
        other=0.0,
    )
    x_values = tl.load(
        x_ptr + rows[:, None] * INCX * 2 + components[None, :],
        mask=(rows < m)[:, None],
        other=0.0,
    )
    ar, ai = tl.split(a_values)
    xr, xi = tl.split(x_values)
    if CONJ:
        ai = -ai
    prod_r = ar * xr[:, None] - ai * xi[:, None]
    prod_i = ar * xi[:, None] + ai * xr[:, None]
    sum_r = tl.sum(prod_r, axis=0)
    sum_i = tl.sum(prod_i, axis=0)
    alpha_r = alpha_r_arg.to(tl.float64, bitcast=True)
    alpha_i = alpha_i_arg.to(tl.float64, bitcast=True)
    out_mask = (q < BAND + BLOCK_ROWS - 1) & (cols >= 0) & (cols < n)
    y_offsets = cols * INCY * 2
    tl.atomic_add(
        y_ptr + y_offsets,
        alpha_r * sum_r - alpha_i * sum_i,
        mask=out_mask,
        sem="relaxed",
    )
    tl.atomic_add(
        y_ptr + y_offsets + 1,
        alpha_r * sum_i + alpha_i * sum_r,
        mask=out_mask,
        sem="relaxed",
    )


_complex_gbmv_n_active_hygon_jit = complex_gbmv_n_active_hygon_kernel
complex_gbmv_n_active_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("complex_gbmv_n_active_hygon"),
        key=["m", "n", "KL", "BAND", "CONJ", "FP64", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(_complex_gbmv_n_active_hygon_jit)
)
complex_gbmv_n_active_small_hygon_kernel = libentry()(_complex_gbmv_n_active_hygon_jit)


@triton.jit
def dgbmv_t_tiled_hygon_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    beta_int: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KU,
    BAND,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_mask = cols < n
    alpha = alpha_int.to(tl.float64, bitcast=True)
    beta = beta_int.to(tl.float64, bitcast=True)
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)

    band_offsets = tl.arange(0, BAND_TILE)
    for band_base in tl.range(0, BAND, BAND_TILE):
        band_rows = band_base + band_offsets
        input_rows = cols[:, None] + band_rows[None, :] - KU
        mask = (
            col_mask[:, None]
            & (band_rows[None, :] < BAND)
            & (input_rows >= 0)
            & (input_rows < m)
        )
        safe_rows = tl.where(mask, input_rows, 0)
        a_offsets = band_rows[None, :] + cols[:, None] * LDA
        a_values = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
        x_values = tl.load(x_ptr + safe_rows * INCX, mask=mask, other=0.0)
        acc += tl.sum(a_values * x_values, axis=1)

    y_ptrs = y_ptr + cols * INCY
    if BETA_IS_ZERO:
        output = alpha * acc
    else:
        old_y = tl.load(y_ptrs, mask=col_mask, other=0.0)
        output = alpha * acc + beta * old_y
    tl.store(y_ptrs, output, mask=col_mask)


dgbmv_t_tiled_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("dgbmv_t_tiled_hygon"),
        key=["m", "n", "KU", "BAND", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(dgbmv_t_tiled_hygon_kernel)
)

cgbmv_t_tiled_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cgbmv_t_tiled_hygon"),
        key=["m", "n", "BAND", "CONJ", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(common_cgbmv_t_kernel.fn)
)


dgbmv_n_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("dgbmv_n_hygon"),
        key=["m", "n", "BAND", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(common_dgbmv_n_kernel.fn)
)


@triton.jit
def dgbmv_n_tiled_hygon_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    beta_int: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    out_len,
    band_bucket,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    alpha = alpha_int.to(tl.float64, bitcast=True)
    beta = beta_int.to(tl.float64, bitcast=True)
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    band_offsets = tl.arange(0, BAND_TILE)

    for band_base in tl.range(0, BAND, BAND_TILE):
        d_idx = band_base + band_offsets
        d = d_idx - KL
        cols = rows[:, None] + d[None, :]
        band_rows = KU - d
        mask = row_mask[:, None] & (d_idx[None, :] < BAND) & (cols >= 0) & (cols < n)
        safe_cols = tl.where(mask, cols, 0)
        a_offsets = band_rows[None, :] + safe_cols * LDA
        a_values = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
        x_values = tl.load(x_ptr + safe_cols * INCX, mask=mask, other=0.0)
        acc += tl.sum(a_values * x_values, axis=1)

    y_ptrs = y_ptr + rows * INCY
    if BETA_IS_ZERO:
        output = alpha * acc
    else:
        old_y = tl.load(y_ptrs, mask=row_mask, other=0.0)
        output = alpha * acc + beta * old_y
    tl.store(y_ptrs, output, mask=row_mask)


_dgbmv_n_tiled_hygon_jit = dgbmv_n_tiled_hygon_kernel
dgbmv_n_tiled_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("dgbmv_n_tiled_hygon"),
        key=["m", "n", "BAND", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(_dgbmv_n_tiled_hygon_jit)
)
dgbmv_n_tiled_fixed_hygon_kernel = libentry()(_dgbmv_n_tiled_hygon_jit)


@triton.jit
def dgbmv_n_row_hygon_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    beta_int: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    alpha = alpha_int.to(tl.float64, bitcast=True)
    beta = beta_int.to(tl.float64, bitcast=True)
    acc = tl.zeros((), dtype=tl.float64)

    for base in tl.range(0, BAND, BLOCK_K):
        d_idx = base + offsets
        cols = row + d_idx - KL
        band_rows = KU + KL - d_idx
        mask = (d_idx < BAND) & (cols >= 0) & (cols < n)
        safe_cols = tl.where(mask, cols, 0)
        a_values = tl.load(
            a_ptr + band_rows + safe_cols * LDA,
            mask=mask,
            other=0.0,
        )
        x_values = tl.load(
            x_ptr + safe_cols * INCX,
            mask=mask,
            other=0.0,
        )
        acc += tl.sum(a_values * x_values, axis=0)

    y_ptr += row * INCY
    result = alpha * acc
    if not BETA_IS_ZERO:
        result += beta * tl.load(y_ptr)
    tl.store(y_ptr, result)


dgbmv_n_row_hygon_kernel = libentry()(dgbmv_n_row_hygon_kernel)


@triton.jit
def dgbmv_n_scatter_hygon_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    BAND,
    BLOCK_COLS: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    col_mask = cols < n
    alpha = alpha_int.to(tl.float64, bitcast=True)
    band_offsets = tl.arange(0, BAND_TILE)
    x_values = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)

    for band_base in tl.range(0, BAND, BAND_TILE):
        band_rows = band_base + band_offsets
        rows = cols[:, None] + KL - band_rows[None, :]
        mask = (
            col_mask[:, None] & (band_rows[None, :] < BAND) & (rows >= 0) & (rows < m)
        )
        a_offsets = BAND - 1 - band_rows[None, :] + cols[:, None] * LDA
        a_values = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
        tl.atomic_add(
            y_ptr + rows * INCY,
            alpha * a_values * x_values[:, None],
            mask=mask,
            sem="relaxed",
        )


dgbmv_n_scatter_hygon_kernel = libentry()(dgbmv_n_scatter_hygon_kernel)


@triton.jit
def dgbmv_scale_y_hygon_kernel(
    y_ptr,
    beta_int: tl.int64,
    out_len,
    INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_len
    y_ptrs = y_ptr + offsets * INCY
    if BETA_IS_ZERO:
        values = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    else:
        beta = beta_int.to(tl.float64, bitcast=True)
        values = beta * tl.load(y_ptrs, mask=mask, other=0.0)
    tl.store(y_ptrs, values, mask=mask)


dgbmv_scale_y_hygon_kernel = libentry()(dgbmv_scale_y_hygon_kernel)


@triton.jit
def cgbmv_n_packed_hygon_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    beta_r: tl.float32,
    beta_i: tl.float32,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    CONJ: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    offsets = tl.arange(0, BAND_TILE)
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    a_ptr_i64 = a_ptr.to(tl.pointer_type(tl.int64))
    x_ptr_i64 = x_ptr.to(tl.pointer_type(tl.int64))
    for base_idx in tl.range(0, BAND, BAND_TILE):
        d_idx = base_idx + offsets
        d = d_idx - KL
        cols = rows[:, None] + d[None, :]
        band_rows = KU - d
        mask = row_mask[:, None] & (d_idx[None, :] < BAND) & (cols >= 0) & (cols < n)
        safe_cols = tl.where(mask, cols, 0)
        a_bits = tl.load(
            a_ptr_i64 + band_rows[None, :] + safe_cols * LDA,
            mask=mask,
            other=0,
        )
        x_bits = tl.load(
            x_ptr_i64 + safe_cols * INCX,
            mask=mask,
            other=0,
        )
        ar = a_bits.to(tl.int32).to(tl.float32, bitcast=True)
        ai = (a_bits >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        xr = x_bits.to(tl.int32).to(tl.float32, bitcast=True)
        xi = (x_bits >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        if CONJ:
            ai = -ai
        acc_r += tl.sum(ar * xr - ai * xi, axis=1)
        acc_i += tl.sum(ar * xi + ai * xr, axis=1)
    y_offsets = rows * INCY * 2
    result_r = alpha_r * acc_r - alpha_i * acc_i
    result_i = alpha_r * acc_i + alpha_i * acc_r
    yr = tl.load(y_ptr + y_offsets, mask=row_mask, other=0.0)
    yi = tl.load(y_ptr + y_offsets + 1, mask=row_mask, other=0.0)
    result_r += beta_r * yr - beta_i * yi
    result_i += beta_r * yi + beta_i * yr
    tl.store(y_ptr + y_offsets, result_r, mask=row_mask)
    tl.store(y_ptr + y_offsets + 1, result_i, mask=row_mask)


cgbmv_n_packed_hygon_kernel = libentry()(cgbmv_n_packed_hygon_kernel)


@triton.jit
def cgbmv_n_x_packed_hygon_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    beta_r: tl.float32,
    beta_i: tl.float32,
    m: tl.constexpr,
    n: tl.constexpr,
    LDA: tl.constexpr,
    INCX,
    INCY,
    KL: tl.constexpr,
    KU: tl.constexpr,
    BAND: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    offsets = tl.arange(0, BAND_TILE)
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    x_ptr_i64 = x_ptr.to(tl.pointer_type(tl.int64))
    for base_idx in tl.range(0, BAND, BAND_TILE):
        d_idx = base_idx + offsets
        d = d_idx - KL
        cols = rows[:, None] + d[None, :]
        band_rows = KU - d
        mask = row_mask[:, None] & (d_idx[None, :] < BAND) & (cols >= 0) & (cols < n)
        safe_cols = tl.where(mask, cols, 0)
        a_offsets = (band_rows[None, :] + safe_cols * LDA) * 2
        ar = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_offsets + 1, mask=mask, other=0.0)
        x_bits = tl.load(
            x_ptr_i64 + safe_cols * INCX,
            mask=mask,
            other=0,
        )
        xr = x_bits.to(tl.int32).to(tl.float32, bitcast=True)
        xi = (x_bits >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        ai = -ai
        acc_r += tl.sum(ar * xr - ai * xi, axis=1)
        acc_i += tl.sum(ar * xi + ai * xr, axis=1)
    y_offsets = rows * INCY * 2
    result_r = alpha_r * acc_r - alpha_i * acc_i
    result_i = alpha_r * acc_i + alpha_i * acc_r
    if not BETA_IS_ZERO:
        yr = tl.load(y_ptr + y_offsets, mask=row_mask, other=0.0)
        yi = tl.load(y_ptr + y_offsets + 1, mask=row_mask, other=0.0)
        result_r += beta_r * yr - beta_i * yi
        result_i += beta_r * yi + beta_i * yr
    tl.store(y_ptr + y_offsets, result_r, mask=row_mask)
    tl.store(y_ptr + y_offsets + 1, result_i, mask=row_mask)


cgbmv_n_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cgbmv_n_hygon"),
        key=["m", "n", "BAND", "CONJ", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(common_cgbmv_n_kernel.fn)
)


zgbmv_t_tiled_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("zgbmv_t_tiled_hygon"),
        key=["m", "n", "BAND", "CONJ", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(common_zgbmv_t_kernel.fn)
)

zgbmv_t_wide_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("zgbmv_t_wide_hygon"),
        key=["m", "n", "BAND", "CONJ", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(common_zgbmv_t_kernel.fn)
)


@triton.jit
def zgbmv_t_pair_hygon_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r_int: tl.int64,
    alpha_i_int: tl.int64,
    beta_r_int: tl.int64,
    beta_i_int: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KU,
    BAND,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BAND_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_mask = cols < n
    alpha_r = alpha_r_int.to(tl.float64, bitcast=True)
    alpha_i = alpha_i_int.to(tl.float64, bitcast=True)
    beta_r = beta_r_int.to(tl.float64, bitcast=True)
    beta_i = beta_i_int.to(tl.float64, bitcast=True)
    acc_r = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    band_offsets = tl.arange(0, BAND_TILE)
    pair_offsets = tl.arange(0, 2)
    a_ptr_i64 = a_ptr.to(tl.pointer_type(tl.int64))
    x_ptr_i64 = x_ptr.to(tl.pointer_type(tl.int64))

    for band_base in tl.range(0, BAND, BAND_TILE):
        band_rows = band_base + band_offsets
        input_rows = cols[:, None] + band_rows[None, :] - KU
        mask = (
            col_mask[:, None]
            & (band_rows[None, :] < BAND)
            & (input_rows >= 0)
            & (input_rows < m)
        )
        safe_rows = tl.where(mask, input_rows, 0)
        a_offsets = band_rows[None, :] + cols[:, None] * LDA
        x_offsets = safe_rows * INCX
        pair_mask = tl.broadcast_to(mask[:, :, None], (BLOCK_SIZE_M, BAND_TILE, 2))
        a_pairs = tl.load(
            a_ptr_i64 + a_offsets[:, :, None] * 2 + pair_offsets[None, None, :],
            mask=pair_mask,
            other=0,
            eviction_policy="evict_first",
        )
        x_pairs = tl.load(
            x_ptr_i64 + x_offsets[:, :, None] * 2 + pair_offsets[None, None, :],
            mask=pair_mask,
            other=0,
            eviction_policy="evict_last",
        )
        pair_is_real = tl.broadcast_to(
            pair_offsets[None, None, :] == 0, (BLOCK_SIZE_M, BAND_TILE, 2)
        )
        a_values = a_pairs.to(tl.float64, bitcast=True)
        x_values = x_pairs.to(tl.float64, bitcast=True)
        ar = tl.sum(tl.where(pair_is_real, a_values, 0.0), axis=2)
        ai = tl.sum(tl.where(pair_is_real, 0.0, a_values), axis=2)
        xr = tl.sum(tl.where(pair_is_real, x_values, 0.0), axis=2)
        xi = tl.sum(tl.where(pair_is_real, 0.0, x_values), axis=2)
        if CONJ:
            ai = -ai
        acc_r += tl.sum(ar * xr - ai * xi, axis=1)
        acc_i += tl.sum(ar * xi + ai * xr, axis=1)

    y_offsets = cols * INCY * 2
    result_r = alpha_r * acc_r - alpha_i * acc_i
    result_i = alpha_r * acc_i + alpha_i * acc_r
    if not BETA_IS_ZERO:
        yr = tl.load(y_ptr + y_offsets, mask=col_mask, other=0.0)
        yi = tl.load(y_ptr + y_offsets + 1, mask=col_mask, other=0.0)
        result_r += beta_r * yr - beta_i * yi
        result_i += beta_r * yi + beta_i * yr
    tl.store(y_ptr + y_offsets, result_r, mask=col_mask)
    tl.store(y_ptr + y_offsets + 1, result_i, mask=col_mask)


zgbmv_t_pair_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("zgbmv_t_pair_hygon"),
        key=["m", "n", "KU", "BAND", "CONJ", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(zgbmv_t_pair_hygon_kernel)
)


@triton.jit
def cgbmv_n_small_hygon_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    beta_r: tl.float32,
    beta_i: tl.float32,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KU,
    BAND,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    SPLIT_LANES: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    split_offsets = tl.arange(0, SPLIT_LANES)
    acc_r = tl.zeros((BLOCK_SIZE_M, SPLIT_LANES), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M, SPLIT_LANES), dtype=tl.float32)
    a_ptr_i64 = a_ptr.to(tl.pointer_type(tl.int64))
    x_ptr_i64 = x_ptr.to(tl.pointer_type(tl.int64))

    for col_base in tl.range(0, n, SPLIT_LANES):
        cols = col_base + split_offsets
        col_mask = cols < n
        band_rows = KU + rows[:, None] - cols[None, :]
        mask = (
            row_mask[:, None]
            & col_mask[None, :]
            & (band_rows >= 0)
            & (band_rows < BAND)
        )
        safe_band_rows = tl.where(mask, band_rows, 0)
        safe_cols = tl.where(col_mask, cols, 0)
        a_offsets = safe_band_rows + safe_cols[None, :] * LDA
        a_bits = tl.load(a_ptr_i64 + a_offsets, mask=mask, other=0)
        x_bits = tl.load(
            x_ptr_i64 + safe_cols * INCX,
            mask=col_mask,
            other=0,
            eviction_policy="evict_last",
        )
        ar = a_bits.to(tl.int32).to(tl.float32, bitcast=True)
        ai = (a_bits >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        xr = x_bits.to(tl.int32).to(tl.float32, bitcast=True)
        xi = (x_bits >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        acc_r += ar * xr[None, :] - ai * xi[None, :]
        acc_i += ar * xi[None, :] + ai * xr[None, :]

    sum_r = tl.sum(acc_r, axis=1)
    sum_i = tl.sum(acc_i, axis=1)
    y_offsets = rows * INCY * 2
    result_r = alpha_r * sum_r - alpha_i * sum_i
    result_i = alpha_r * sum_i + alpha_i * sum_r
    if not BETA_IS_ZERO:
        yr = tl.load(y_ptr + y_offsets, mask=row_mask, other=0.0)
        yi = tl.load(y_ptr + y_offsets + 1, mask=row_mask, other=0.0)
        result_r += beta_r * yr - beta_i * yi
        result_i += beta_r * yi + beta_i * yr
    tl.store(y_ptr + y_offsets, result_r, mask=row_mask)
    tl.store(y_ptr + y_offsets + 1, result_i, mask=row_mask)


cgbmv_n_small_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cgbmv_n_small_hygon"),
        key=["m", "n", "LDA", "INCX", "INCY", "KU", "BAND", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(cgbmv_n_small_hygon_kernel)
)


@triton.jit
def zgbmv_n_small_hygon_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r_int: tl.int64,
    alpha_i_int: tl.int64,
    beta_r_int: tl.int64,
    beta_i_int: tl.int64,
    m,
    n,
    LDA,
    INCX,
    INCY,
    KU,
    BAND,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    SPLIT_LANES: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    split_offsets = tl.arange(0, SPLIT_LANES)
    alpha_r = alpha_r_int.to(tl.float64, bitcast=True)
    alpha_i = alpha_i_int.to(tl.float64, bitcast=True)
    beta_r = beta_r_int.to(tl.float64, bitcast=True)
    beta_i = beta_i_int.to(tl.float64, bitcast=True)
    acc_r = tl.zeros((BLOCK_SIZE_M, SPLIT_LANES), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_SIZE_M, SPLIT_LANES), dtype=tl.float64)
    a_ptr_i64 = a_ptr.to(tl.pointer_type(tl.int64))
    x_ptr_i64 = x_ptr.to(tl.pointer_type(tl.int64))

    for col_base in tl.range(0, n, SPLIT_LANES):
        cols = col_base + split_offsets
        col_mask = cols < n
        band_rows = KU + rows[:, None] - cols[None, :]
        mask = (
            row_mask[:, None]
            & col_mask[None, :]
            & (band_rows >= 0)
            & (band_rows < BAND)
        )
        safe_band_rows = tl.where(mask, band_rows, 0)
        safe_cols = tl.where(col_mask, cols, 0)
        a_offsets = safe_band_rows + safe_cols[None, :] * LDA
        x_offsets = safe_cols * INCX
        ar = tl.load(
            a_ptr_i64 + a_offsets * 2,
            mask=mask,
            other=0,
            eviction_policy="evict_first",
        ).to(tl.float64, bitcast=True)
        ai = tl.load(
            a_ptr_i64 + a_offsets * 2 + 1,
            mask=mask,
            other=0,
            eviction_policy="evict_first",
        ).to(tl.float64, bitcast=True)
        if CONJ:
            ai = -ai
        xr = tl.load(
            x_ptr_i64 + x_offsets * 2,
            mask=col_mask,
            other=0,
            eviction_policy="evict_last",
        ).to(tl.float64, bitcast=True)
        xi = tl.load(
            x_ptr_i64 + x_offsets * 2 + 1,
            mask=col_mask,
            other=0,
            eviction_policy="evict_last",
        ).to(tl.float64, bitcast=True)
        acc_r += ar * xr[None, :] - ai * xi[None, :]
        acc_i += ar * xi[None, :] + ai * xr[None, :]

    sum_r = tl.sum(acc_r, axis=1)
    sum_i = tl.sum(acc_i, axis=1)
    y_offsets = rows * INCY * 2
    result_r = alpha_r * sum_r - alpha_i * sum_i
    result_i = alpha_r * sum_i + alpha_i * sum_r
    if not BETA_IS_ZERO:
        yr = tl.load(y_ptr + y_offsets, mask=row_mask, other=0.0)
        yi = tl.load(y_ptr + y_offsets + 1, mask=row_mask, other=0.0)
        result_r += beta_r * yr - beta_i * yi
        result_i += beta_r * yi + beta_i * yr
    tl.store(y_ptr + y_offsets, result_r, mask=row_mask)
    tl.store(y_ptr + y_offsets + 1, result_i, mask=row_mask)


zgbmv_n_small_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("zgbmv_n_small_hygon"),
        key=["m", "n", "LDA", "INCX", "INCY", "KU", "BAND", "CONJ", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(zgbmv_n_small_hygon_kernel)
)


dgbmv_n_split_band_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("dgbmv_n_split_band_hygon"),
        key=_DGBMV_KEY,
        restore_value=["y_ptr"],
    )(common_dgbmv_n_split_band_kernel.fn)
)

dgbmv_t_split_band_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("dgbmv_t_split_band_hygon"),
        key=_DGBMV_KEY,
        restore_value=["y_ptr"],
    )(common_dgbmv_t_split_band_kernel.fn)
)


def _pick_dgbmv_split_hygon(trans: int, m: int, n: int, out_len: int, band: int) -> int:
    split = _pick_split_band(out_len, band)
    if trans == CUBLAS_OP_N and band >= 512 and out_len >= 2048 and m <= n:
        return 1
    if trans == CUBLAS_OP_N or band < 256:
        return split
    if band < 512:
        return 8
    if m < n:
        return 8
    if out_len <= 4096:
        return 8
    if out_len <= 10000:
        return 4
    return 2


def dgbmv(
    trans: int,
    m: int,
    n: int,
    kl: int,
    ku: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert A.dtype == torch.float64 == x.dtype == y.dtype
    _check_common(A, x, y, trans, m, n, kl, ku, lda, incx, incy, complex_ok=False)
    if m == 0 or n == 0:
        return

    alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta_val = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    if alpha_val == 0.0:
        if beta_val == 0.0:
            y.zero_()
        elif beta_val != 1.0:
            y.mul_(beta_val)
        return

    public_trans, public_m, public_n, public_kl, public_ku = trans, m, n, kl, ku
    trans, m, n, kl, ku, conj = _row_major_gbmv_args(trans, m, n, kl, ku)
    band = kl + ku + 1
    out_len = m if trans == CUBLAS_OP_N else n
    if trans == CUBLAS_OP_N and 512 <= band < 1024 and out_len >= 16384 and m > n:
        with torch_device_fn.device(A.device):
            if beta_val != 1.0:
                dgbmv_scale_y_hygon_kernel[(triton.cdiv(out_len, 256),)](
                    y,
                    _f64_to_i64(beta_val),
                    out_len,
                    incy,
                    BETA_IS_ZERO=beta_val == 0.0,
                    BLOCK_SIZE=256,
                    num_warps=4,
                    num_stages=1,
                )
            dgbmv_n_scatter_hygon_kernel[(triton.cdiv(n, 1),)](
                A,
                x,
                y,
                _f64_to_i64(alpha_val),
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                band,
                BLOCK_COLS=1,
                BAND_TILE=64,
                num_warps=1,
                num_stages=1,
            )
        return
    if trans == CUBLAS_OP_N and 64 <= band < 128 and out_len >= 16384:
        with torch_device_fn.device(A.device):
            dgbmv_n_tiled_fixed_hygon_kernel[(triton.cdiv(out_len, 16),)](
                A,
                x,
                y,
                _f64_to_i64(alpha_val),
                _f64_to_i64(beta_val),
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                out_len,
                _band_bucket(band),
                BETA_IS_ZERO=beta_val == 0.0,
                BLOCK_SIZE_M=16,
                BAND_TILE=32,
                num_warps=2,
                num_stages=1,
            )
        return
    if trans == CUBLAS_OP_N and 128 <= band < 512 and 4095 <= out_len <= 4096:
        with torch_device_fn.device(A.device):
            dgbmv_n_row_hygon_kernel[(out_len,)](
                A,
                x,
                y,
                _f64_to_i64(alpha_val),
                _f64_to_i64(beta_val),
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                BETA_IS_ZERO=beta_val == 0.0,
                BLOCK_K=512,
                num_warps=2,
                num_stages=2,
            )
        return
    if trans == CUBLAS_OP_N and 64 <= band < 128 and out_len <= 1024:
        with torch_device_fn.device(A.device):
            dgbmv_n_row_hygon_kernel[(out_len,)](
                A,
                x,
                y,
                _f64_to_i64(alpha_val),
                _f64_to_i64(beta_val),
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                BETA_IS_ZERO=beta_val == 0.0,
                BLOCK_K=64,
                num_warps=1,
                num_stages=1,
            )
        return
    if trans != CUBLAS_OP_N and band >= 32:
        with torch_device_fn.device(A.device):
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
            dgbmv_t_tiled_hygon_kernel[grid](
                A,
                x,
                y,
                _f64_to_i64(alpha_val),
                _f64_to_i64(beta_val),
                m,
                n,
                lda,
                incx,
                incy,
                ku,
                band,
                CONJ=False,
                BETA_IS_ZERO=beta_val == 0.0,
            )
        return

    split_band = _pick_dgbmv_split_hygon(trans, m, n, out_len, band)
    if split_band == 1:
        if trans == CUBLAS_OP_N:
            with torch_device_fn.device(A.device):
                grid = lambda meta: (triton.cdiv(out_len, meta["BLOCK_SIZE_M"]),)
                kernel = (
                    dgbmv_n_tiled_hygon_kernel
                    if band >= 128 and out_len >= 1024
                    else dgbmv_n_hygon_kernel
                )
                kernel[grid](
                    A,
                    x,
                    y,
                    _f64_to_i64(alpha_val),
                    _f64_to_i64(beta_val),
                    m,
                    n,
                    lda,
                    incx,
                    incy,
                    kl,
                    ku,
                    band,
                    out_len,
                    _band_bucket(band),
                    BETA_IS_ZERO=beta_val == 0.0,
                )
            return
        common_dgbmv(
            public_trans,
            public_m,
            public_n,
            public_kl,
            public_ku,
            alpha,
            A,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
        )
        return

    if beta_val == 0.0:
        y.zero_()
    elif beta_val != 1.0:
        y.mul_(beta_val)

    with torch_device_fn.device(A.device):
        grid = lambda meta: (
            triton.cdiv(out_len, meta["BLOCK_SIZE_M"]),
            split_band,
        )
        kernel = (
            dgbmv_n_split_band_hygon_kernel
            if trans == CUBLAS_OP_N
            else dgbmv_t_split_band_hygon_kernel
        )
        kernel[grid](
            A,
            x,
            y,
            _f64_to_i64(alpha_val),
            m,
            n,
            lda,
            incx,
            incy,
            kl,
            ku,
            band,
            split_band,
            out_len,
            _band_bucket(band),
        )


def cgbmv(
    trans: int,
    m: int,
    n: int,
    kl: int,
    ku: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert A.dtype == torch.complex64 == x.dtype == y.dtype
    _check_common(A, x, y, trans, m, n, kl, ku, lda, incx, incy, complex_ok=True)
    ar, ai, br, bi = _complex_scalars(alpha, beta)
    public_trans, public_m, public_n, public_kl, public_ku = trans, m, n, kl, ku
    trans, m, n, kl, ku, conj = _row_major_gbmv_args(trans, m, n, kl, ku)
    band = kl + ku + 1
    if m == 0 or n == 0 or band < 32 or (ar == 0.0 and ai == 0.0):
        common_cgbmv(
            public_trans,
            public_m,
            public_n,
            public_kl,
            public_ku,
            alpha,
            A,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
        )
        return
    out_len = m if trans == CUBLAS_OP_N else n
    if (
        trans == CUBLAS_OP_N
        and not conj
        and out_len >= 16384
        and 128 <= band < 1024
        and (band < 512 or m > n)
    ):
        A_real = triton.reinterpret(A, tl.float32)
        x_real = triton.reinterpret(x, tl.float32)
        y_real = triton.reinterpret(y, tl.float32)
        with torch_device_fn.device(A.device):
            cgbmv_n_packed_hygon_kernel[(triton.cdiv(out_len, 8),)](
                A_real,
                x_real,
                y_real,
                ar,
                ai,
                br,
                bi,
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                CONJ=False,
                BLOCK_SIZE_M=8,
                BAND_TILE=32,
                num_warps=1,
                num_stages=1,
            )
        return
    if trans == CUBLAS_OP_N and conj and m == 16384 and n == 16384 and band == 513:
        A_real = triton.reinterpret(A, tl.float32)
        x_real = triton.reinterpret(x, tl.float32)
        y_real = triton.reinterpret(y, tl.float32)
        beta_is_zero = br == 0.0 and bi == 0.0
        with torch_device_fn.device(A.device):
            cgbmv_n_x_packed_hygon_kernel[(triton.cdiv(out_len, 16),)](
                A_real,
                x_real,
                y_real,
                ar,
                ai,
                br,
                bi,
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                BETA_IS_ZERO=beta_is_zero,
                BLOCK_SIZE_M=16,
                BAND_TILE=64,
                num_warps=4,
                num_stages=2,
                waves_per_eu=2,
            )
        return
    if trans == CUBLAS_OP_N and 64 <= band < 128 and out_len <= 1024:
        A_real = triton.reinterpret(A, tl.float32)
        x_real = triton.reinterpret(x, tl.float32)
        y_real = triton.reinterpret(y, tl.float32)
        beta_is_zero = br == 0.0 and bi == 0.0
        with torch_device_fn.device(A.device):
            complex_gbmv_n_active_small_hygon_kernel[(triton.cdiv(out_len, 4),)](
                A_real,
                x_real,
                y_real,
                _f32_to_i32(ar),
                _f32_to_i32(ai),
                _f32_to_i32(br),
                _f32_to_i32(bi),
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                CONJ=conj,
                FP64=False,
                BETA_IS_ZERO=beta_is_zero,
                BLOCK_SIZE_M=4,
                BAND_TILE=128,
                num_warps=1,
                num_stages=1,
            )
        return
    if trans == CUBLAS_OP_N and m > n + kl:
        A_real = triton.reinterpret(A, tl.float32)
        x_real = triton.reinterpret(x, tl.float32)
        y_real = triton.reinterpret(y, tl.float32)
        beta_is_zero = br == 0.0 and bi == 0.0
        with torch_device_fn.device(A.device):
            grid = lambda meta: (triton.cdiv(out_len, meta["BLOCK_SIZE_M"]),)
            complex_gbmv_n_active_hygon_kernel[grid](
                A_real,
                x_real,
                y_real,
                _f32_to_i32(ar),
                _f32_to_i32(ai),
                _f32_to_i32(br),
                _f32_to_i32(bi),
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                CONJ=conj,
                FP64=False,
                BETA_IS_ZERO=beta_is_zero,
            )
        return
    split_band = _pick_split_band(out_len, band)
    if trans == CUBLAS_OP_N and out_len >= 2048:
        split_band = 1
    if trans == CUBLAS_OP_N and split_band == 1 and (conj or m > 256 or n > 256):
        A_real = triton.reinterpret(A, tl.float32)
        x_real = triton.reinterpret(x, tl.float32)
        y_real = triton.reinterpret(y, tl.float32)
        beta_is_zero = br == 0.0 and bi == 0.0
        with torch_device_fn.device(A.device):
            grid = lambda meta: (triton.cdiv(out_len, meta["BLOCK_SIZE_M"]),)
            cgbmv_n_hygon_kernel[grid](
                A_real,
                x_real,
                y_real,
                ar,
                ai,
                br,
                bi,
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                out_len,
                _band_bucket(band),
                CONJ=conj,
                BETA_IS_ZERO=beta_is_zero,
            )
        return
    if trans == CUBLAS_OP_N and (conj or m > 256 or n > 256):
        common_cgbmv(
            public_trans,
            public_m,
            public_n,
            public_kl,
            public_ku,
            alpha,
            A,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
        )
        return

    A_real = triton.reinterpret(A, tl.float32)
    x_real = triton.reinterpret(x, tl.float32)
    y_real = triton.reinterpret(y, tl.float32)
    beta_is_zero = br == 0.0 and bi == 0.0
    with torch_device_fn.device(A.device):
        grid = lambda meta: (triton.cdiv(out_len, meta["BLOCK_SIZE_M"]),)
        if trans == CUBLAS_OP_N:
            cgbmv_n_small_hygon_kernel[grid](
                A_real,
                x_real,
                y_real,
                ar,
                ai,
                br,
                bi,
                m,
                n,
                lda,
                incx,
                incy,
                ku,
                band,
                BETA_IS_ZERO=beta_is_zero,
            )
            return
        cgbmv_t_tiled_hygon_kernel[grid](
            A_real,
            x_real,
            y_real,
            ar,
            ai,
            br,
            bi,
            m,
            out_len,
            lda,
            incx,
            incy,
            kl,
            ku,
            band,
            n,
            _band_bucket(band),
            CONJ=conj,
            BETA_IS_ZERO=beta_is_zero,
        )


def zgbmv(
    trans: int,
    m: int,
    n: int,
    kl: int,
    ku: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert A.dtype == torch.complex128 == x.dtype == y.dtype
    _check_common(A, x, y, trans, m, n, kl, ku, lda, incx, incy, complex_ok=True)
    ar, ai, br, bi = _complex_scalars(alpha, beta)
    public_trans, public_m, public_n, public_kl, public_ku = trans, m, n, kl, ku
    trans, m, n, kl, ku, conj = _row_major_gbmv_args(trans, m, n, kl, ku)
    band = kl + ku + 1
    if m == 0 or n == 0 or band < 32 or (ar == 0.0 and ai == 0.0):
        common_zgbmv(
            public_trans,
            public_m,
            public_n,
            public_kl,
            public_ku,
            alpha,
            A,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
        )
        return
    out_len = m if trans == CUBLAS_OP_N else n
    if trans == CUBLAS_OP_N and conj and m == 1023 and n == 1023 and band == 257:
        A_real = triton.reinterpret(A, tl.float64)
        x_real = triton.reinterpret(x, tl.float64)
        y_real = triton.reinterpret(y, tl.float64)
        beta_is_zero = br == 0.0 and bi == 0.0
        with torch_device_fn.device(A.device):
            _complex_gbmv_n_active_hygon_jit[(triton.cdiv(out_len, 4),)](
                A_real,
                x_real,
                y_real,
                _f64_to_i64(ar),
                _f64_to_i64(ai),
                _f64_to_i64(br),
                _f64_to_i64(bi),
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                CONJ=True,
                FP64=True,
                BETA_IS_ZERO=beta_is_zero,
                BLOCK_SIZE_M=4,
                BAND_TILE=64,
                num_warps=1,
                num_stages=1,
            )
        return
    if (
        public_trans in (CUBLAS_OP_T, CUBLAS_OP_C)
        and public_m == 4096
        and public_n == 16384
        and band == 513
    ):
        A_real = triton.reinterpret(A, tl.float64)
        x_real = triton.reinterpret(x, tl.float64)
        y_real = triton.reinterpret(y, tl.float64)
        with torch_device_fn.device(A.device):
            zgbmv_scale_y_hygon_kernel[(triton.cdiv(public_n, 256),)](
                y_real,
                _f64_to_i64(br),
                _f64_to_i64(bi),
                public_n,
                incy,
                BLOCK_SIZE=256,
                num_warps=4,
            )
            zgbmv_grouped_scatter_hygon_kernel[
                (
                    triton.cdiv(public_m, 32),
                    triton.cdiv(band + 31, 16),
                )
            ](
                A_real,
                x_real,
                y_real,
                _f64_to_i64(ar),
                _f64_to_i64(ai),
                public_m,
                public_n,
                lda,
                incx,
                incy,
                public_kl,
                band,
                CONJ=public_trans == CUBLAS_OP_C,
                BLOCK_ROWS=32,
                OUTPUT_TILE=16,
                num_warps=4,
                num_stages=1,
            )
        return

    if (
        trans == CUBLAS_OP_N
        and public_trans in (CUBLAS_OP_T, CUBLAS_OP_C)
        and public_m == 4096
        and public_n == 16384
        and band == 257
    ):
        A_real = triton.reinterpret(A, tl.float64)
        x_real = triton.reinterpret(x, tl.float64)
        y_real = triton.reinterpret(y, tl.float64)
        beta_is_zero = br == 0.0 and bi == 0.0
        with torch_device_fn.device(A.device):
            _complex_gbmv_n_active_hygon_jit[(triton.cdiv(out_len, 8),)](
                A_real,
                x_real,
                y_real,
                _f64_to_i64(ar),
                _f64_to_i64(ai),
                _f64_to_i64(br),
                _f64_to_i64(bi),
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                CONJ=conj,
                FP64=True,
                BETA_IS_ZERO=beta_is_zero,
                BLOCK_SIZE_M=8,
                BAND_TILE=128,
                num_warps=4,
                num_stages=1,
            )
        return
    if (
        trans == CUBLAS_OP_N
        and band in (257, 513)
        and (
            (
                public_trans == CUBLAS_OP_C
                and public_m == public_n
                and public_m in (4095, 4096)
            )
            or (
                public_trans in (CUBLAS_OP_T, CUBLAS_OP_C)
                and public_m == 16384
                and public_n == 4096
            )
        )
    ):
        A_real = triton.reinterpret(A, tl.float64)
        x_real = triton.reinterpret(x, tl.float64)
        y_real = triton.reinterpret(y, tl.float64)
        beta_is_zero = br == 0.0 and bi == 0.0
        with torch_device_fn.device(A.device):
            zgbmv_n_vector_load_hygon_kernel[(triton.cdiv(out_len, 8),)](
                A_real,
                x_real,
                y_real,
                _f64_to_i64(ar),
                _f64_to_i64(ai),
                _f64_to_i64(br),
                _f64_to_i64(bi),
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                CONJ=conj,
                BETA_IS_ZERO=beta_is_zero,
                BLOCK_SIZE_M=8,
                BAND_TILE=128,
                num_warps=4,
                num_stages=1,
            )
        return

    if (
        trans == CUBLAS_OP_N
        and m == n
        and m in (10000, 16384)
        and band in (257, 513)
        and (not conj or band == 257 or m == 10000)
    ):
        A_real = triton.reinterpret(A, tl.float64)
        x_real = triton.reinterpret(x, tl.float64)
        y_real = triton.reinterpret(y, tl.float64)
        beta_is_zero = br == 0.0 and bi == 0.0
        with torch_device_fn.device(A.device):
            _complex_gbmv_n_active_hygon_jit[(triton.cdiv(out_len, 32),)](
                A_real,
                x_real,
                y_real,
                _f64_to_i64(ar),
                _f64_to_i64(ai),
                _f64_to_i64(br),
                _f64_to_i64(bi),
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                CONJ=conj,
                FP64=True,
                BETA_IS_ZERO=beta_is_zero,
                BLOCK_SIZE_M=32,
                BAND_TILE=64,
                num_warps=8,
                num_stages=1,
            )
        return
    if trans == CUBLAS_OP_N and 64 <= band < 128 and out_len <= 1024:
        A_real = triton.reinterpret(A, tl.float64)
        x_real = triton.reinterpret(x, tl.float64)
        y_real = triton.reinterpret(y, tl.float64)
        beta_is_zero = br == 0.0 and bi == 0.0
        with torch_device_fn.device(A.device):
            complex_gbmv_n_active_small_hygon_kernel[(triton.cdiv(out_len, 4),)](
                A_real,
                x_real,
                y_real,
                _f64_to_i64(ar),
                _f64_to_i64(ai),
                _f64_to_i64(br),
                _f64_to_i64(bi),
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                CONJ=conj,
                FP64=True,
                BETA_IS_ZERO=beta_is_zero,
                BLOCK_SIZE_M=4,
                BAND_TILE=128,
                num_warps=1,
                num_stages=1,
            )
        return
    if trans == CUBLAS_OP_N and out_len >= 2048:
        A_real = triton.reinterpret(A, tl.float64)
        x_real = triton.reinterpret(x, tl.float64)
        y_real = triton.reinterpret(y, tl.float64)
        beta_is_zero = br == 0.0 and bi == 0.0
        with torch_device_fn.device(A.device):
            grid = lambda meta: (triton.cdiv(out_len, meta["BLOCK_SIZE_M"]),)
            complex_gbmv_n_active_hygon_kernel[grid](
                A_real,
                x_real,
                y_real,
                _f64_to_i64(ar),
                _f64_to_i64(ai),
                _f64_to_i64(br),
                _f64_to_i64(bi),
                m,
                n,
                lda,
                incx,
                incy,
                kl,
                ku,
                band,
                CONJ=conj,
                FP64=True,
                BETA_IS_ZERO=beta_is_zero,
            )
        return
    if trans == CUBLAS_OP_N and (m > 256 or n > 256 or (conj and band < 256)):
        common_zgbmv(
            public_trans,
            public_m,
            public_n,
            public_kl,
            public_ku,
            alpha,
            A,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
        )
        return

    A_real = triton.reinterpret(A, tl.float64)
    x_real = triton.reinterpret(x, tl.float64)
    y_real = triton.reinterpret(y, tl.float64)
    beta_is_zero = br == 0.0 and bi == 0.0
    with torch_device_fn.device(A.device):
        grid = lambda meta: (triton.cdiv(out_len, meta["BLOCK_SIZE_M"]),)
        if trans == CUBLAS_OP_N:
            zgbmv_n_small_hygon_kernel[grid](
                A_real,
                x_real,
                y_real,
                _f64_to_i64(ar),
                _f64_to_i64(ai),
                _f64_to_i64(br),
                _f64_to_i64(bi),
                m,
                n,
                lda,
                incx,
                incy,
                ku,
                band,
                CONJ=conj,
                BETA_IS_ZERO=beta_is_zero,
            )
            return
        if m >= 1024 and n >= 4096:
            zgbmv_t_pair_hygon_kernel[grid](
                A_real,
                x_real,
                y_real,
                _f64_to_i64(ar),
                _f64_to_i64(ai),
                _f64_to_i64(br),
                _f64_to_i64(bi),
                m,
                n,
                lda,
                incx,
                incy,
                ku,
                band,
                CONJ=conj,
                BETA_IS_ZERO=beta_is_zero,
            )
            return
        if band < 256:
            kernel = zgbmv_t_tiled_hygon_kernel
        else:
            kernel = zgbmv_t_wide_hygon_kernel
        kernel[grid](
            A_real,
            x_real,
            y_real,
            _f64_to_i64(ar),
            _f64_to_i64(ai),
            _f64_to_i64(br),
            _f64_to_i64(bi),
            m,
            out_len,
            lda,
            incx,
            incy,
            kl,
            ku,
            band,
            n,
            _band_bucket(band),
            CONJ=conj,
            BETA_IS_ZERO=beta_is_zero,
        )


__all__ = ["cgbmv", "dgbmv", "zgbmv"]
