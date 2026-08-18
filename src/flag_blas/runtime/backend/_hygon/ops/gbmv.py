import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2._constants import CUBLAS_OP_C, CUBLAS_OP_N
from flag_blas.ops.level2.gbmv import (
    ScalarType,
    _band_bucket,
    _check_common,
    _complex_scalars,
    _f64_to_i64,
    _pick_split_band,
)
from flag_blas.ops.level2.gbmv import cgbmv as common_cgbmv
from flag_blas.ops.level2.gbmv import cgbmv_t_kernel as common_cgbmv_t_kernel
from flag_blas.ops.level2.gbmv import dgbmv as common_dgbmv
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
        key=["m", "n", "LDA", "INCX", "INCY", "KU", "BAND", "BETA_IS_ZERO"],
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

    band = kl + ku + 1
    out_len = m if trans == CUBLAS_OP_N else n
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
                BETA_IS_ZERO=beta_val == 0.0,
            )
        return

    split_band = _pick_dgbmv_split_hygon(trans, m, n, out_len, band)
    if split_band == 1:
        common_dgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
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
    band = kl + ku + 1
    if m == 0 or n == 0 or band < 32 or (ar == 0.0 and ai == 0.0):
        common_cgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
        return
    if trans == CUBLAS_OP_N and (m > 256 or n > 256):
        common_cgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
        return

    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)
    conj = trans == CUBLAS_OP_C
    beta_is_zero = br == 0.0 and bi == 0.0
    out_len = m if trans == CUBLAS_OP_N else n
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
    band = kl + ku + 1
    if m == 0 or n == 0 or band < 32 or (ar == 0.0 and ai == 0.0):
        common_zgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
        return
    if trans == CUBLAS_OP_N and (m > 256 or n > 256):
        common_zgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
        return

    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)
    conj = trans == CUBLAS_OP_C
    beta_is_zero = br == 0.0 and bi == 0.0
    out_len = m if trans == CUBLAS_OP_N else n
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
