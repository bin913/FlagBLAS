import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2._constants import CUBLAS_OP_C, CUBLAS_OP_N, CUBLAS_OP_T
from flag_blas.ops.level2.gemv import SPLITK_K_THRESHOLD, SPLITK_M_THRESHOLD, ScalarType
from flag_blas.ops.level2.gemv import cgemv as common_cgemv
from flag_blas.ops.level2.gemv import zgemv as common_zgemv
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner
from flag_blas.utils.libentry import LibTuner

_HYGON_CGEMV_N_KEY = ["m", "n", "STRIDE_AM", "INCX", "INCY", "BETA_IS_ZERO"]
_HYGON_CGEMV_T_KEY = [
    "m",
    "n",
    "STRIDE_AM",
    "STRIDE_AN",
    "INCX",
    "INCY",
    "CONJ",
    "BETA_IS_ZERO",
]


@LibTuner.register_policy("hygon_cgemv_n_stable")
def _hygon_cgemv_n_stable_policy(bench_fn, configs, args, kwargs):
    timings = {config: bench_fn(config) for config in configs}
    best_config = min(timings, key=lambda config: timings[config][-1])
    return best_config, timings


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cgemv_n"),
    key=_HYGON_CGEMV_N_KEY,
    restore_value=["y_ptr"],
    policy="hygon_cgemv_n_stable",
)
@triton.jit
def hygon_cgemv_n_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
    beta_real: tl.float32,
    beta_imag: tl.float32,
    m,
    n,
    STRIDE_AM,
    INCX,
    INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < m

    k_offsets_init = tl.arange(0, BLOCK_SIZE_K)
    a_base = row_offsets[:, None] * STRIDE_AM + k_offsets_init[None, :]
    x_base = k_offsets_init * INCX

    a_ptr_int64 = a_ptr.to(tl.pointer_type(tl.int64))
    x_ptr_int64 = x_ptr.to(tl.pointer_type(tl.int64))

    a_ptrs_int64 = a_ptr_int64 + a_base
    x_ptrs_int64 = x_ptr_int64 + x_base

    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    step_a = BLOCK_SIZE_K
    step_x = BLOCK_SIZE_K * INCX

    for k_start in range(0, n, BLOCK_SIZE_K):
        k_offsets = k_start + k_offsets_init
        k_mask = k_offsets < n
        a_mask = row_mask[:, None] & k_mask[None, :]

        a_val = tl.load(
            a_ptrs_int64, mask=a_mask, other=0, eviction_policy="evict_first"
        )
        x_val = tl.load(
            x_ptrs_int64, mask=k_mask, other=0, eviction_policy="evict_last"
        )

        a_real = a_val.to(tl.int32).to(tl.float32, bitcast=True)
        a_imag = (a_val >> 32).to(tl.int32).to(tl.float32, bitcast=True)

        x_real = x_val.to(tl.int32).to(tl.float32, bitcast=True)
        x_imag = (x_val >> 32).to(tl.int32).to(tl.float32, bitcast=True)

        xr_block = x_real[None, :]
        xi_block = x_imag[None, :]

        acc_real_2d += a_real * xr_block - a_imag * xi_block
        acc_imag_2d += a_real * xi_block + a_imag * xr_block

        a_ptrs_int64 += step_a
        x_ptrs_int64 += step_x

    acc_real = tl.sum(acc_real_2d, axis=1)
    acc_imag = tl.sum(acc_imag_2d, axis=1)

    y_base = row_offsets * INCY * 2
    if BETA_IS_ZERO:
        result_real = alpha_real * acc_real - alpha_imag * acc_imag
        result_imag = alpha_real * acc_imag + alpha_imag * acc_real
    else:
        y_real = tl.load(y_ptr + y_base, mask=row_mask, other=0.0)
        y_imag = tl.load(y_ptr + y_base + 1, mask=row_mask, other=0.0)
        result_real = (alpha_real * acc_real - alpha_imag * acc_imag) + (
            beta_real * y_real - beta_imag * y_imag
        )
        result_imag = (alpha_real * acc_imag + alpha_imag * acc_real) + (
            beta_real * y_imag + beta_imag * y_real
        )

    tl.store(y_ptr + y_base, result_real, mask=row_mask)
    tl.store(y_ptr + y_base + 1, result_imag, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cgemv_t_small_hygon"),
    key=_HYGON_CGEMV_T_KEY,
    restore_value=["y_ptr"],
)
@triton.jit
def hygon_cgemv_t_small_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
    beta_real: tl.float32,
    beta_imag: tl.float32,
    m,
    n,
    STRIDE_AM,
    STRIDE_AN,
    INCX,
    INCY,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < m

    k_offsets_init = tl.arange(0, BLOCK_SIZE_K)
    a_base = row_offsets[:, None] * STRIDE_AM + k_offsets_init[None, :] * STRIDE_AN
    x_base = k_offsets_init * INCX

    a_ptr_int64 = a_ptr.to(tl.pointer_type(tl.int64))
    x_ptr_int64 = x_ptr.to(tl.pointer_type(tl.int64))

    a_ptrs_int64 = a_ptr_int64 + a_base
    x_ptrs_int64 = x_ptr_int64 + x_base

    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    step_a = BLOCK_SIZE_K * STRIDE_AN
    step_x = BLOCK_SIZE_K * INCX

    for k_start in range(0, n, BLOCK_SIZE_K):
        k_offsets = k_start + k_offsets_init
        k_mask = k_offsets < n
        a_mask = row_mask[:, None] & k_mask[None, :]

        a_val = tl.load(
            a_ptrs_int64, mask=a_mask, other=0, eviction_policy="evict_first"
        )
        x_val = tl.load(
            x_ptrs_int64, mask=k_mask, other=0, eviction_policy="evict_last"
        )

        a_real = a_val.to(tl.int32).to(tl.float32, bitcast=True)
        a_imag = (a_val >> 32).to(tl.int32).to(tl.float32, bitcast=True)

        x_real = x_val.to(tl.int32).to(tl.float32, bitcast=True)
        x_imag = (x_val >> 32).to(tl.int32).to(tl.float32, bitcast=True)

        xr_block = x_real[None, :]
        xi_block = x_imag[None, :]

        if CONJ == 1:
            acc_real_2d += a_real * xr_block + a_imag * xi_block
            acc_imag_2d += a_real * xi_block - a_imag * xr_block
        else:
            acc_real_2d += a_real * xr_block - a_imag * xi_block
            acc_imag_2d += a_real * xi_block + a_imag * xr_block

        a_ptrs_int64 += step_a
        x_ptrs_int64 += step_x

    acc_real = tl.sum(acc_real_2d, axis=1)
    acc_imag = tl.sum(acc_imag_2d, axis=1)

    y_base = row_offsets * INCY * 2
    if BETA_IS_ZERO:
        result_real = alpha_real * acc_real - alpha_imag * acc_imag
        result_imag = alpha_real * acc_imag + alpha_imag * acc_real
    else:
        y_real = tl.load(y_ptr + y_base, mask=row_mask, other=0.0)
        y_imag = tl.load(y_ptr + y_base + 1, mask=row_mask, other=0.0)
        result_real = (alpha_real * acc_real - alpha_imag * acc_imag) + (
            beta_real * y_real - beta_imag * y_imag
        )
        result_imag = (alpha_real * acc_imag + alpha_imag * acc_real) + (
            beta_real * y_imag + beta_imag * y_real
        )

    tl.store(y_ptr + y_base, result_real, mask=row_mask)
    tl.store(y_ptr + y_base + 1, result_imag, mask=row_mask)


def _scale_strided_y(y: torch.Tensor, length: int, incy: int, beta: complex) -> None:
    logical_y = y[: 1 + (length - 1) * incy : incy]
    if beta == 0.0:
        logical_y.zero_()
    elif beta != 1.0:
        logical_y.mul_(beta)


def cgemv(
    trans: int,
    m: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == torch.complex64
    assert x.dtype == torch.complex64
    assert y.dtype == torch.complex64
    assert A.device == x.device == y.device
    assert trans in [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
    assert incx > 0 and incy > 0
    assert lda >= n

    if m == 0 or n == 0:
        return

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    beta = beta.item() if isinstance(beta, torch.Tensor) else beta
    alpha_real = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    alpha_imag = float(alpha.imag) if isinstance(alpha, complex) else 0.0
    beta_real = float(beta.real) if isinstance(beta, complex) else float(beta)
    beta_imag = float(beta.imag) if isinstance(beta, complex) else 0.0

    if trans == CUBLAS_OP_N:
        len_x, len_y = n, m
        eff_m, eff_n = m, n
        stride_am, stride_an = lda, 1
    else:
        len_x, len_y = m, n
        eff_m, eff_n = n, m
        stride_am, stride_an = 1, lda

    assert x.numel() >= 1 + (len_x - 1) * incx
    assert y.numel() >= 1 + (len_y - 1) * incy

    if alpha_real == 0.0 and alpha_imag == 0.0:
        _scale_strided_y(y, len_y, incy, beta)
        return

    use_splitk = eff_m <= SPLITK_M_THRESHOLD and eff_n >= SPLITK_K_THRESHOLD
    if use_splitk:
        _scale_strided_y(y, len_y, incy, beta)
        return common_cgemv(
            trans,
            m,
            n,
            alpha,
            A,
            lda,
            x,
            incx,
            1.0 + 0.0j,
            y,
            incy,
        )

    use_small_t = trans == CUBLAS_OP_T and m <= 512 and n <= 512
    if trans != CUBLAS_OP_N and not use_small_t:
        return common_cgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)

    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)
    beta_is_zero = beta_real == 0.0 and beta_imag == 0.0
    grid = lambda meta: (triton.cdiv(eff_m, meta["BLOCK_SIZE_M"]),)

    with torch_device_fn.device(A.device):
        if use_small_t:
            hygon_cgemv_t_small_kernel[grid](
                A_real,
                x_real,
                y_real,
                alpha_real,
                alpha_imag,
                beta_real,
                beta_imag,
                eff_m,
                eff_n,
                stride_am,
                stride_an,
                incx,
                incy,
                0,
                beta_is_zero,
            )
        else:
            hygon_cgemv_n_kernel[grid](
                A_real,
                x_real,
                y_real,
                alpha_real,
                alpha_imag,
                beta_real,
                beta_imag,
                eff_m,
                eff_n,
                stride_am,
                incx,
                incy,
                beta_is_zero,
            )


def zgemv(
    trans: int,
    m: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == torch.complex128
    assert x.dtype == torch.complex128
    assert y.dtype == torch.complex128
    assert A.device == x.device == y.device
    assert trans in [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
    assert incx > 0 and incy > 0
    assert lda >= n

    if m == 0 or n == 0:
        return

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    beta = beta.item() if isinstance(beta, torch.Tensor) else beta
    alpha_real = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    alpha_imag = float(alpha.imag) if isinstance(alpha, complex) else 0.0

    if trans == CUBLAS_OP_N:
        len_x, len_y = n, m
        eff_m, eff_n = m, n
    else:
        len_x, len_y = m, n
        eff_m, eff_n = n, m

    assert x.numel() >= 1 + (len_x - 1) * incx
    assert y.numel() >= 1 + (len_y - 1) * incy

    if alpha_real == 0.0 and alpha_imag == 0.0:
        _scale_strided_y(y, len_y, incy, beta)
        return

    use_splitk = eff_m <= SPLITK_M_THRESHOLD and eff_n >= SPLITK_K_THRESHOLD
    if use_splitk:
        _scale_strided_y(y, len_y, incy, beta)
        return common_zgemv(
            trans,
            m,
            n,
            alpha,
            A,
            lda,
            x,
            incx,
            1.0 + 0.0j,
            y,
            incy,
        )

    return common_zgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
