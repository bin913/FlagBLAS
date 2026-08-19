import torch
import triton

from flag_blas.ops.level2.ger import (
    _CGER_CONFIGS,
    _ZGER_CONFIGS,
    ScalarType,
    _check_ger_common,
    _f64_to_i64,
    _grid,
    _scalar_to_complex_parts,
)
from flag_blas.ops.level2.ger import cger_kernel as common_cger_kernel
from flag_blas.ops.level2.ger import zger_kernel as common_zger_kernel
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

_ZGER_HYGON_CONFIGS = _ZGER_CONFIGS + [
    triton.Config(
        {"BLOCK_SIZE_M": 1, "BLOCK_SIZE_N": 128},
        num_warps=1,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 1, "BLOCK_SIZE_N": 256},
        num_warps=1,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 1, "BLOCK_SIZE_N": 512},
        num_warps=1,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 1, "BLOCK_SIZE_N": 512},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 1, "BLOCK_SIZE_N": 1024},
        num_warps=1,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 1, "BLOCK_SIZE_N": 1024},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 1, "BLOCK_SIZE_N": 1024},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 256},
        num_warps=1,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 1, "BLOCK_SIZE_N": 4096},
        num_warps=8,
        num_stages=2,
    ),
]

cger_hygon_kernel = libentry()(
    libtuner(
        configs=_CGER_CONFIGS,
        key=["m", "n", "LDA", "INCX", "INCY", "CONJ_Y"],
        restore_value=["A_ptr"],
    )(common_cger_kernel.fn.fn)
)

zger_hygon_kernel = libentry()(
    libtuner(
        configs=_ZGER_HYGON_CONFIGS,
        key=["m", "n", "LDA", "INCX", "INCY", "CONJ_Y"],
        restore_value=["A_ptr"],
    )(common_zger_kernel.fn.fn)
)


def _cger(
    m: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
    conj_y: bool,
) -> None:
    if not _check_ger_common(m, n, x, incx, y, incy, A, lda, torch.complex64):
        return
    alpha_real, alpha_imag = _scalar_to_complex_parts(alpha)
    if alpha_real == 0.0 and alpha_imag == 0.0:
        return
    with torch_device_fn.device(A.device):
        cger_hygon_kernel[_grid(m, n)](
            torch.view_as_real(x),
            torch.view_as_real(y),
            torch.view_as_real(A),
            alpha_real,
            alpha_imag,
            m,
            n,
            incx,
            incy,
            lda,
            conj_y,
        )


def cgeru(m, n, alpha, x, incx, y, incy, A, lda) -> None:
    _cger(m, n, alpha, x, incx, y, incy, A, lda, False)


def cgerc(m, n, alpha, x, incx, y, incy, A, lda) -> None:
    _cger(m, n, alpha, x, incx, y, incy, A, lda, True)


def _zger(
    m: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
    conj_y: bool,
) -> None:
    if not _check_ger_common(m, n, x, incx, y, incy, A, lda, torch.complex128):
        return
    alpha_real, alpha_imag = _scalar_to_complex_parts(alpha)
    if alpha_real == 0.0 and alpha_imag == 0.0:
        return
    with torch_device_fn.device(A.device):
        zger_hygon_kernel[_grid(m, n)](
            torch.view_as_real(x),
            torch.view_as_real(y),
            torch.view_as_real(A),
            _f64_to_i64(alpha_real),
            _f64_to_i64(alpha_imag),
            m,
            n,
            incx,
            incy,
            lda,
            conj_y,
        )


def zgeru(m, n, alpha, x, incx, y, incy, A, lda) -> None:
    _zger(m, n, alpha, x, incx, y, incy, A, lda, False)


def zgerc(m, n, alpha, x, incx, y, incy, A, lda) -> None:
    _zger(m, n, alpha, x, incx, y, incy, A, lda, True)


__all__ = ["cgeru", "cgerc", "zgeru", "zgerc"]
