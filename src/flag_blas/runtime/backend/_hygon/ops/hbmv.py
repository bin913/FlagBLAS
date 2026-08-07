import torch
import triton

from flag_blas import runtime
from flag_blas.ops.level2.hbmv import (
    ScalarType,
    _band_bucket,
    _check_common,
    _complex_scalars,
    _f64_to_i64,
    _strided_y,
    chbmv as common_chbmv,
    chbmv_kernel as common_chbmv_kernel,
    zhbmv as common_zhbmv,
    zhbmv_kernel as common_zhbmv_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner


chbmv_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("chbmv_hygon"),
        key=["n", "k_bucket", "uplo_key"],
        restore_value=["y_ptr"],
    )(common_chbmv_kernel.fn)
)

zhbmv_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("zhbmv_hygon"),
        key=["n", "k_bucket", "uplo_key"],
        restore_value=["y_ptr"],
    )(common_zhbmv_kernel.fn)
)


def _use_chbmv_hygon_kernel(n: int, k: int) -> bool:
    return 0 < n <= 1024 and k >= 128


def _use_zhbmv_hygon_kernel(n: int, k: int) -> bool:
    return 0 < n <= 512 and k >= 128


def chbmv(
    uplo: int,
    n: int,
    k: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    if not _use_chbmv_hygon_kernel(n, k):
        common_chbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
        return

    assert A.dtype == torch.complex64 == x.dtype == y.dtype
    _check_common(A, x, y, uplo, n, k, lda, incx, incy)
    ar, ai, br, bi = _complex_scalars(alpha, beta)
    y_view = _strided_y(y, n, incy)
    if ar == 0.0 and ai == 0.0:
        if br == 0.0 and bi == 0.0:
            y_view.zero_()
        elif br != 1.0 or bi != 0.0:
            y_view.mul_(complex(br, bi))
        return

    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)
    with torch_device_fn.device(A.device):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        chbmv_hygon_kernel[grid](
            A_real,
            x_real,
            y_real,
            ar,
            ai,
            br,
            bi,
            n,
            k,
            lda,
            incx,
            incy,
            _band_bucket(k + 1),
            uplo,
            UPLO=uplo,
            BETA_IS_ZERO=br == 0.0 and bi == 0.0,
        )


def zhbmv(
    uplo: int,
    n: int,
    k: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    if not _use_zhbmv_hygon_kernel(n, k):
        common_zhbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
        return

    assert A.dtype == torch.complex128 == x.dtype == y.dtype
    _check_common(A, x, y, uplo, n, k, lda, incx, incy)
    ar, ai, br, bi = _complex_scalars(alpha, beta)
    y_view = _strided_y(y, n, incy)
    if ar == 0.0 and ai == 0.0:
        if br == 0.0 and bi == 0.0:
            y_view.zero_()
        elif br != 1.0 or bi != 0.0:
            y_view.mul_(complex(br, bi))
        return

    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)
    with torch_device_fn.device(A.device):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        zhbmv_hygon_kernel[grid](
            A_real,
            x_real,
            y_real,
            _f64_to_i64(ar),
            _f64_to_i64(ai),
            _f64_to_i64(br),
            _f64_to_i64(bi),
            n,
            k,
            lda,
            incx,
            incy,
            _band_bucket(k + 1),
            uplo,
            UPLO=uplo,
            BETA_IS_ZERO=br == 0.0 and bi == 0.0,
        )


__all__ = ["chbmv", "zhbmv"]
