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
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2.syr import (
    ScalarType,
    _check_syr_args,
    _f64_to_i64,
    _row_major_uplo,
)
from flag_blas.ops.level2.syr import _syr_complex_kernel as common_syr_complex_kernel
from flag_blas.ops.level2.syr import _syr_kernel as common_syr_kernel
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

_SYR_KEY = ["n", "lda", "incx", "uplo"]
_SYR_RESTORE = ["A"]

_common_syr_kernel_jit = common_syr_kernel.fn
_common_syr_complex_kernel_jit = common_syr_complex_kernel.fn


@triton.jit
def _hygon_syr_kernel(
    A,
    x,
    alpha,
    n: tl.constexpr,
    lda: tl.constexpr,
    incx: tl.constexpr,
    uplo: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    _common_syr_kernel_jit(
        A,
        x,
        alpha,
        n,
        lda,
        incx,
        uplo,
        IS_DOUBLE,
        BLOCK_M,
        BLOCK_N,
    )


@triton.jit
def _hygon_syr_complex_kernel(
    A,
    x,
    alpha_r,
    alpha_i,
    n: tl.constexpr,
    lda: tl.constexpr,
    incx: tl.constexpr,
    uplo: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
    TRIANGULAR_GRID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Keep the computation shared while giving Hygon LibTuner an independent
    # kernel identity, so public-kernel benchmark history cannot select stale
    # winners for the Hygon-only candidate set.
    _common_syr_complex_kernel_jit(
        A,
        x,
        alpha_r,
        alpha_i,
        n,
        lda,
        incx,
        uplo,
        IS_DOUBLE,
        TRIANGULAR_GRID,
        BLOCK_M,
        BLOCK_N,
    )


ssyr_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("ssyr_hygon"),
        key=_SYR_KEY,
        restore_value=_SYR_RESTORE,
    )(_hygon_syr_kernel)
)

dsyr_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("dsyr_hygon"),
        key=_SYR_KEY,
        restore_value=_SYR_RESTORE,
    )(_hygon_syr_kernel)
)

csyr_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("csyr_hygon"),
        key=_SYR_KEY,
        restore_value=_SYR_RESTORE,
    )(_hygon_syr_complex_kernel)
)

zsyr_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("zsyr_hygon"),
        key=_SYR_KEY,
        restore_value=_SYR_RESTORE,
    )(_hygon_syr_complex_kernel)
)


def _syr_real_hygon(
    uplo,
    n,
    alpha,
    x,
    incx,
    A,
    lda,
    dtype,
    kernel,
):
    _check_syr_args(uplo, n, x, incx, A, lda, dtype)
    if n == 0:
        return A
    uplo = _row_major_uplo(uplo)
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    is_double = dtype == torch.float64
    kernel_alpha = _f64_to_i64(alpha_value) if is_double else alpha_value

    def grid(meta):
        tile_count = triton.cdiv(n, meta["BLOCK_M"])
        return (tile_count * (tile_count + 1) // 2,)

    with torch_device_fn.device(A.device):
        kernel[grid](
            A,
            x,
            kernel_alpha,
            n,
            lda,
            incx,
            uplo,
            IS_DOUBLE=is_double,
        )
    return A


def _syr_complex_hygon(
    uplo,
    n,
    alpha,
    x,
    incx,
    A,
    lda,
    dtype,
    kernel,
):
    _check_syr_args(uplo, n, x, incx, A, lda, dtype)
    if n == 0:
        return A
    uplo = _row_major_uplo(uplo)
    alpha_value = complex(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    is_double = dtype == torch.complex128
    alpha_r = _f64_to_i64(alpha_value.real) if is_double else alpha_value.real
    alpha_i = _f64_to_i64(alpha_value.imag) if is_double else alpha_value.imag
    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)

    def grid(meta):
        tile_count = triton.cdiv(n, meta["BLOCK_M"])
        return (tile_count * (tile_count + 1) // 2,)

    with torch_device_fn.device(A.device):
        kernel[grid](
            A_real,
            x_real,
            alpha_r,
            alpha_i,
            n,
            lda,
            incx,
            uplo,
            IS_DOUBLE=is_double,
            TRIANGULAR_GRID=True,
        )
    return A


def ssyr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    return _syr_real_hygon(
        uplo, n, alpha, x, incx, A, lda, torch.float32, ssyr_hygon_kernel
    )


def dsyr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    return _syr_real_hygon(
        uplo, n, alpha, x, incx, A, lda, torch.float64, dsyr_hygon_kernel
    )


def csyr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    return _syr_complex_hygon(
        uplo, n, alpha, x, incx, A, lda, torch.complex64, csyr_hygon_kernel
    )


def zsyr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    return _syr_complex_hygon(
        uplo, n, alpha, x, incx, A, lda, torch.complex128, zsyr_hygon_kernel
    )


__all__ = ["csyr", "dsyr", "ssyr", "zsyr"]
