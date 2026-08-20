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

from flag_blas.ops.level2.tbsv import (
    CUBLAS_DIAG_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_OP_N,
    _band_bucket,
    _check_tbsv,
    _complex_tbsv_blocked_kernel,
    _real_tbsv_blocked_kernel,
    _row_major_tbsv_args,
    _tbsv_flags,
)
from flag_blas.ops.level2.tbsv import ctbsv as _common_ctbsv
from flag_blas.ops.level2.tbsv import dtbsv as _common_dtbsv
from flag_blas.ops.level2.tbsv import stbsv as _common_stbsv
from flag_blas.ops.level2.tbsv import ztbsv as _common_ztbsv
from flag_blas.runtime import torch_device_fn

ztbsv = _common_ztbsv


def _blocked_state(uplo, trans):
    physical_uplo, physical_trans, conj = _row_major_tbsv_args(uplo, trans)
    trans_flag = int(physical_trans != CUBLAS_OP_N)
    lower_eff = int((physical_uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans_flag == 1))
    return physical_uplo, trans_flag, conj, lower_eff


def _launch_real_blocked(A, x, uplo, trans, diag, n, k, lda, is_double):
    physical_uplo, trans_flag, _, lower_eff = _blocked_state(uplo, trans)
    block_n = 32
    _real_tbsv_blocked_kernel[(triton.cdiv(n, block_n),)](
        A,
        x,
        _tbsv_flags(A.device),
        n,
        k,
        lda,
        UPLO=physical_uplo,
        TRANS=trans_flag,
        UNIT=int(diag == CUBLAS_DIAG_UNIT),
        LOWER_EFF=lower_eff,
        FORWARD=lower_eff,
        IS_DOUBLE=is_double,
        BLOCK_N=block_n,
        BAND_K=_band_bucket(k),
        num_warps=4,
    )


def _launch_complex64_blocked(A, x, uplo, trans, diag, n, k, lda):
    physical_uplo, trans_flag, conj, lower_eff = _blocked_state(uplo, trans)
    block_n = 32
    _complex_tbsv_blocked_kernel[(triton.cdiv(n, block_n),)](
        torch.view_as_real(A),
        torch.view_as_real(x),
        _tbsv_flags(A.device),
        n,
        k,
        lda,
        UPLO=physical_uplo,
        TRANS=trans_flag,
        UNIT=int(diag == CUBLAS_DIAG_UNIT),
        CONJ=conj,
        LOWER_EFF=lower_eff,
        FORWARD=lower_eff,
        IS_DOUBLE=False,
        BLOCK_N=block_n,
        BAND_K=_band_bucket(k),
        num_warps=4,
    )


def stbsv(uplo, trans, diag, n, k, A, lda, x, incx):
    if incx != 1 or k == 0:
        return _common_stbsv(uplo, trans, diag, n, k, A, lda, x, incx)
    assert A.dtype == torch.float32 == x.dtype
    _check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=False)
    if n == 0:
        return
    with torch_device_fn.device(A.device):
        _launch_real_blocked(A, x, uplo, trans, diag, n, k, lda, False)


def dtbsv(uplo, trans, diag, n, k, A, lda, x, incx):
    if incx != 1 or k != 4:
        return _common_dtbsv(uplo, trans, diag, n, k, A, lda, x, incx)
    assert A.dtype == torch.float64 == x.dtype
    _check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=False)
    if n == 0:
        return
    with torch_device_fn.device(A.device):
        _launch_real_blocked(A, x, uplo, trans, diag, n, k, lda, True)


def ctbsv(uplo, trans, diag, n, k, A, lda, x, incx):
    if incx != 1 or k != 4:
        return _common_ctbsv(uplo, trans, diag, n, k, A, lda, x, incx)
    assert A.dtype == torch.complex64 == x.dtype
    _check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=True)
    if n == 0:
        return
    with torch_device_fn.device(A.device):
        _launch_complex64_blocked(A, x, uplo, trans, diag, n, k, lda)
