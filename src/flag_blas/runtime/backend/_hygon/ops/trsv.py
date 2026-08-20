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

from flag_blas.ops.level2.trsv import (
    CUBLAS_DIAG_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    _check_trsv,
    _forward,
    _mode_key,
    _row_major_dispatch,
    _trsv_flags,
)
from flag_blas.ops.level2.trsv import ctrsv as _common_ctrsv
from flag_blas.ops.level2.trsv import ctrsv_bwd_fused_kernel, ctrsv_fwd_fused_kernel
from flag_blas.ops.level2.trsv import dtrsv as _common_dtrsv
from flag_blas.ops.level2.trsv import (
    dtrsv_bwd_fused_kernel,
    dtrsv_bwd_inv_kernel,
    dtrsv_diag_inv_kernel,
    dtrsv_fwd_fused_kernel,
    dtrsv_fwd_inv_kernel,
)
from flag_blas.ops.level2.trsv import strsv as _common_strsv
from flag_blas.ops.level2.trsv import (
    strsv_bwd_fused_kernel,
    strsv_bwd_inv_kernel,
    strsv_bwd_kernel,
    strsv_diag_inv_kernel,
    strsv_fwd_fused_kernel,
    strsv_fwd_inv_kernel,
    strsv_n64_kernel,
)
from flag_blas.ops.level2.trsv import ztrsv as _common_ztrsv
from flag_blas.ops.level2.trsv import ztrsv_bwd_fused_kernel, ztrsv_fwd_fused_kernel
from flag_blas.runtime import torch_device_fn


@triton.jit
def _strsv_n64_rowmajor_kernel(
    a_ptr,
    x_ptr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    values = tl.load(x_ptr + offs)
    for row in tl.static_range(0, BLOCK_N):
        matrix_row = tl.load(
            a_ptr + row * BLOCK_N + offs,
            mask=offs < row,
            other=0.0,
            eviction_policy="evict_last",
        )
        current = tl.sum(tl.where(offs == row, values, 0.0), axis=0)
        current -= tl.sum(matrix_row * values, axis=0)
        if not UNIT:
            current /= tl.load(a_ptr + row * BLOCK_N + row)
        values = tl.where(offs == row, current, values)
    tl.store(x_ptr + offs, values)


def _physical_trsv_args(uplo, trans):
    internal_uplo, trans_flag, conj = _row_major_dispatch(uplo, trans)
    internal_trans = CUBLAS_OP_N if trans_flag == 0 else CUBLAS_OP_T
    return internal_uplo, internal_trans, conj


def _strsv_safe_inverse(uplo, diag, n, A, lda, x, incx):
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    forward = _forward(uplo, CUBLAS_OP_N)
    mode_key = _mode_key(uplo, 0, unit)
    bb = 64 if n >= 4096 else 32
    npanel = triton.cdiv(n, bb)
    lower_eff = 1 if uplo == CUBLAS_FILL_MODE_LOWER else 0
    flags = _trsv_flags(A.device)
    grid = (npanel,)

    if bb == 32 and n <= 512:
        if forward:
            strsv_fwd_fused_kernel[grid](
                A,
                x,
                flags,
                n,
                lda,
                incx,
                mode_key,
                TRANS=0,
                UNIT=unit,
                LOWER_EFF=lower_eff,
                BLOCK_N=bb,
            )
        else:
            strsv_bwd_fused_kernel[grid](
                A,
                x,
                flags,
                n,
                lda,
                incx,
                mode_key,
                TRANS=0,
                UNIT=unit,
                LOWER_EFF=lower_eff,
                BLOCK_N=bb,
            )
        return

    searched_n8192_nonunit = n == 8192 and unit == 0
    if searched_n8192_nonunit:
        kernel = strsv_fwd_fused_kernel if forward else strsv_bwd_fused_kernel
        kernel.fn.fn[grid](
            A,
            x,
            flags,
            n,
            lda,
            incx,
            mode_key,
            TRANS=0,
            UNIT=0,
            LOWER_EFF=lower_eff,
            BLOCK_N=64,
            CHUNK=1,
            num_warps=4,
            num_stages=3,
        )
        return

    dinv = torch.empty((npanel, bb, bb), dtype=torch.float32, device=A.device)
    searched_n4096_lu = n == 4096 and uplo == CUBLAS_FILL_MODE_LOWER and unit == 1
    if searched_n4096_lu:
        strsv_diag_inv_kernel[grid](
            A,
            dinv,
            n,
            lda,
            TRANS=0,
            UNIT=1,
            LOWER_EFF=1,
            BB=64,
            num_warps=4,
            num_stages=3,
        )
        strsv_fwd_inv_kernel.fn.fn[grid](
            A,
            x,
            dinv,
            flags,
            n,
            lda,
            incx,
            mode_key,
            TRANS=0,
            UNIT=1,
            BLOCK_N=64,
            CHUNK=1,
            num_warps=4,
            num_stages=3,
        )
        return

    searched_n4096_un = n == 4096 and uplo == CUBLAS_FILL_MODE_UPPER and unit == 0
    if searched_n4096_un:
        strsv_diag_inv_kernel[grid](
            A,
            dinv,
            n,
            lda,
            TRANS=0,
            UNIT=0,
            LOWER_EFF=0,
            BB=64,
            num_warps=4,
            num_stages=2,
        )
        strsv_bwd_inv_kernel.fn.fn[grid](
            A,
            x,
            dinv,
            flags,
            n,
            lda,
            incx,
            mode_key,
            TRANS=0,
            UNIT=0,
            BLOCK_N=64,
            CHUNK=1,
            num_warps=4,
            num_stages=2,
        )
        return

    searched_n2048_l = n == 2048 and uplo == CUBLAS_FILL_MODE_LOWER
    if searched_n2048_l:
        strsv_diag_inv_kernel[grid](
            A,
            dinv,
            n,
            lda,
            TRANS=0,
            UNIT=unit,
            LOWER_EFF=1,
            BB=32,
            num_warps=1,
            num_stages=1,
        )
        strsv_fwd_inv_kernel.fn.fn[grid](
            A,
            x,
            dinv,
            flags,
            n,
            lda,
            incx,
            mode_key,
            TRANS=0,
            UNIT=unit,
            BLOCK_N=32,
            CHUNK=1,
            num_warps=1,
            num_stages=1,
        )
        return

    searched_n2048_un = n == 2048 and uplo == CUBLAS_FILL_MODE_UPPER and unit == 0
    if searched_n2048_un:
        strsv_diag_inv_kernel[grid](
            A,
            dinv,
            n,
            lda,
            TRANS=0,
            UNIT=0,
            LOWER_EFF=0,
            BB=32,
            num_warps=1,
            num_stages=1,
        )
        strsv_bwd_inv_kernel.fn.fn[grid](
            A,
            x,
            dinv,
            flags,
            n,
            lda,
            incx,
            mode_key,
            TRANS=0,
            UNIT=0,
            BLOCK_N=32,
            CHUNK=1,
            num_warps=1,
            num_stages=1,
        )
        return

    searched_n4096_ln = n == 4096 and uplo == CUBLAS_FILL_MODE_LOWER and unit == 0
    if searched_n4096_ln:
        strsv_diag_inv_kernel[grid](
            A,
            dinv,
            n,
            lda,
            TRANS=0,
            UNIT=0,
            LOWER_EFF=1,
            BB=64,
            num_warps=4,
            num_stages=4,
        )
        strsv_fwd_inv_kernel.fn.fn[grid](
            A,
            x,
            dinv,
            flags,
            n,
            lda,
            incx,
            mode_key,
            TRANS=0,
            UNIT=0,
            BLOCK_N=64,
            CHUNK=1,
            num_warps=4,
            num_stages=4,
        )
        return

    searched_n8192_lu = n == 8192 and uplo == CUBLAS_FILL_MODE_LOWER and unit == 1
    if searched_n8192_lu:
        strsv_diag_inv_kernel[grid](
            A,
            dinv,
            n,
            lda,
            TRANS=0,
            UNIT=1,
            LOWER_EFF=1,
            BB=64,
            num_warps=4,
            num_stages=1,
        )
        strsv_fwd_inv_kernel.fn.fn[grid](
            A,
            x,
            dinv,
            flags,
            n,
            lda,
            incx,
            mode_key,
            TRANS=0,
            UNIT=1,
            BLOCK_N=64,
            CHUNK=1,
            num_warps=4,
            num_stages=1,
        )
        return

    strsv_diag_inv_kernel[grid](
        A,
        dinv,
        n,
        lda,
        TRANS=0,
        UNIT=unit,
        LOWER_EFF=lower_eff,
        BB=bb,
        num_warps=1 if bb == 32 else 2,
    )
    if forward:
        strsv_fwd_inv_kernel[grid](
            A,
            x,
            dinv,
            flags,
            n,
            lda,
            incx,
            mode_key,
            TRANS=0,
            UNIT=unit,
            BLOCK_N=bb,
        )
    else:
        strsv_bwd_inv_kernel[grid](
            A,
            x,
            dinv,
            flags,
            n,
            lda,
            incx,
            mode_key,
            TRANS=0,
            UNIT=unit,
            BLOCK_N=bb,
        )


def strsv(uplo, trans, diag, n, A, lda, x, incx):
    public_uplo, public_trans = uplo, trans
    uplo, trans, conj = _physical_trsv_args(uplo, trans)
    if conj:
        return _common_strsv(public_uplo, public_trans, diag, n, A, lda, x, incx)

    if (
        n == 64
        and public_uplo == CUBLAS_FILL_MODE_LOWER
        and public_trans == CUBLAS_OP_N
        and incx == 1
        and lda == n
    ):
        assert A.dtype == torch.float32 == x.dtype
        _check_trsv(
            A,
            x,
            public_uplo,
            public_trans,
            diag,
            n,
            lda,
            incx,
            complex_ok=False,
        )
        unit = int(diag == CUBLAS_DIAG_UNIT)
        with torch_device_fn.device(A.device):
            _strsv_n64_rowmajor_kernel[(1,)](
                A,
                x,
                UNIT=unit,
                BLOCK_N=64,
                num_warps=1,
                num_stages=1,
            )
        return

    optimized_n64_trans = (
        n == 64
        and trans == CUBLAS_OP_T
        and uplo == CUBLAS_FILL_MODE_LOWER
        and diag != CUBLAS_DIAG_UNIT
        and incx == 1
        and lda == n
    )
    if optimized_n64_trans:
        assert A.dtype == torch.float32 == x.dtype
        _check_trsv(
            A,
            x,
            public_uplo,
            public_trans,
            diag,
            n,
            lda,
            incx,
            complex_ok=False,
        )
        with torch_device_fn.device(A.device):
            flags = _trsv_flags(A.device)
            strsv_bwd_kernel.fn.fn[(8,)](
                A,
                x,
                flags,
                n,
                lda,
                incx,
                _mode_key(uplo, 1, 0),
                TRANS=1,
                UNIT=0,
                BLOCK_N=8,
                CHUNK=1,
                num_warps=2,
                num_stages=1,
            )
        return

    optimized_n64 = (
        n == 64
        and trans == CUBLAS_OP_N
        and uplo in (CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER)
        and diag != CUBLAS_DIAG_UNIT
        and incx == 1
        and lda == n
    )
    if optimized_n64:
        assert A.dtype == torch.float32 == x.dtype
        _check_trsv(
            A,
            x,
            public_uplo,
            public_trans,
            diag,
            n,
            lda,
            incx,
            complex_ok=False,
        )
        lower = uplo == CUBLAS_FILL_MODE_LOWER
        with torch_device_fn.device(A.device):
            strsv_n64_kernel[(1,)](
                A,
                x,
                n,
                LOWER=lower,
                UNIT=0,
                BLOCK_N=64,
                num_warps=1 if lower else 2,
                num_stages=1 if lower else 2,
            )
        return

    incompatible = (
        trans == CUBLAS_OP_N
        and incx == 1
        and lda == n
        and n >= 256
        and (
            (uplo == CUBLAS_FILL_MODE_LOWER and diag == CUBLAS_DIAG_UNIT)
            or (uplo == CUBLAS_FILL_MODE_UPPER and diag != CUBLAS_DIAG_UNIT)
            or (
                n in (2048, 4096, 8192)
                and uplo == CUBLAS_FILL_MODE_LOWER
                and diag != CUBLAS_DIAG_UNIT
            )
        )
    )
    if not incompatible:
        return _common_strsv(public_uplo, public_trans, diag, n, A, lda, x, incx)

    assert A.dtype == torch.float32 == x.dtype
    _check_trsv(
        A,
        x,
        public_uplo,
        public_trans,
        diag,
        n,
        lda,
        incx,
        complex_ok=False,
    )
    with torch_device_fn.device(A.device):
        _strsv_safe_inverse(uplo, diag, n, A, lda, x, incx)


def dtrsv(uplo, trans, diag, n, A, lda, x, incx):
    public_uplo, public_trans = uplo, trans
    uplo, trans, conj = _physical_trsv_args(uplo, trans)
    if conj:
        return _common_dtrsv(public_uplo, public_trans, diag, n, A, lda, x, incx)

    searched_n4096_ln = (
        n == 4096
        and trans == CUBLAS_OP_N
        and diag != CUBLAS_DIAG_UNIT
        and uplo == CUBLAS_FILL_MODE_LOWER
        and incx == 1
        and lda == n
    )
    searched_n4096_un = (
        n == 4096
        and trans == CUBLAS_OP_N
        and diag != CUBLAS_DIAG_UNIT
        and uplo == CUBLAS_FILL_MODE_UPPER
        and incx == 1
        and lda == n
    )
    searched_n8192_ln = (
        n == 8192
        and trans == CUBLAS_OP_N
        and diag != CUBLAS_DIAG_UNIT
        and uplo == CUBLAS_FILL_MODE_LOWER
        and incx == 1
        and lda == n
    )
    searched_n8192_un = (
        n == 8192
        and trans == CUBLAS_OP_N
        and diag != CUBLAS_DIAG_UNIT
        and uplo == CUBLAS_FILL_MODE_UPPER
        and incx == 1
        and lda == n
    )
    searched_n2048_ln = (
        n == 2048
        and trans == CUBLAS_OP_N
        and diag != CUBLAS_DIAG_UNIT
        and uplo == CUBLAS_FILL_MODE_LOWER
        and incx == 1
        and lda == n
    )
    searched_n2048_un = (
        n == 2048
        and trans == CUBLAS_OP_N
        and diag != CUBLAS_DIAG_UNIT
        and uplo == CUBLAS_FILL_MODE_UPPER
        and incx == 1
        and lda == n
    )
    searched_n4096_lt = (
        n == 4096
        and trans == CUBLAS_OP_T
        and diag != CUBLAS_DIAG_UNIT
        and uplo == CUBLAS_FILL_MODE_LOWER
        and incx == 1
        and lda == n
    )
    optimized = (
        (
            1024 <= n <= 8192
            and trans == CUBLAS_OP_N
            and diag == CUBLAS_DIAG_UNIT
            and uplo == CUBLAS_FILL_MODE_LOWER
            and incx == 1
            and lda == n
        )
        or searched_n2048_ln
        or searched_n2048_un
        or searched_n4096_ln
        or searched_n4096_un
        or searched_n4096_lt
        or searched_n8192_ln
        or searched_n8192_un
    )
    if not optimized:
        return _common_dtrsv(public_uplo, public_trans, diag, n, A, lda, x, incx)

    assert A.dtype == torch.float64 == x.dtype
    _check_trsv(
        A,
        x,
        public_uplo,
        public_trans,
        diag,
        n,
        lda,
        incx,
        complex_ok=False,
    )
    bb = 32
    npanel = triton.cdiv(n, bb)
    num_warps = 8 if n < 8192 else 4
    with torch_device_fn.device(A.device):
        flags = _trsv_flags(A.device)
        if searched_n4096_lt:
            dinv = torch.empty((npanel, bb, bb), dtype=torch.float64, device=A.device)
            dtrsv_diag_inv_kernel[(npanel,)](
                A,
                dinv,
                n,
                lda,
                TRANS=1,
                UNIT=0,
                LOWER_EFF=0,
                BB=32,
                num_warps=4,
                num_stages=3,
            )
            dtrsv_bwd_inv_kernel.fn.fn[(npanel,)](
                A,
                x,
                dinv,
                flags,
                n,
                lda,
                _mode_key(uplo, 1, 0),
                TRANS=1,
                UNIT=0,
                BLOCK_N=32,
                CHUNK=1,
                num_warps=4,
                num_stages=3,
            )
            return
        if searched_n2048_un:
            dinv = torch.empty((npanel, bb, bb), dtype=torch.float64, device=A.device)
            dtrsv_diag_inv_kernel[(npanel,)](
                A,
                dinv,
                n,
                lda,
                TRANS=0,
                UNIT=0,
                LOWER_EFF=0,
                BB=32,
                num_warps=4,
                num_stages=4,
            )
            dtrsv_bwd_inv_kernel.fn.fn[(npanel,)](
                A,
                x,
                dinv,
                flags,
                n,
                lda,
                _mode_key(uplo, 0, 0),
                TRANS=0,
                UNIT=0,
                BLOCK_N=32,
                CHUNK=1,
                num_warps=4,
                num_stages=4,
            )
            return
        if searched_n2048_ln:
            dinv = torch.empty((npanel, bb, bb), dtype=torch.float64, device=A.device)
            dtrsv_diag_inv_kernel[(npanel,)](
                A,
                dinv,
                n,
                lda,
                TRANS=0,
                UNIT=0,
                LOWER_EFF=1,
                BB=32,
                num_warps=4,
                num_stages=3,
            )
            dtrsv_fwd_inv_kernel.fn.fn[(npanel,)](
                A,
                x,
                dinv,
                flags,
                n,
                lda,
                _mode_key(uplo, 0, 0),
                TRANS=0,
                UNIT=0,
                BLOCK_N=32,
                CHUNK=1,
                num_warps=4,
                num_stages=3,
            )
            return
        if searched_n8192_un:
            dtrsv_bwd_fused_kernel.fn[(triton.cdiv(n, 64),)](
                A,
                x,
                flags,
                n,
                lda,
                _mode_key(uplo, 0, 0),
                TRANS=0,
                UNIT=0,
                LOWER_EFF=0,
                BLOCK_N=64,
                CHUNK=1,
                ROWLOAD=0,
                num_warps=4,
                num_stages=4,
            )
            return
        if searched_n8192_ln:
            dtrsv_fwd_fused_kernel.fn[(triton.cdiv(n, 64),)](
                A,
                x,
                flags,
                n,
                lda,
                _mode_key(uplo, 0, 0),
                TRANS=0,
                UNIT=0,
                LOWER_EFF=1,
                BLOCK_N=64,
                CHUNK=1,
                ROWLOAD=0,
                num_warps=4,
                num_stages=1,
            )
            return
        if searched_n4096_un:
            dinv = torch.empty((npanel, bb, bb), dtype=torch.float64, device=A.device)
            dtrsv_diag_inv_kernel[(npanel,)](
                A,
                dinv,
                n,
                lda,
                TRANS=0,
                UNIT=0,
                LOWER_EFF=0,
                BB=32,
                num_warps=4,
                num_stages=3,
            )
            dtrsv_bwd_inv_kernel.fn.fn[(npanel,)](
                A,
                x,
                dinv,
                flags,
                n,
                lda,
                _mode_key(uplo, 0, 0),
                TRANS=0,
                UNIT=0,
                BLOCK_N=32,
                CHUNK=1,
                num_warps=4,
                num_stages=3,
            )
            return
        if n == 8192 and diag == CUBLAS_DIAG_UNIT:
            dtrsv_fwd_fused_kernel.fn[(triton.cdiv(n, 64),)](
                A,
                x,
                flags,
                n,
                lda,
                _mode_key(uplo, 0, 1),
                TRANS=0,
                UNIT=1,
                LOWER_EFF=1,
                BLOCK_N=64,
                CHUNK=1,
                ROWLOAD=0,
                num_warps=4,
                num_stages=2,
            )
            return
        if searched_n4096_ln:
            dinv = torch.empty((npanel, bb, bb), dtype=torch.float64, device=A.device)
            dtrsv_diag_inv_kernel[(npanel,)](
                A,
                dinv,
                n,
                lda,
                TRANS=0,
                UNIT=0,
                LOWER_EFF=1,
                BB=32,
                num_warps=4,
                num_stages=3,
            )
            dtrsv_fwd_inv_kernel.fn.fn[(npanel,)](
                A,
                x,
                dinv,
                flags,
                n,
                lda,
                _mode_key(uplo, 0, 0),
                TRANS=0,
                UNIT=0,
                BLOCK_N=32,
                CHUNK=1,
                num_warps=4,
                num_stages=3,
            )
            return
        if n == 1024:
            dinv = torch.empty((npanel, bb, bb), dtype=torch.float64, device=A.device)
            dtrsv_diag_inv_kernel[(npanel,)](
                A,
                dinv,
                n,
                lda,
                TRANS=0,
                UNIT=1,
                LOWER_EFF=1,
                BB=32,
                num_warps=4,
                num_stages=3,
            )
            dtrsv_fwd_inv_kernel.fn.fn[(npanel,)](
                A,
                x,
                dinv,
                flags,
                n,
                lda,
                _mode_key(uplo, 0, 1),
                TRANS=0,
                UNIT=1,
                BLOCK_N=32,
                CHUNK=1,
                num_warps=4,
                num_stages=3,
            )
            return
        if n == 2048:
            dinv = torch.empty((npanel, bb, bb), dtype=torch.float64, device=A.device)
            dtrsv_diag_inv_kernel[(npanel,)](
                A,
                dinv,
                n,
                lda,
                TRANS=0,
                UNIT=1,
                LOWER_EFF=1,
                BB=32,
                num_warps=4,
                num_stages=4,
            )
            dtrsv_fwd_inv_kernel.fn.fn[(npanel,)](
                A,
                x,
                dinv,
                flags,
                n,
                lda,
                _mode_key(uplo, 0, 1),
                TRANS=0,
                UNIT=1,
                BLOCK_N=32,
                CHUNK=1,
                num_warps=4,
                num_stages=4,
            )
            return
        if n == 4096:
            dinv = torch.empty((npanel, bb, bb), dtype=torch.float64, device=A.device)
            dtrsv_diag_inv_kernel[(npanel,)](
                A,
                dinv,
                n,
                lda,
                TRANS=0,
                UNIT=1,
                LOWER_EFF=1,
                BB=32,
                num_warps=4,
                num_stages=1,
            )
            dtrsv_fwd_inv_kernel.fn.fn[(npanel,)](
                A,
                x,
                dinv,
                flags,
                n,
                lda,
                _mode_key(uplo, 0, 1),
                TRANS=0,
                UNIT=1,
                BLOCK_N=32,
                CHUNK=1,
                num_warps=4,
                num_stages=1,
            )
            return
        dtrsv_fwd_fused_kernel[(npanel,)](
            A,
            x,
            flags,
            n,
            lda,
            _mode_key(uplo, 0, 1),
            TRANS=0,
            UNIT=1,
            LOWER_EFF=1,
            BLOCK_N=bb,
            CHUNK=1,
            ROWLOAD=1 if n <= 4096 else 0,
            num_warps=num_warps,
        )


def ctrsv(uplo, trans, diag, n, A, lda, x, incx):
    public_uplo, public_trans = uplo, trans
    uplo, trans, conj = _physical_trsv_args(uplo, trans)
    if conj:
        return _common_ctrsv(public_uplo, public_trans, diag, n, A, lda, x, incx)

    searched_n4096_public_ln = (
        n == 4096
        and public_uplo == CUBLAS_FILL_MODE_LOWER
        and public_trans == CUBLAS_OP_N
        and diag != CUBLAS_DIAG_UNIT
        and incx == 1
        and lda == n
    )

    searched_n4096_lt = (
        n == 4096
        and trans == CUBLAS_OP_T
        and diag != CUBLAS_DIAG_UNIT
        and uplo == CUBLAS_FILL_MODE_LOWER
        and incx == 1
        and lda == n
    )
    searched_n4096_lc = (
        n == 4096
        and trans == CUBLAS_OP_C
        and diag != CUBLAS_DIAG_UNIT
        and uplo == CUBLAS_FILL_MODE_LOWER
        and incx == 1
        and lda == n
    )
    searched_n8192_lt = (
        n == 8192
        and trans == CUBLAS_OP_T
        and diag != CUBLAS_DIAG_UNIT
        and uplo == CUBLAS_FILL_MODE_LOWER
        and incx == 1
        and lda == n
    )
    searched_n8192_lc = (
        n == 8192
        and trans == CUBLAS_OP_C
        and diag != CUBLAS_DIAG_UNIT
        and uplo == CUBLAS_FILL_MODE_LOWER
        and incx == 1
        and lda == n
    )
    optimized = (
        (
            _forward(uplo, trans)
            and diag == CUBLAS_DIAG_UNIT
            and incx == 1
            and lda == n
            and (n == 256 or 512 <= n <= 4096)
        )
        or searched_n4096_public_ln
        or searched_n4096_lt
        or searched_n4096_lc
        or searched_n8192_lt
        or searched_n8192_lc
    )
    if not optimized:
        return _common_ctrsv(public_uplo, public_trans, diag, n, A, lda, x, incx)

    assert A.dtype == torch.complex64 == x.dtype
    _check_trsv(
        A,
        x,
        public_uplo,
        public_trans,
        diag,
        n,
        lda,
        incx,
        complex_ok=True,
    )
    unit = int(diag == CUBLAS_DIAG_UNIT)
    trans_flag = int(trans != CUBLAS_OP_N)
    conj = int(trans == CUBLAS_OP_C)
    lower_eff = int((uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans_flag == 1))
    mode_key = _mode_key(uplo, trans_flag, unit) | (conj << 8)
    bb = 32
    npanel = triton.cdiv(n, bb)
    chunk = 4 if n == 256 else 1
    num_warps = 4
    with torch_device_fn.device(A.device):
        A_real = torch.view_as_real(A)
        x_real = torch.view_as_real(x)
        flags = _trsv_flags(A.device)
        if searched_n4096_public_ln:
            ctrsv_fwd_fused_kernel.fn[(npanel,)](
                A_real,
                x_real,
                flags,
                n,
                lda,
                mode_key,
                TRANS=1,
                UNIT=0,
                CONJ=0,
                LOWER_EFF=1,
                BLOCK_N=32,
                CHUNK=1,
                ROWLOAD=0,
                INV_DOT=0,
                VEC64=1,
                num_warps=8,
                num_stages=2,
            )
            return
        if searched_n4096_lc or searched_n8192_lc:
            ctrsv_bwd_fused_kernel.fn[(npanel,)](
                A_real,
                x_real,
                flags,
                n,
                lda,
                mode_key,
                TRANS=1,
                UNIT=0,
                CONJ=1,
                LOWER_EFF=0,
                BLOCK_N=32,
                CHUNK=1,
                ROWLOAD=0,
                INV_DOT=0,
                num_warps=8,
                num_stages=1,
            )
            return
        if searched_n4096_lt or searched_n8192_lt:
            ctrsv_bwd_fused_kernel.fn[(npanel,)](
                A_real,
                x_real,
                flags,
                n,
                lda,
                mode_key,
                TRANS=1,
                UNIT=0,
                CONJ=0,
                LOWER_EFF=0,
                BLOCK_N=32,
                CHUNK=1,
                ROWLOAD=0,
                INV_DOT=0,
                num_warps=8,
                num_stages=1 if searched_n8192_lt else 4,
            )
            return
        if (
            n == 4096
            and public_uplo == CUBLAS_FILL_MODE_LOWER
            and public_trans == CUBLAS_OP_N
            and diag == CUBLAS_DIAG_UNIT
            and incx == 1
            and lda == n
        ):
            ctrsv_fwd_fused_kernel.fn[(npanel,)](
                A_real,
                x_real,
                flags,
                n,
                lda,
                mode_key,
                TRANS=1,
                UNIT=1,
                CONJ=0,
                LOWER_EFF=1,
                BLOCK_N=32,
                CHUNK=1,
                ROWLOAD=0,
                INV_DOT=0,
                VEC64=0,
                num_warps=8,
                num_stages=2,
            )
            return
        if (
            n == 256
            and public_uplo == CUBLAS_FILL_MODE_LOWER
            and public_trans == CUBLAS_OP_N
        ):
            ctrsv_fwd_fused_kernel[(npanel,)](
                A_real,
                x_real,
                flags,
                n,
                lda,
                mode_key,
                TRANS=trans_flag,
                UNIT=unit,
                CONJ=conj,
                LOWER_EFF=lower_eff,
                BLOCK_N=32,
                CHUNK=1,
                ROWLOAD=0,
                INV_DOT=0,
                VEC64=1,
                num_warps=4,
                num_stages=2,
            )
            return
        ctrsv_fwd_fused_kernel[(npanel,)](
            A_real,
            x_real,
            flags,
            n,
            lda,
            mode_key,
            TRANS=trans_flag,
            UNIT=unit,
            CONJ=conj,
            LOWER_EFF=lower_eff,
            BLOCK_N=bb,
            CHUNK=chunk,
            ROWLOAD=1,
            INV_DOT=0,
            VEC64=0,
            num_warps=num_warps,
        )


def ztrsv(uplo, trans, diag, n, A, lda, x, incx):
    public_uplo, public_trans = uplo, trans
    uplo, trans, conj = _physical_trsv_args(uplo, trans)
    if conj:
        return _common_ztrsv(public_uplo, public_trans, diag, n, A, lda, x, incx)

    contiguous = incx == 1 and lda == n
    nonunit = diag != CUBLAS_DIAG_UNIT
    config = None
    if contiguous:
        if (
            nonunit
            and uplo == CUBLAS_FILL_MODE_UPPER
            and trans == CUBLAS_OP_T
            and n == 1024
        ):
            config = (16, 1, 1, 0, 1, 3)
        elif (
            nonunit
            and uplo == CUBLAS_FILL_MODE_UPPER
            and trans == CUBLAS_OP_T
            and n in (512, 8192)
        ):
            config = (16, 1, 0, 0, 8, 1) if n == 8192 else (16, 1, 0, 1, 4, 4)
        elif (
            not nonunit
            and uplo == CUBLAS_FILL_MODE_UPPER
            and trans == CUBLAS_OP_T
            and n == 1024
        ):
            config = (16, 1, 1, 1, 1, 3)
        elif (
            not nonunit
            and uplo == CUBLAS_FILL_MODE_UPPER
            and trans == CUBLAS_OP_T
            and n == 512
        ):
            config = (16, 1, 1, 0, 1, 2)
        elif (
            not nonunit
            and uplo == CUBLAS_FILL_MODE_UPPER
            and trans == CUBLAS_OP_T
            and (256 <= n <= 512 or n >= 8192)
        ):
            config = (16, 1, 0, 1, 4, 3)
        elif (
            n == 512
            and uplo == CUBLAS_FILL_MODE_LOWER
            and trans == CUBLAS_OP_N
            and not nonunit
        ):
            config = (16, 1, 1, 0, 4, 1)
        elif n == 1024:
            if uplo == CUBLAS_FILL_MODE_LOWER and trans == CUBLAS_OP_C and nonunit:
                config = (16, 1, 0, 0, 1, 3)
            elif uplo == CUBLAS_FILL_MODE_LOWER and trans == CUBLAS_OP_T and nonunit:
                config = (16, 1, 0, 0, 1, 1)
            elif uplo == CUBLAS_FILL_MODE_LOWER and trans == CUBLAS_OP_N:
                config = (16, 1, 0 if nonunit else 1, 0, 1, 1 if nonunit else 2)
            elif uplo == CUBLAS_FILL_MODE_UPPER and trans == CUBLAS_OP_N and nonunit:
                config = (16, 1, 0, 0, 1, 4)
        elif n == 2048 and uplo == CUBLAS_FILL_MODE_LOWER:
            if trans == CUBLAS_OP_N and not nonunit:
                config = (32, 1, 1, 0, 4, 1)
            elif trans == CUBLAS_OP_T and nonunit:
                config = (16, 1, 0, 0, 1, 3)
        elif n == 8192 and uplo == CUBLAS_FILL_MODE_LOWER and nonunit:
            if trans == CUBLAS_OP_C:
                config = (16, 1, 0, 1, 4, 2)
            elif trans == CUBLAS_OP_T:
                config = (16, 1, 0, 1, 4, 4)

    if config is None:
        return _common_ztrsv(public_uplo, public_trans, diag, n, A, lda, x, incx)

    assert A.dtype == torch.complex128 == x.dtype
    _check_trsv(
        A,
        x,
        public_uplo,
        public_trans,
        diag,
        n,
        lda,
        incx,
        complex_ok=True,
    )
    unit = int(diag == CUBLAS_DIAG_UNIT)
    trans_flag = int(trans != CUBLAS_OP_N)
    conj = int(trans == CUBLAS_OP_C)
    lower_eff = int((uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans_flag == 1))
    mode_key = _mode_key(uplo, trans_flag, unit) | (conj << 8)
    block_n, chunk, rowload, inv_dot, num_warps, num_stages = config
    npanel = triton.cdiv(n, block_n)
    kernel = ztrsv_fwd_fused_kernel if _forward(uplo, trans) else ztrsv_bwd_fused_kernel

    with torch_device_fn.device(A.device):
        A_real = torch.view_as_real(A)
        x_real = torch.view_as_real(x)
        flags = _trsv_flags(A.device)
        kernel.fn[(npanel,)](
            A_real,
            x_real,
            flags,
            n,
            lda,
            mode_key,
            TRANS=trans_flag,
            UNIT=unit,
            CONJ=conj,
            LOWER_EFF=lower_eff,
            BLOCK_N=block_n,
            CHUNK=chunk,
            ROWLOAD=rowload,
            INV_DOT=inv_dot,
            num_warps=num_warps,
            num_stages=num_stages,
        )
