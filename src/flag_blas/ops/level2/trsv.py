import logging
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2._constants import (
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_DIAG_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]


_TRSV_KEY = ["n", "mode_key"]
_TRSV_RESTORE = ["x_ptr", "flag_ptr"]

_STRSV_INV_MIN_N = 256
_STRSV_INV_BB64_MIN_N = 2048
_STRSV_FUSE_MAX_N = 512
_DTRSV_INV_BB64_MIN_N = 1 << 30
_DTRSV_FUSE_MAX_N = 8192
_DTRSV_ROWLOAD_MAX_N = 4096
_ZTRSV_ROWLOAD_MAX_N = 8192

_DTRSV_FUSED_FORCE = None


def _dtrsv_fused_cfg(forward, n, bb, unit):
    if _DTRSV_FUSED_FORCE is not None:
        return _DTRSV_FUSED_FORCE
    if forward and unit and n >= 8192:
        return (1, 2)
    return (1, 4)


_ZTRSV_FUSED_FORCE = None


def _ztrsv_fused_cfg(forward, n, bb, unit):
    if _ZTRSV_FUSED_FORCE is not None:
        return _ZTRSV_FUSED_FORCE
    if unit and n <= 64:
        return (1, 2)
    return (1, 4)


_CTRSV_FUSED_FORCE = None


def _ctrsv_fused_cfg(forward, n, bb, unit):
    if _CTRSV_FUSED_FORCE is not None:
        return _CTRSV_FUSED_FORCE
    if unit:
        if n <= 64:
            return (1, 1)
        if n > _DTRSV_ROWLOAD_MAX_N:
            return (1, 4)
        return (4, 4)
    if forward:
        if n <= 256:
            return (4, 4)
        return (1, 4)
    if n <= 64:
        return (1, 2)
    return (1, 4)


@triton.jit
def strsv_n64_kernel(
    a_ptr,
    x_ptr,
    n,
    LOWER: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n
    sx = tl.load(x_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_last")

    if LOWER:
        for step in tl.static_range(0, BLOCK_N):
            active = step < n
            cols = offs
            col_mask = (offs < step) & mask
            a_vals = tl.load(
                a_ptr + step + cols * n,
                mask=col_mask & active,
                other=0.0,
                eviction_policy="evict_last",
            )
            cur = tl.sum(tl.where(offs == step, sx, 0.0), axis=0) - tl.sum(
                a_vals * sx, axis=0
            )
            if not UNIT:
                diag = tl.load(
                    a_ptr + step + step * n,
                    mask=active,
                    other=1.0,
                    eviction_policy="evict_last",
                )
                cur = cur / diag
            sx = tl.where(offs == step, cur, sx)
    else:
        for step in tl.static_range(0, BLOCK_N):
            row = BLOCK_N - 1 - step
            active = row < n
            cols = offs
            col_mask = (offs > row) & mask
            a_vals = tl.load(
                a_ptr + row + cols * n,
                mask=col_mask & active,
                other=0.0,
                eviction_policy="evict_last",
            )
            cur = tl.sum(tl.where(offs == row, sx, 0.0), axis=0) - tl.sum(
                a_vals * sx, axis=0
            )
            if not UNIT:
                diag = tl.load(
                    a_ptr + row + row * n,
                    mask=active,
                    other=1.0,
                    eviction_policy="evict_last",
                )
                cur = cur / diag
            sx = tl.where(offs == row, cur, sx)

    tl.store(x_ptr + offs, sx, mask=mask)


@triton.jit
def _diag_block_inv_unit_lower_rowload(a_ptr, s, offs, BLOCK_N: tl.constexpr):
    X = tl.where(offs[:, None] == offs[None, :], 1.0, 0.0)
    for i in tl.static_range(0, BLOCK_N):
        mrow = tl.load(
            a_ptr + (s + i) + (s + offs) * 256,
            mask=offs < i,
            other=0.0,
            eviction_policy="evict_last",
        )
        contrib = tl.sum(mrow[:, None] * X, axis=0)
        ei = tl.where(offs == i, 1.0, 0.0)
        xi = ei - contrib
        X = tl.where(offs[:, None] == i, xi[None, :], X)
    return X


@triton.jit
def _diag_block_inv_nonunit_upper_rowload(a_ptr, s, offs, BLOCK_N: tl.constexpr):
    X = tl.where(offs[:, None] == offs[None, :], 1.0, 0.0)
    for ii in tl.static_range(0, BLOCK_N):
        i = BLOCK_N - 1 - ii
        mrow = tl.load(
            a_ptr + (s + i) + (s + offs) * 256,
            mask=offs > i,
            other=0.0,
            eviction_policy="evict_last",
        )
        contrib = tl.sum(mrow[:, None] * X, axis=0)
        ei = tl.where(offs == i, 1.0, 0.0)
        diag = tl.load(a_ptr + (s + i) + (s + i) * 256, eviction_policy="evict_last")
        xi = (ei - contrib) / diag
        X = tl.where(offs[:, None] == i, xi[None, :], X)
    return X


@triton.jit
def dtrsv_n64_kernel(
    a_ptr,
    x_ptr,
    n,
    LOWER: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n
    sx = tl.load(x_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_last")
    if LOWER:
        for step in tl.static_range(0, BLOCK_N):
            active = step < n
            col_mask = (offs < step) & mask
            a_vals = tl.load(
                a_ptr + step + offs * n,
                mask=col_mask & active,
                other=0.0,
                eviction_policy="evict_last",
            )
            cur = tl.sum(tl.where(offs == step, sx, 0.0), axis=0) - tl.sum(
                a_vals * sx, axis=0
            )
            if not UNIT:
                diag = tl.load(
                    a_ptr + step + step * n,
                    mask=active,
                    other=1.0,
                    eviction_policy="evict_last",
                )
                cur = cur / diag
            sx = tl.where(offs == step, cur, sx)
    else:
        for step in tl.static_range(0, BLOCK_N):
            row = BLOCK_N - 1 - step
            active = row < n
            col_mask = (offs > row) & mask
            a_vals = tl.load(
                a_ptr + row + offs * n,
                mask=col_mask & active,
                other=0.0,
                eviction_policy="evict_last",
            )
            cur = tl.sum(tl.where(offs == row, sx, 0.0), axis=0) - tl.sum(
                a_vals * sx, axis=0
            )
            if not UNIT:
                diag = tl.load(
                    a_ptr + row + row * n,
                    mask=active,
                    other=1.0,
                    eviction_policy="evict_last",
                )
                cur = cur / diag
            sx = tl.where(offs == row, cur, sx)
    tl.store(x_ptr + offs, sx, mask=mask)


@triton.jit
def strsv_lu256_rowinv_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    rows = row_start + offs
    dinv = _diag_block_inv_unit_lower_rowload(a_ptr, row_start, offs, BLOCK_N)
    sx = tl.load(x_ptr + rows, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = pid_lo * BLOCK_N
        c1 = pid_hi * BLOCK_N
        kcols = c0 + koffs
        col_mask = kcols < c1
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        a_vals = tl.load(
            a_ptr + rows[:, None] + kcols[None, :] * 256,
            mask=col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@triton.jit
def strsv_un256_rowinv_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels: tl.constexpr = 8
    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    rows = row_start + offs
    dinv = _diag_block_inv_nonunit_upper_rowload(a_ptr, row_start, offs, BLOCK_N)
    sx = tl.load(x_ptr + rows, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = kcols < c1
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        a_vals = tl.load(
            a_ptr + rows[:, None] + kcols[None, :] * 256,
            mask=col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("strsv_fwd"),
    key=_TRSV_KEY,
    restore_value=_TRSV_RESTORE,
)
@triton.jit
def strsv_fwd_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    INCX,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)

    row_start = pid * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size

    sx = tl.load(
        x_ptr + rows * INCX, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = pid_lo * BLOCK_N
        c1 = pid_hi * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols * INCX, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        mask2d = row_mask[:, None] & col_mask[None, :]
        a_vals = tl.load(
            a_ptr + a_off, mask=mask2d, other=0.0, eviction_policy="evict_first"
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    if TRANS == 1:
        rr = row_start + offs
        blk_off = rr[None, :] + rr[:, None] * LDA
        blk_mask = row_mask[:, None] & row_mask[None, :]
        diag_blk = tl.load(
            a_ptr + blk_off, mask=blk_mask, other=0.0, eviction_policy="evict_last"
        )
        for step in tl.static_range(0, BLOCK_N):
            active = step < size
            row_sel = offs == step
            arow = tl.sum(tl.where(row_sel[:, None], diag_blk, 0.0), axis=0)
            dval = tl.sum(tl.where(row_sel, arow, 0.0), axis=0)
            col_mask = (offs < step) & row_mask
            arow = tl.where(col_mask, arow, 0.0)
            inner_sum = tl.sum(arow * sx, axis=0)
            cur = tl.sum(tl.where(row_sel, sx, 0.0), axis=0) - inner_sum
            if not UNIT:
                cur = cur / tl.where(active, dval, 1.0)
            sx = tl.where(row_sel, cur, sx)
    else:
        for step in tl.static_range(0, BLOCK_N):
            active = step < size
            row = row_start + step
            cols = row_start + offs
            col_mask = (offs < step) & row_mask
            a_off = row + cols * LDA
            a_vals = tl.load(a_ptr + a_off, mask=col_mask & active, other=0.0)
            inner_sum = tl.sum(a_vals * sx, axis=0)
            cur = tl.sum(tl.where(offs == step, sx, 0.0), axis=0) - inner_sum
            if not UNIT:
                diag = tl.load(a_ptr + row + row * LDA, mask=active, other=1.0)
                cur = cur / diag
            sx = tl.where(offs == step, cur, sx)

    tl.store(x_ptr + rows * INCX, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("strsv_bwd"),
    key=_TRSV_KEY,
    restore_value=_TRSV_RESTORE,
)
@triton.jit
def strsv_bwd_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    INCX,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels = tl.cdiv(n, BLOCK_N)

    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size

    sx = tl.load(
        x_ptr + rows * INCX, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols * INCX, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        mask2d = row_mask[:, None] & col_mask[None, :]
        a_vals = tl.load(
            a_ptr + a_off, mask=mask2d, other=0.0, eviction_policy="evict_first"
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    if TRANS == 1:
        rr = row_start + offs
        blk_off = rr[None, :] + rr[:, None] * LDA
        blk_mask = row_mask[:, None] & row_mask[None, :]
        diag_blk = tl.load(
            a_ptr + blk_off, mask=blk_mask, other=0.0, eviction_policy="evict_last"
        )
        for step in tl.static_range(0, BLOCK_N):
            local = BLOCK_N - 1 - step
            active = local < size
            row_sel = offs == local
            arow = tl.sum(tl.where(row_sel[:, None], diag_blk, 0.0), axis=0)
            dval = tl.sum(tl.where(row_sel, arow, 0.0), axis=0)
            col_mask = (offs > local) & (offs < size)
            arow = tl.where(col_mask, arow, 0.0)
            inner_sum = tl.sum(arow * sx, axis=0)
            cur = tl.sum(tl.where(row_sel, sx, 0.0), axis=0) - inner_sum
            if not UNIT:
                cur = cur / tl.where(active, dval, 1.0)
            sx = tl.where(row_sel, cur, sx)
    else:
        for step in tl.static_range(0, BLOCK_N):
            local = BLOCK_N - 1 - step
            active = local < size
            row = row_start + local
            cols = row_start + offs
            col_mask = (offs > local) & (offs < size)
            a_off = row + cols * LDA
            a_vals = tl.load(a_ptr + a_off, mask=col_mask & active, other=0.0)
            inner_sum = tl.sum(a_vals * sx, axis=0)
            cur = tl.sum(tl.where(offs == local, sx, 0.0), axis=0) - inner_sum
            if not UNIT:
                diag = tl.load(a_ptr + row + row * LDA, mask=active, other=1.0)
                cur = cur / diag
            sx = tl.where(offs == local, cur, sx)

    tl.store(x_ptr + rows * INCX, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@triton.jit
def strsv_diag_inv_kernel(
    a_ptr,
    dinv_ptr,
    n,
    LDA,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BB: tl.constexpr,
):
    blk = tl.program_id(0)
    offs = tl.arange(0, BB)
    s = blk * BB
    grow = s + offs
    rmask = grow < n
    if TRANS == 0:
        m_off = (s + offs)[:, None] + (s + offs)[None, :] * LDA
    else:
        m_off = (s + offs)[None, :] + (s + offs)[:, None] * LDA
    mmask = rmask[:, None] & rmask[None, :]
    M = tl.load(a_ptr + m_off, mask=mmask, other=0.0)
    if not UNIT:
        diag_each = tl.sum(tl.where(offs[:, None] == offs[None, :], M, 0.0), axis=1)
        safe_diag = tl.where(rmask, diag_each, 1.0)
    X = tl.where(offs[:, None] == offs[None, :], 1.0, 0.0)
    if LOWER_EFF:
        for i in tl.range(0, BB):
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs < i, mrow, 0.0)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, 1.0, 0.0)
            if UNIT:
                xi = ei - contrib
            else:
                di = tl.sum(tl.where(offs == i, safe_diag, 0.0), axis=0)
                xi = (ei - contrib) / di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    else:
        for ii in tl.range(0, BB):
            i = BB - 1 - ii
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs > i, mrow, 0.0)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, 1.0, 0.0)
            if UNIT:
                xi = ei - contrib
            else:
                di = tl.sum(tl.where(offs == i, safe_diag, 0.0), axis=0)
                xi = (ei - contrib) / di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    d_off = blk * BB * BB + offs[:, None] * BB + offs[None, :]
    tl.store(dinv_ptr + d_off, X, mask=mmask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("strsv_fwd_inv"),
    key=_TRSV_KEY,
    restore_value=_TRSV_RESTORE,
)
@triton.jit
def strsv_fwd_inv_kernel(
    a_ptr,
    x_ptr,
    dinv_ptr,
    flag_ptr,
    n,
    LDA,
    INCX,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size
    sx = tl.load(
        x_ptr + rows * INCX, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = pid_lo * BLOCK_N
        c1 = pid_hi * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols * INCX, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        mask2d = row_mask[:, None] & col_mask[None, :]
        a_vals = tl.load(
            a_ptr + a_off, mask=mask2d, other=0.0, eviction_policy="evict_first"
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    d_off = pid * BLOCK_N * BLOCK_N + offs[:, None] * BLOCK_N + offs[None, :]
    dmask = row_mask[:, None] & row_mask[None, :]
    dinv = tl.load(dinv_ptr + d_off, mask=dmask, other=0.0)
    sx = tl.sum(dinv * sx[None, :], axis=1)

    tl.store(x_ptr + rows * INCX, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("strsv_bwd_inv"),
    key=_TRSV_KEY,
    restore_value=_TRSV_RESTORE,
)
@triton.jit
def strsv_bwd_inv_kernel(
    a_ptr,
    x_ptr,
    dinv_ptr,
    flag_ptr,
    n,
    LDA,
    INCX,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels = tl.cdiv(n, BLOCK_N)

    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size
    sx = tl.load(
        x_ptr + rows * INCX, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols * INCX, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        mask2d = row_mask[:, None] & col_mask[None, :]
        a_vals = tl.load(
            a_ptr + a_off, mask=mask2d, other=0.0, eviction_policy="evict_first"
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    d_off = blk * BLOCK_N * BLOCK_N + offs[:, None] * BLOCK_N + offs[None, :]
    dmask = row_mask[:, None] & row_mask[None, :]
    dinv = tl.load(dinv_ptr + d_off, mask=dmask, other=0.0)
    sx = tl.sum(dinv * sx[None, :], axis=1)

    tl.store(x_ptr + rows * INCX, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@triton.jit
def _diag_block_inv(
    a_ptr,
    s,
    LDA,
    offs,
    row_mask,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if TRANS == 0:
        m_off = (s + offs)[:, None] + (s + offs)[None, :] * LDA
    else:
        m_off = (s + offs)[None, :] + (s + offs)[:, None] * LDA
    mmask = row_mask[:, None] & row_mask[None, :]
    M = tl.load(a_ptr + m_off, mask=mmask, other=0.0)
    if not UNIT:
        diag_each = tl.sum(tl.where(offs[:, None] == offs[None, :], M, 0.0), axis=1)
        safe_diag = tl.where(row_mask, diag_each, 1.0)
    X = tl.where(offs[:, None] == offs[None, :], 1.0, 0.0)
    if LOWER_EFF:
        for i in tl.range(0, BLOCK_N):
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs < i, mrow, 0.0)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, 1.0, 0.0)
            if UNIT:
                xi = ei - contrib
            else:
                di = tl.sum(tl.where(offs == i, safe_diag, 0.0), axis=0)
                xi = (ei - contrib) / di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    else:
        for ii in tl.range(0, BLOCK_N):
            i = BLOCK_N - 1 - ii
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs > i, mrow, 0.0)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, 1.0, 0.0)
            if UNIT:
                xi = ei - contrib
            else:
                di = tl.sum(tl.where(offs == i, safe_diag, 0.0), axis=0)
                xi = (ei - contrib) / di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    return X


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("strsv_fwd_fused"),
    key=_TRSV_KEY,
    restore_value=_TRSV_RESTORE,
)
@triton.jit
def strsv_fwd_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    INCX,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size
    dinv = _diag_block_inv(
        a_ptr, row_start, LDA, offs, row_mask, TRANS, UNIT, LOWER_EFF, BLOCK_N
    )
    sx = tl.load(
        x_ptr + rows * INCX, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = pid_lo * BLOCK_N
        c1 = pid_hi * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols * INCX, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        mask2d = row_mask[:, None] & col_mask[None, :]
        a_vals = tl.load(
            a_ptr + a_off, mask=mask2d, other=0.0, eviction_policy="evict_first"
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows * INCX, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("strsv_bwd_fused"),
    key=_TRSV_KEY,
    restore_value=_TRSV_RESTORE,
)
@triton.jit
def strsv_bwd_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    INCX,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels = tl.cdiv(n, BLOCK_N)

    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size
    dinv = _diag_block_inv(
        a_ptr, row_start, LDA, offs, row_mask, TRANS, UNIT, LOWER_EFF, BLOCK_N
    )
    sx = tl.load(
        x_ptr + rows * INCX, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols * INCX, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        mask2d = row_mask[:, None] & col_mask[None, :]
        a_vals = tl.load(
            a_ptr + a_off, mask=mask2d, other=0.0, eviction_policy="evict_first"
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows * INCX, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@triton.jit
def dtrsv_panel_kernel(
    a_ptr,
    x_ptr,
    start,
    end,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    FORWARD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    size = end - start
    for step in tl.static_range(0, BLOCK_N):
        if FORWARD:
            i = start + step
            active = step < size
            j = start + offs
            mask = (offs < step) & (offs < size)
        else:
            i = end - 1 - step
            active = step < size
            j = end - 1 - offs
            mask = (offs < step) & (offs < size)
        mask = mask & active
        if TRANS == 0:
            a_off = i + j * LDA
        else:
            a_off = j + i * LDA
        a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + j * INCX, mask=mask, other=0.0)
        value = tl.load(x_ptr + i * INCX, mask=active, other=0.0) - tl.sum(
            a_vals * x_vals, axis=0
        )
        if not UNIT:
            diag = tl.load(a_ptr + i + i * LDA, mask=active, other=1.0)
            value = value / diag
        tl.store(x_ptr + i * INCX, value, mask=active)


@libentry()
@triton.jit
def dtrsv_update_kernel(
    a_ptr,
    x_ptr,
    row_base,
    row_count,
    start,
    end,
    LDA,
    INCX,
    TRANS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = row_base + pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = start + tl.arange(0, BLOCK_N)
    row_mask = rows < row_base + row_count
    col_mask = cols < end
    if TRANS == 0:
        a_off = rows[:, None] + cols[None, :] * LDA
    else:
        a_off = cols[None, :] + rows[:, None] * LDA
    mask = row_mask[:, None] & col_mask[None, :]
    a_vals = tl.load(a_ptr + a_off, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)
    acc = tl.sum(a_vals * x_vals[None, :], axis=1)
    rhs = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
    tl.store(x_ptr + rows * INCX, rhs - acc, mask=row_mask)


@triton.jit
def dtrsv_diag_inv_kernel(
    a_ptr,
    dinv_ptr,
    n,
    LDA,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BB: tl.constexpr,
):
    blk = tl.program_id(0)
    offs = tl.arange(0, BB)
    s = blk * BB
    grow = s + offs
    rmask = grow < n
    if TRANS == 0:
        m_off = (s + offs)[:, None] + (s + offs)[None, :] * LDA
    else:
        m_off = (s + offs)[None, :] + (s + offs)[:, None] * LDA
    mmask = rmask[:, None] & rmask[None, :]
    M = tl.load(a_ptr + m_off, mask=mmask, other=0.0)
    one = tl.full((BB,), 1.0, tl.float64)
    zero = tl.full((BB,), 0.0, tl.float64)
    if not UNIT:
        diag_each = tl.sum(tl.where(offs[:, None] == offs[None, :], M, 0.0), axis=1)
        safe_diag = tl.where(rmask, diag_each, one)
    X = tl.where(
        offs[:, None] == offs[None, :],
        tl.full((BB, BB), 1.0, tl.float64),
        tl.full((BB, BB), 0.0, tl.float64),
    )
    if LOWER_EFF:
        for i in tl.range(0, BB):
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs < i, mrow, zero)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, one, zero)
            if UNIT:
                xi = ei - contrib
            else:
                di = tl.sum(tl.where(offs == i, safe_diag, zero), axis=0)
                inv_di = one / di
                xi = (ei - contrib) * inv_di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    else:
        for ii in tl.range(0, BB):
            i = BB - 1 - ii
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs > i, mrow, zero)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, one, zero)
            if UNIT:
                xi = ei - contrib
            else:
                di = tl.sum(tl.where(offs == i, safe_diag, zero), axis=0)
                inv_di = one / di
                xi = (ei - contrib) * inv_di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    d_off = blk * BB * BB + offs[:, None] * BB + offs[None, :]
    tl.store(dinv_ptr + d_off, X, mask=mmask)


@triton.jit
def _dtrsv_diag_block_inv(
    a_ptr,
    s,
    LDA,
    offs,
    row_mask,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWLOAD: tl.constexpr,
):
    if UNIT and LOWER_EFF and TRANS == 0 and ROWLOAD:
        one = tl.full((BLOCK_N,), 1.0, tl.float64)
        zero = tl.full((BLOCK_N,), 0.0, tl.float64)
        X = tl.where(
            offs[:, None] == offs[None, :],
            tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float64),
            tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float64),
        )
        for i in tl.static_range(0, BLOCK_N):
            mrow = tl.load(
                a_ptr + (s + i) + (s + offs) * LDA,
                mask=(offs < i) & row_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            contrib = tl.sum(mrow[:, None] * X, axis=0)
            ei = tl.where(offs == i, one, zero)
            xi = ei - contrib
            X = tl.where(offs[:, None] == i, xi[None, :], X)
        return X
    if TRANS == 0:
        m_off = (s + offs)[:, None] + (s + offs)[None, :] * LDA
    else:
        m_off = (s + offs)[None, :] + (s + offs)[:, None] * LDA
    mmask = row_mask[:, None] & row_mask[None, :]
    M = tl.load(a_ptr + m_off, mask=mmask, other=0.0)
    one = tl.full((BLOCK_N,), 1.0, tl.float64)
    zero = tl.full((BLOCK_N,), 0.0, tl.float64)
    if not UNIT:
        diag_each = tl.sum(tl.where(offs[:, None] == offs[None, :], M, 0.0), axis=1)
        safe_diag = tl.where(row_mask, diag_each, one)
    X = tl.where(
        offs[:, None] == offs[None, :],
        tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float64),
        tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float64),
    )
    if LOWER_EFF:
        for i in tl.range(0, BLOCK_N):
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs < i, mrow, zero)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, one, zero)
            if UNIT:
                xi = ei - contrib
            else:
                di = tl.sum(tl.where(offs == i, safe_diag, zero), axis=0)
                inv_di = one / di
                xi = (ei - contrib) * inv_di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    else:
        for ii in tl.range(0, BLOCK_N):
            i = BLOCK_N - 1 - ii
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs > i, mrow, zero)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, one, zero)
            if UNIT:
                xi = ei - contrib
            else:
                di = tl.sum(tl.where(offs == i, safe_diag, zero), axis=0)
                inv_di = one / di
                xi = (ei - contrib) * inv_di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    return X


@triton.jit
def _dtrsv_diag_block_inv_32_nomask(
    a_ptr, s, offs, LOWER_EFF: tl.constexpr, BLOCK_N: tl.constexpr
):
    m_off = (s + offs)[:, None] + (s + offs)[None, :] * 256
    M = tl.load(a_ptr + m_off)
    one = tl.full((BLOCK_N,), 1.0, tl.float64)
    zero = tl.full((BLOCK_N,), 0.0, tl.float64)
    diag_each = tl.sum(tl.where(offs[:, None] == offs[None, :], M, 0.0), axis=1)
    X = tl.where(
        offs[:, None] == offs[None, :],
        tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float64),
        tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float64),
    )
    if LOWER_EFF:
        for i in tl.range(0, BLOCK_N):
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs < i, mrow, zero)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, one, zero)
            di = tl.sum(tl.where(offs == i, diag_each, zero), axis=0)
            inv_di = one / di
            xi = (ei - contrib) * inv_di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    else:
        for ii in tl.range(0, BLOCK_N):
            i = BLOCK_N - 1 - ii
            mrow = tl.sum(tl.where(offs[:, None] == i, M, 0.0), axis=0)
            mtri = tl.where(offs > i, mrow, zero)
            contrib = tl.sum(mtri[:, None] * X, axis=0)
            ei = tl.where(offs == i, one, zero)
            di = tl.sum(tl.where(offs == i, diag_each, zero), axis=0)
            inv_di = one / di
            xi = (ei - contrib) * inv_di
            X = tl.where(offs[:, None] == i, xi[None, :], X)
    return X


@triton.jit
def dtrsv_n256_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels: tl.constexpr = 8
    if LOWER_EFF:
        blk = pid
    else:
        blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    rows = row_start + offs
    dinv = _dtrsv_diag_block_inv_32_nomask(a_ptr, row_start, offs, LOWER_EFF, BLOCK_N)
    sx = tl.load(x_ptr + rows, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        if LOWER_EFF:
            c0 = pid_lo * BLOCK_N
            c1 = pid_hi * BLOCK_N
        else:
            c0 = (num_panels - pid_hi) * BLOCK_N
            c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = kcols < c1
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        a_vals = tl.load(
            a_ptr + rows[:, None] + kcols[None, :] * 256,
            mask=col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("dtrsv_fwd_inv"),
    key=_TRSV_KEY,
    restore_value=_TRSV_RESTORE,
)
@triton.jit
def dtrsv_fwd_inv_kernel(
    a_ptr,
    x_ptr,
    dinv_ptr,
    flag_ptr,
    n,
    LDA,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size
    sx = tl.load(x_ptr + rows, mask=row_mask, other=0.0, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = pid_lo * BLOCK_N
        c1 = pid_hi * BLOCK_N
        kcols = c0 + koffs
        col_mask = kcols < c1
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        a_vals = tl.load(
            a_ptr + a_off,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    d_off = pid * BLOCK_N * BLOCK_N + offs[:, None] * BLOCK_N + offs[None, :]
    dmask = row_mask[:, None] & row_mask[None, :]
    dinv = tl.load(dinv_ptr + d_off, mask=dmask, other=0.0)
    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("dtrsv_bwd_inv"),
    key=_TRSV_KEY,
    restore_value=_TRSV_RESTORE,
)
@triton.jit
def dtrsv_bwd_inv_kernel(
    a_ptr,
    x_ptr,
    dinv_ptr,
    flag_ptr,
    n,
    LDA,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels = tl.cdiv(n, BLOCK_N)
    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size
    sx = tl.load(x_ptr + rows, mask=row_mask, other=0.0, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        a_vals = tl.load(
            a_ptr + a_off,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    d_off = blk * BLOCK_N * BLOCK_N + offs[:, None] * BLOCK_N + offs[None, :]
    dmask = row_mask[:, None] & row_mask[None, :]
    dinv = tl.load(dinv_ptr + d_off, mask=dmask, other=0.0)
    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@triton.jit
def dtrsv_fwd_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWLOAD: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size
    dinv = _dtrsv_diag_block_inv(
        a_ptr, row_start, LDA, offs, row_mask, TRANS, UNIT, LOWER_EFF, BLOCK_N, ROWLOAD
    )
    sx = tl.load(x_ptr + rows, mask=row_mask, other=0.0, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = pid_lo * BLOCK_N
        c1 = pid_hi * BLOCK_N
        kcols = c0 + koffs
        col_mask = kcols < c1
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        a_vals = tl.load(
            a_ptr + a_off,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@triton.jit
def dtrsv_bwd_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWLOAD: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels = tl.cdiv(n, BLOCK_N)
    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_N), BLOCK_N)
    row_mask = offs < size
    dinv = _dtrsv_diag_block_inv(
        a_ptr, row_start, LDA, offs, row_mask, TRANS, UNIT, LOWER_EFF, BLOCK_N, ROWLOAD
    )
    sx = tl.load(x_ptr + rows, mask=row_mask, other=0.0, eviction_policy="evict_last")

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        xprev = tl.load(
            x_ptr + kcols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        )
        if TRANS == 0:
            a_off = rows[:, None] + kcols[None, :] * LDA
        else:
            a_off = kcols[None, :] + rows[:, None] * LDA
        a_vals = tl.load(
            a_ptr + a_off,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        sx -= tl.sum(a_vals * xprev[None, :], axis=1)

    sx = tl.sum(dinv * sx[None, :], axis=1)
    tl.store(x_ptr + rows, sx, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@triton.jit
def ctrsv_panel_kernel(
    a_ptr,
    x_ptr,
    start,
    end,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    FORWARD: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    size = end - start
    incx2 = INCX * 2
    for step in tl.static_range(0, BLOCK_N):
        if FORWARD:
            i = start + step
            active = step < size
            j = start + offs
            mask = (offs < step) & (offs < size)
        else:
            i = end - 1 - step
            active = step < size
            j = end - 1 - offs
            mask = (offs < step) & (offs < size)
        mask = mask & active
        if TRANS == 0:
            a_base = (i + j * LDA) * 2
        else:
            a_base = (j + i * LDA) * 2
        ar = tl.load(
            a_ptr + a_base, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        ai = tl.load(
            a_ptr + a_base + 1, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        xr = tl.load(
            x_ptr + j * incx2, mask=mask, other=0.0, eviction_policy="evict_last"
        )
        xi = tl.load(
            x_ptr + j * incx2 + 1, mask=mask, other=0.0, eviction_policy="evict_last"
        )
        if CONJ:
            ai = -ai
        value_r = tl.load(x_ptr + i * incx2, mask=active, other=0.0) - tl.sum(
            ar * xr - ai * xi, axis=0
        )
        value_i = tl.load(x_ptr + i * incx2 + 1, mask=active, other=0.0) - tl.sum(
            ar * xi + ai * xr, axis=0
        )
        if not UNIT:
            diag_r = tl.load(a_ptr + (i + i * LDA) * 2, mask=active, other=1.0)
            diag_i = tl.load(a_ptr + (i + i * LDA) * 2 + 1, mask=active, other=0.0)
            if CONJ:
                diag_i = -diag_i
            denom = diag_r * diag_r + diag_i * diag_i
            out_r = (value_r * diag_r + value_i * diag_i) / denom
            out_i = (value_i * diag_r - value_r * diag_i) / denom
            value_r = out_r
            value_i = out_i
        tl.store(x_ptr + i * incx2, value_r, mask=active)
        tl.store(x_ptr + i * incx2 + 1, value_i, mask=active)


@libentry()
@triton.jit
def ztrsv_panel_kernel(
    a_ptr,
    x_ptr,
    start,
    end,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    FORWARD: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    size = end - start
    incx2 = INCX * 2
    for step in tl.static_range(0, BLOCK_N):
        if FORWARD:
            i = start + step
            active = step < size
            j = start + offs
            mask = (offs < step) & (offs < size)
        else:
            i = end - 1 - step
            active = step < size
            j = end - 1 - offs
            mask = (offs < step) & (offs < size)
        mask = mask & active
        if TRANS == 0:
            a_base = (i + j * LDA) * 2
        else:
            a_base = (j + i * LDA) * 2
        ar = tl.load(a_ptr + a_base, mask=mask, other=0.0)
        ai = tl.load(a_ptr + a_base + 1, mask=mask, other=0.0)
        xr = tl.load(x_ptr + j * incx2, mask=mask, other=0.0)
        xi = tl.load(x_ptr + j * incx2 + 1, mask=mask, other=0.0)
        if CONJ:
            ai = -ai
        value_r = tl.load(x_ptr + i * incx2, mask=active, other=0.0) - tl.sum(
            ar * xr - ai * xi, axis=0
        )
        value_i = tl.load(x_ptr + i * incx2 + 1, mask=active, other=0.0) - tl.sum(
            ar * xi + ai * xr, axis=0
        )
        if not UNIT:
            diag_r = tl.load(a_ptr + (i + i * LDA) * 2, mask=active, other=1.0)
            diag_i = tl.load(a_ptr + (i + i * LDA) * 2 + 1, mask=active, other=0.0)
            if CONJ:
                diag_i = -diag_i
            denom = diag_r * diag_r + diag_i * diag_i
            out_r = (value_r * diag_r + value_i * diag_i) / denom
            out_i = (value_i * diag_r - value_r * diag_i) / denom
            value_r = out_r
            value_i = out_i
        tl.store(x_ptr + i * incx2, value_r, mask=active)
        tl.store(x_ptr + i * incx2 + 1, value_i, mask=active)


@libentry()
@triton.jit
def ctrsv_update_kernel(
    a_ptr,
    x_ptr,
    row_base,
    row_count,
    start,
    end,
    LDA,
    INCX,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = row_base + pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = start + tl.arange(0, BLOCK_N)
    row_mask = rows < row_base + row_count
    col_mask = cols < end
    if TRANS == 0:
        a_base = (rows[:, None] + cols[None, :] * LDA) * 2
    else:
        a_base = (cols[None, :] + rows[:, None] * LDA) * 2
    mask = row_mask[:, None] & col_mask[None, :]
    ar = tl.load(a_ptr + a_base, mask=mask, other=0.0, eviction_policy="evict_first")
    ai = tl.load(
        a_ptr + a_base + 1, mask=mask, other=0.0, eviction_policy="evict_first"
    )
    xr = tl.load(
        x_ptr + cols * INCX * 2, mask=col_mask, other=0.0, eviction_policy="evict_last"
    )
    xi = tl.load(
        x_ptr + cols * INCX * 2 + 1,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    )
    if CONJ:
        ai = -ai
    acc_r = tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
    acc_i = tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)
    rhs_r = tl.load(x_ptr + rows * INCX * 2, mask=row_mask, other=0.0)
    rhs_i = tl.load(x_ptr + rows * INCX * 2 + 1, mask=row_mask, other=0.0)
    tl.store(x_ptr + rows * INCX * 2, rhs_r - acc_r, mask=row_mask)
    tl.store(x_ptr + rows * INCX * 2 + 1, rhs_i - acc_i, mask=row_mask)


@libentry()
@triton.jit
def ztrsv_update_kernel(
    a_ptr,
    x_ptr,
    row_base,
    row_count,
    start,
    end,
    LDA,
    INCX,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = row_base + pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = start + tl.arange(0, BLOCK_N)
    row_mask = rows < row_base + row_count
    col_mask = cols < end
    if TRANS == 0:
        a_base = (rows[:, None] + cols[None, :] * LDA) * 2
    else:
        a_base = (cols[None, :] + rows[:, None] * LDA) * 2
    mask = row_mask[:, None] & col_mask[None, :]
    ar = tl.load(a_ptr + a_base, mask=mask, other=0.0)
    ai = tl.load(a_ptr + a_base + 1, mask=mask, other=0.0)
    xr = tl.load(x_ptr + cols * INCX * 2, mask=col_mask, other=0.0)
    xi = tl.load(x_ptr + cols * INCX * 2 + 1, mask=col_mask, other=0.0)
    if CONJ:
        ai = -ai
    acc_r = tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
    acc_i = tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)
    rhs_r = tl.load(x_ptr + rows * INCX * 2, mask=row_mask, other=0.0)
    rhs_i = tl.load(x_ptr + rows * INCX * 2 + 1, mask=row_mask, other=0.0)
    tl.store(x_ptr + rows * INCX * 2, rhs_r - acc_r, mask=row_mask)
    tl.store(x_ptr + rows * INCX * 2 + 1, rhs_i - acc_i, mask=row_mask)


def _check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok):
    assert A.is_contiguous() and x.is_contiguous()
    assert A.device == x.device
    assert uplo in (CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER)
    allowed = (
        [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
        if complex_ok
        else [CUBLAS_OP_N, CUBLAS_OP_T]
    )
    assert trans in allowed
    assert diag in (CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT)
    assert incx > 0
    assert n >= 0
    assert lda >= max(1, n)
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert A.numel() >= n * lda


def _mode_key(uplo, trans, unit):
    return (uplo << 4) | (trans << 2) | unit


def _forward(uplo, trans):
    if trans == CUBLAS_OP_N:
        return 1 if uplo == CUBLAS_FILL_MODE_LOWER else 0
    return 1 if uplo == CUBLAS_FILL_MODE_UPPER else 0


def _panel_ranges(n, block_n, forward):
    if forward:
        start = 0
        while start < n:
            end = min(start + block_n, n)
            yield start, end
            start = end
    else:
        end = n
        while end > 0:
            start = max(0, end - block_n)
            yield start, end
            end = start


def strsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == torch.float32 == x.dtype
    _check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=False)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    forward = _forward(uplo, trans)
    mode_key = _mode_key(uplo, trans_flag, unit)

    with torch_device_fn.device(A.device):
        if trans_flag == 0 and incx == 1 and lda == n and n <= 64:
            strsv_n64_kernel[(1,)](
                A,
                x,
                n,
                LOWER=1 if uplo == CUBLAS_FILL_MODE_LOWER else 0,
                UNIT=unit,
                BLOCK_N=64,
                num_warps=1,
            )
            return
        if (
            trans_flag == 0
            and incx == 1
            and lda == n
            and n == 256
            and uplo == CUBLAS_FILL_MODE_LOWER
            and unit == 1
        ):
            flags = torch.full((1,), -1, dtype=torch.int32, device=A.device)
            strsv_lu256_rowinv_kernel[(8,)](
                A,
                x,
                flags,
                BLOCK_N=32,
                CHUNK=2,
                num_warps=4,
            )
            return

        if (
            trans_flag == 0
            and incx == 1
            and lda == n
            and n == 256
            and uplo == CUBLAS_FILL_MODE_UPPER
            and unit == 0
        ):
            flags = torch.full((1,), -1, dtype=torch.int32, device=A.device)
            strsv_un256_rowinv_kernel[(8,)](
                A,
                x,
                flags,
                BLOCK_N=32,
                CHUNK=2,
                num_warps=4,
            )
            return

        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_N"]),)

        flags = torch.full((1,), -1, dtype=torch.int32, device=A.device)
        if n >= _STRSV_INV_MIN_N:
            bb = 64 if n >= _STRSV_INV_BB64_MIN_N else 32
            npanel = triton.cdiv(n, bb)
            lower_eff = 1 if (uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans_flag == 1) else 0
            inv_grid = (npanel,)
            if bb == 32 and n <= _STRSV_FUSE_MAX_N and trans_flag == 0:
                if forward:
                    strsv_fwd_fused_kernel[inv_grid](
                        A,
                        x,
                        flags,
                        n,
                        lda,
                        incx,
                        mode_key,
                        TRANS=trans_flag,
                        UNIT=unit,
                        LOWER_EFF=lower_eff,
                        BLOCK_N=bb,
                    )
                else:
                    strsv_bwd_fused_kernel[inv_grid](
                        A,
                        x,
                        flags,
                        n,
                        lda,
                        incx,
                        mode_key,
                        TRANS=trans_flag,
                        UNIT=unit,
                        LOWER_EFF=lower_eff,
                        BLOCK_N=bb,
                    )
            else:
                dinv = torch.empty(
                    (npanel, bb, bb), dtype=torch.float32, device=A.device
                )
                strsv_diag_inv_kernel[(npanel,)](
                    A,
                    dinv,
                    n,
                    lda,
                    TRANS=trans_flag,
                    UNIT=unit,
                    LOWER_EFF=lower_eff,
                    BB=bb,
                    num_warps=1 if bb == 32 else 2,
                )
                if forward:
                    strsv_fwd_inv_kernel[inv_grid](
                        A,
                        x,
                        dinv,
                        flags,
                        n,
                        lda,
                        incx,
                        mode_key,
                        TRANS=trans_flag,
                        UNIT=unit,
                        BLOCK_N=bb,
                    )
                else:
                    strsv_bwd_inv_kernel[inv_grid](
                        A,
                        x,
                        dinv,
                        flags,
                        n,
                        lda,
                        incx,
                        mode_key,
                        TRANS=trans_flag,
                        UNIT=unit,
                        BLOCK_N=bb,
                    )
        elif forward:
            strsv_fwd_kernel[grid](
                A,
                x,
                flags,
                n,
                lda,
                incx,
                mode_key,
                TRANS=trans_flag,
                UNIT=unit,
            )
        else:
            strsv_bwd_kernel[grid](
                A,
                x,
                flags,
                n,
                lda,
                incx,
                mode_key,
                TRANS=trans_flag,
                UNIT=unit,
            )


def dtrsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == torch.float64 == x.dtype
    _check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=False)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    forward = _forward(uplo, trans)
    mode_key = _mode_key(uplo, trans_flag, unit)

    with torch_device_fn.device(A.device):
        if incx == 1 and lda == n:
            if (
                trans_flag == 0
                and uplo == CUBLAS_FILL_MODE_LOWER
                and unit == 1
                and n <= 64
            ):
                dtrsv_n64_kernel[(1,)](
                    A,
                    x,
                    n,
                    LOWER=1,
                    UNIT=1,
                    BLOCK_N=64,
                    num_warps=1,
                )
                return
            if trans_flag == 0 and n == 256 and unit == 0:
                flags = torch.full((1,), -1, dtype=torch.int32, device=A.device)
                dtrsv_n256_fused_kernel[(8,)](
                    A,
                    x,
                    flags,
                    LOWER_EFF=1 if uplo == CUBLAS_FILL_MODE_LOWER else 0,
                    BLOCK_N=32,
                    CHUNK=4,
                    num_warps=4,
                )
                return
            if n <= 64:
                bb = 16
            else:
                bb = 64 if n >= _DTRSV_INV_BB64_MIN_N else 32
            npanel = triton.cdiv(n, bb)
            flags = torch.full((1,), -1, dtype=torch.int32, device=A.device)
            lower_eff = 1 if (uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans_flag == 1) else 0
            inv_grid = (npanel,)
            rowload = 1 if n <= _DTRSV_ROWLOAD_MAX_N else 0
            if (bb == 16 or bb == 32) and n <= _DTRSV_FUSE_MAX_N:
                chunk, nw = _dtrsv_fused_cfg(forward, n, bb, unit)
                if forward:
                    dtrsv_fwd_fused_kernel[inv_grid](
                        A,
                        x,
                        flags,
                        n,
                        lda,
                        mode_key,
                        TRANS=trans_flag,
                        UNIT=unit,
                        LOWER_EFF=lower_eff,
                        BLOCK_N=bb,
                        CHUNK=chunk,
                        ROWLOAD=rowload,
                        num_warps=nw,
                    )
                else:
                    dtrsv_bwd_fused_kernel[inv_grid](
                        A,
                        x,
                        flags,
                        n,
                        lda,
                        mode_key,
                        TRANS=trans_flag,
                        UNIT=unit,
                        LOWER_EFF=lower_eff,
                        BLOCK_N=bb,
                        CHUNK=chunk,
                        ROWLOAD=rowload,
                        num_warps=nw,
                    )
            else:
                dinv = torch.empty(
                    (npanel, bb, bb), dtype=torch.float64, device=A.device
                )
                dtrsv_diag_inv_kernel[(npanel,)](
                    A,
                    dinv,
                    n,
                    lda,
                    TRANS=trans_flag,
                    UNIT=unit,
                    LOWER_EFF=lower_eff,
                    BB=bb,
                    num_warps=1 if bb == 32 else 2,
                )
                if forward:
                    dtrsv_fwd_inv_kernel[inv_grid](
                        A,
                        x,
                        dinv,
                        flags,
                        n,
                        lda,
                        mode_key,
                        TRANS=trans_flag,
                        UNIT=unit,
                        BLOCK_N=bb,
                    )
                else:
                    dtrsv_bwd_inv_kernel[inv_grid](
                        A,
                        x,
                        dinv,
                        flags,
                        n,
                        lda,
                        mode_key,
                        TRANS=trans_flag,
                        UNIT=unit,
                        BLOCK_N=bb,
                    )
            return

        block_n = 16
        block_m = 128
        for start, end in _panel_ranges(n, block_n, forward):
            dtrsv_panel_kernel[(1,)](
                A,
                x,
                start,
                end,
                lda,
                incx,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                FORWARD=forward,
                BLOCK_N=block_n,
            )
            if forward:
                row_base = end
                row_count = n - end
            else:
                row_base = 0
                row_count = start
            if row_count > 0:
                grid = (triton.cdiv(row_count, block_m),)
                dtrsv_update_kernel[grid](
                    A,
                    x,
                    row_base,
                    row_count,
                    start,
                    end,
                    lda,
                    incx,
                    TRANS=trans_flag,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                )


@triton.jit
def _ctrsv_diag_block_inv(
    a_ptr,
    s,
    LDA,
    offs,
    row_mask,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWLOAD: tl.constexpr,
):
    if UNIT and LOWER_EFF and TRANS == 0 and ROWLOAD:
        one = tl.full((BLOCK_N,), 1.0, tl.float32)
        zero = tl.full((BLOCK_N,), 0.0, tl.float32)
        Xr = tl.where(
            offs[:, None] == offs[None, :],
            tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float32),
            tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float32),
        )
        Xi = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
        for i in tl.static_range(0, BLOCK_N):
            roff = ((s + i) + (s + offs) * LDA) * 2
            mrr = tl.load(
                a_ptr + roff,
                mask=(offs < i) & row_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            mri = tl.load(
                a_ptr + roff + 1,
                mask=(offs < i) & row_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if CONJ:
                mri = -mri
            cr = tl.sum(mrr[:, None] * Xr - mri[:, None] * Xi, axis=0)
            ci = tl.sum(mrr[:, None] * Xi + mri[:, None] * Xr, axis=0)
            ei = tl.where(offs == i, one, zero)
            Xr = tl.where(offs[:, None] == i, (ei - cr)[None, :], Xr)
            Xi = tl.where(offs[:, None] == i, (-ci)[None, :], Xi)
        return Xr, Xi
    if TRANS == 0:
        m_off = ((s + offs)[:, None] + (s + offs)[None, :] * LDA) * 2
    else:
        m_off = ((s + offs)[None, :] + (s + offs)[:, None] * LDA) * 2
    mmask = row_mask[:, None] & row_mask[None, :]
    Mr = tl.load(a_ptr + m_off, mask=mmask, other=0.0)
    Mi = tl.load(a_ptr + m_off + 1, mask=mmask, other=0.0)
    if CONJ:
        Mi = -Mi
    one = tl.full((BLOCK_N,), 1.0, tl.float32)
    zero = tl.full((BLOCK_N,), 0.0, tl.float32)
    if not UNIT:
        dr = tl.sum(tl.where(offs[:, None] == offs[None, :], Mr, 0.0), axis=1)
        di = tl.sum(tl.where(offs[:, None] == offs[None, :], Mi, 0.0), axis=1)
        dr = tl.where(row_mask, dr, one)
        di = tl.where(row_mask, di, zero)
        denom = dr * dr + di * di
    Xr = tl.where(
        offs[:, None] == offs[None, :],
        tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float32),
        tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float32),
    )
    Xi = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    if LOWER_EFF:
        for i in tl.range(0, BLOCK_N):
            mrr = tl.sum(tl.where(offs[:, None] == i, Mr, 0.0), axis=0)
            mri = tl.sum(tl.where(offs[:, None] == i, Mi, 0.0), axis=0)
            tri = offs < i
            mrr = tl.where(tri, mrr, zero)
            mri = tl.where(tri, mri, zero)
            cr = tl.sum(mrr[:, None] * Xr - mri[:, None] * Xi, axis=0)
            ci = tl.sum(mrr[:, None] * Xi + mri[:, None] * Xr, axis=0)
            ei = tl.where(offs == i, one, zero)
            ar = ei - cr
            ai = -ci
            if UNIT:
                xir = ar
                xii = ai
            else:
                dir_ = tl.sum(tl.where(offs == i, dr, zero), axis=0)
                dii_ = tl.sum(tl.where(offs == i, di, zero), axis=0)
                den = tl.sum(tl.where(offs == i, denom, zero), axis=0)
                xir = (ar * dir_ + ai * dii_) / den
                xii = (ai * dir_ - ar * dii_) / den
            Xr = tl.where(offs[:, None] == i, xir[None, :], Xr)
            Xi = tl.where(offs[:, None] == i, xii[None, :], Xi)
    else:
        for ii in tl.range(0, BLOCK_N):
            i = BLOCK_N - 1 - ii
            mrr = tl.sum(tl.where(offs[:, None] == i, Mr, 0.0), axis=0)
            mri = tl.sum(tl.where(offs[:, None] == i, Mi, 0.0), axis=0)
            tri = offs > i
            mrr = tl.where(tri, mrr, zero)
            mri = tl.where(tri, mri, zero)
            cr = tl.sum(mrr[:, None] * Xr - mri[:, None] * Xi, axis=0)
            ci = tl.sum(mrr[:, None] * Xi + mri[:, None] * Xr, axis=0)
            ei = tl.where(offs == i, one, zero)
            ar = ei - cr
            ai = -ci
            if UNIT:
                xir = ar
                xii = ai
            else:
                dir_ = tl.sum(tl.where(offs == i, dr, zero), axis=0)
                dii_ = tl.sum(tl.where(offs == i, di, zero), axis=0)
                den = tl.sum(tl.where(offs == i, denom, zero), axis=0)
                xir = (ar * dir_ + ai * dii_) / den
                xii = (ai * dir_ - ar * dii_) / den
            Xr = tl.where(offs[:, None] == i, xir[None, :], Xr)
            Xi = tl.where(offs[:, None] == i, xii[None, :], Xi)
    return Xr, Xi


@libentry()
@triton.jit
def ctrsv_fwd_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWLOAD: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    row_mask = offs < size
    Xr, Xi = _ctrsv_diag_block_inv(
        a_ptr,
        row_start,
        LDA,
        offs,
        row_mask,
        TRANS,
        UNIT,
        CONJ,
        LOWER_EFF,
        BLOCK_N,
        ROWLOAD,
    )
    sxr = tl.load(
        x_ptr + rows * 2, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )
    sxi = tl.load(
        x_ptr + rows * 2 + 1, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = pid_lo * BLOCK_N
        c1 = pid_hi * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        x2 = tl.load(
            x_ptr + (kcols * 2)[:, None] + tl.arange(0, 2)[None, :],
            mask=col_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        xpr, xpi = tl.split(x2)
        if TRANS == 0:
            a_off = (rows[:, None] + kcols[None, :] * LDA) * 2
        else:
            a_off = (kcols[None, :] + rows[:, None] * LDA) * 2
        m2 = row_mask[:, None] & col_mask[None, :]
        a2 = tl.load(
            a_ptr + a_off[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask=m2[:, :, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        ar, ai = tl.split(a2)
        if CONJ:
            ai = -ai
        sxr -= tl.sum(ar * xpr[None, :] - ai * xpi[None, :], axis=1)
        sxi -= tl.sum(ar * xpi[None, :] + ai * xpr[None, :], axis=1)

    outr = tl.sum(Xr * sxr[None, :] - Xi * sxi[None, :], axis=1)
    outi = tl.sum(Xr * sxi[None, :] + Xi * sxr[None, :], axis=1)
    tl.store(x_ptr + rows * 2, outr, mask=row_mask)
    tl.store(x_ptr + rows * 2 + 1, outi, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@triton.jit
def ctrsv_bwd_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWLOAD: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels = tl.cdiv(n, BLOCK_N)
    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    row_mask = offs < size
    Xr, Xi = _ctrsv_diag_block_inv(
        a_ptr,
        row_start,
        LDA,
        offs,
        row_mask,
        TRANS,
        UNIT,
        CONJ,
        LOWER_EFF,
        BLOCK_N,
        ROWLOAD,
    )
    sxr = tl.load(
        x_ptr + rows * 2, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )
    sxi = tl.load(
        x_ptr + rows * 2 + 1, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        x2 = tl.load(
            x_ptr + (kcols * 2)[:, None] + tl.arange(0, 2)[None, :],
            mask=col_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        xpr, xpi = tl.split(x2)
        if TRANS == 0:
            a_off = (rows[:, None] + kcols[None, :] * LDA) * 2
        else:
            a_off = (kcols[None, :] + rows[:, None] * LDA) * 2
        m2 = row_mask[:, None] & col_mask[None, :]
        a2 = tl.load(
            a_ptr + a_off[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask=m2[:, :, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        ar, ai = tl.split(a2)
        if CONJ:
            ai = -ai
        sxr -= tl.sum(ar * xpr[None, :] - ai * xpi[None, :], axis=1)
        sxi -= tl.sum(ar * xpi[None, :] + ai * xpr[None, :], axis=1)

    outr = tl.sum(Xr * sxr[None, :] - Xi * sxi[None, :], axis=1)
    outi = tl.sum(Xr * sxi[None, :] + Xi * sxr[None, :], axis=1)
    tl.store(x_ptr + rows * 2, outr, mask=row_mask)
    tl.store(x_ptr + rows * 2 + 1, outi, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


def ctrsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == torch.complex64 == x.dtype
    _check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=True)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    conj = 1 if trans == CUBLAS_OP_C else 0
    forward = _forward(uplo, trans)
    mode_key = _mode_key(uplo, trans_flag, unit) | (conj << 8)

    with torch_device_fn.device(A.device):
        A_real = torch.view_as_real(A)
        x_real = torch.view_as_real(x)
        if incx == 1 and lda == n:
            bb = 16 if n <= 64 else 32
            npanel = triton.cdiv(n, bb)
            flags = torch.full((1,), -1, dtype=torch.int32, device=A.device)
            lower_eff = 1 if (uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans_flag == 1) else 0
            rowload = 1 if n <= _DTRSV_ROWLOAD_MAX_N else 0
            chunk, nw = _ctrsv_fused_cfg(forward, n, bb, unit)
            if forward:
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
                    ROWLOAD=rowload,
                    num_warps=nw,
                )
            else:
                ctrsv_bwd_fused_kernel[(npanel,)](
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
                    ROWLOAD=rowload,
                    num_warps=nw,
                )
            return

        block_n = 16
        block_m = 128
        for start, end in _panel_ranges(n, block_n, forward):
            ctrsv_panel_kernel[(1,)](
                A_real,
                x_real,
                start,
                end,
                lda,
                incx,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                FORWARD=forward,
                CONJ=conj,
                BLOCK_N=block_n,
            )
            if forward:
                row_base = end
                row_count = n - end
            else:
                row_base = 0
                row_count = start
            if row_count > 0:
                grid = (triton.cdiv(row_count, block_m),)
                ctrsv_update_kernel[grid](
                    A_real,
                    x_real,
                    row_base,
                    row_count,
                    start,
                    end,
                    lda,
                    incx,
                    TRANS=trans_flag,
                    CONJ=conj,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                )


@triton.jit
def _ztrsv_diag_block_inv(
    a_ptr,
    s,
    LDA,
    offs,
    row_mask,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWLOAD: tl.constexpr,
):
    if UNIT and LOWER_EFF and TRANS == 0 and ROWLOAD:
        one = tl.full((BLOCK_N,), 1.0, tl.float64)
        zero = tl.full((BLOCK_N,), 0.0, tl.float64)
        Xr = tl.where(
            offs[:, None] == offs[None, :],
            tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float64),
            tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float64),
        )
        Xi = tl.zeros((BLOCK_N, BLOCK_N), tl.float64)
        for i in tl.static_range(0, BLOCK_N):
            roff = ((s + i) + (s + offs) * LDA) * 2
            mrr = tl.load(
                a_ptr + roff,
                mask=(offs < i) & row_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            mri = tl.load(
                a_ptr + roff + 1,
                mask=(offs < i) & row_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if CONJ:
                mri = -mri
            cr = tl.sum(mrr[:, None] * Xr - mri[:, None] * Xi, axis=0)
            ci = tl.sum(mrr[:, None] * Xi + mri[:, None] * Xr, axis=0)
            ei = tl.where(offs == i, one, zero)
            Xr = tl.where(offs[:, None] == i, (ei - cr)[None, :], Xr)
            Xi = tl.where(offs[:, None] == i, (-ci)[None, :], Xi)
        return Xr, Xi
    if TRANS == 0:
        m_off = ((s + offs)[:, None] + (s + offs)[None, :] * LDA) * 2
    else:
        m_off = ((s + offs)[None, :] + (s + offs)[:, None] * LDA) * 2
    mmask = row_mask[:, None] & row_mask[None, :]
    Mr = tl.load(a_ptr + m_off, mask=mmask, other=0.0)
    Mi = tl.load(a_ptr + m_off + 1, mask=mmask, other=0.0)
    if CONJ:
        Mi = -Mi
    one = tl.full((BLOCK_N,), 1.0, tl.float64)
    zero = tl.full((BLOCK_N,), 0.0, tl.float64)
    if not UNIT:
        dr = tl.sum(tl.where(offs[:, None] == offs[None, :], Mr, 0.0), axis=1)
        di = tl.sum(tl.where(offs[:, None] == offs[None, :], Mi, 0.0), axis=1)
        dr = tl.where(row_mask, dr, one)
        di = tl.where(row_mask, di, zero)
        denom = dr * dr + di * di
    Xr = tl.where(
        offs[:, None] == offs[None, :],
        tl.full((BLOCK_N, BLOCK_N), 1.0, tl.float64),
        tl.full((BLOCK_N, BLOCK_N), 0.0, tl.float64),
    )
    Xi = tl.zeros((BLOCK_N, BLOCK_N), tl.float64)
    if LOWER_EFF:
        for i in tl.range(0, BLOCK_N):
            mrr = tl.sum(tl.where(offs[:, None] == i, Mr, 0.0), axis=0)
            mri = tl.sum(tl.where(offs[:, None] == i, Mi, 0.0), axis=0)
            tri = offs < i
            mrr = tl.where(tri, mrr, zero)
            mri = tl.where(tri, mri, zero)
            cr = tl.sum(mrr[:, None] * Xr - mri[:, None] * Xi, axis=0)
            ci = tl.sum(mrr[:, None] * Xi + mri[:, None] * Xr, axis=0)
            ei = tl.where(offs == i, one, zero)
            ar = ei - cr
            ai = -ci
            if UNIT:
                xir = ar
                xii = ai
            else:
                dir_ = tl.sum(tl.where(offs == i, dr, zero), axis=0)
                dii_ = tl.sum(tl.where(offs == i, di, zero), axis=0)
                den = tl.sum(tl.where(offs == i, denom, zero), axis=0)
                xir = (ar * dir_ + ai * dii_) / den
                xii = (ai * dir_ - ar * dii_) / den
            Xr = tl.where(offs[:, None] == i, xir[None, :], Xr)
            Xi = tl.where(offs[:, None] == i, xii[None, :], Xi)
    else:
        for ii in tl.range(0, BLOCK_N):
            i = BLOCK_N - 1 - ii
            mrr = tl.sum(tl.where(offs[:, None] == i, Mr, 0.0), axis=0)
            mri = tl.sum(tl.where(offs[:, None] == i, Mi, 0.0), axis=0)
            tri = offs > i
            mrr = tl.where(tri, mrr, zero)
            mri = tl.where(tri, mri, zero)
            cr = tl.sum(mrr[:, None] * Xr - mri[:, None] * Xi, axis=0)
            ci = tl.sum(mrr[:, None] * Xi + mri[:, None] * Xr, axis=0)
            ei = tl.where(offs == i, one, zero)
            ar = ei - cr
            ai = -ci
            if UNIT:
                xir = ar
                xii = ai
            else:
                dir_ = tl.sum(tl.where(offs == i, dr, zero), axis=0)
                dii_ = tl.sum(tl.where(offs == i, di, zero), axis=0)
                den = tl.sum(tl.where(offs == i, denom, zero), axis=0)
                xir = (ar * dir_ + ai * dii_) / den
                xii = (ai * dir_ - ar * dii_) / den
            Xr = tl.where(offs[:, None] == i, xir[None, :], Xr)
            Xi = tl.where(offs[:, None] == i, xii[None, :], Xi)
    return Xr, Xi


@libentry()
@triton.jit
def ztrsv_fwd_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWLOAD: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    row_start = pid * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    row_mask = offs < size
    Xr, Xi = _ztrsv_diag_block_inv(
        a_ptr,
        row_start,
        LDA,
        offs,
        row_mask,
        TRANS,
        UNIT,
        CONJ,
        LOWER_EFF,
        BLOCK_N,
        ROWLOAD,
    )
    sxr = tl.load(
        x_ptr + rows * 2, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )
    sxi = tl.load(
        x_ptr + rows * 2 + 1, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = pid_lo * BLOCK_N
        c1 = pid_hi * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        x2 = tl.load(
            x_ptr + (kcols * 2)[:, None] + tl.arange(0, 2)[None, :],
            mask=col_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        xpr, xpi = tl.split(x2)
        if TRANS == 0:
            a_off = (rows[:, None] + kcols[None, :] * LDA) * 2
        else:
            a_off = (kcols[None, :] + rows[:, None] * LDA) * 2
        m2 = row_mask[:, None] & col_mask[None, :]
        a2 = tl.load(
            a_ptr + a_off[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask=m2[:, :, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        ar, ai = tl.split(a2)
        if CONJ:
            ai = -ai
        sxr -= tl.sum(ar * xpr[None, :] - ai * xpi[None, :], axis=1)
        sxi -= tl.sum(ar * xpi[None, :] + ai * xpr[None, :], axis=1)

    outr = tl.sum(Xr * sxr[None, :] - Xi * sxi[None, :], axis=1)
    outi = tl.sum(Xr * sxi[None, :] + Xi * sxr[None, :], axis=1)
    tl.store(x_ptr + rows * 2, outr, mask=row_mask)
    tl.store(x_ptr + rows * 2 + 1, outi, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


@libentry()
@triton.jit
def ztrsv_bwd_fused_kernel(
    a_ptr,
    x_ptr,
    flag_ptr,
    n,
    LDA,
    mode_key,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    LOWER_EFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWLOAD: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    BK: tl.constexpr = BLOCK_N * CHUNK
    koffs = tl.arange(0, BK)
    num_panels = tl.cdiv(n, BLOCK_N)
    blk = num_panels - 1 - pid
    row_start = blk * BLOCK_N
    row_end = tl.minimum(row_start + BLOCK_N, n)
    size = row_end - row_start
    rows = row_start + offs
    row_mask = offs < size
    Xr, Xi = _ztrsv_diag_block_inv(
        a_ptr,
        row_start,
        LDA,
        offs,
        row_mask,
        TRANS,
        UNIT,
        CONJ,
        LOWER_EFF,
        BLOCK_N,
        ROWLOAD,
    )
    sxr = tl.load(
        x_ptr + rows * 2, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )
    sxi = tl.load(
        x_ptr + rows * 2 + 1, mask=row_mask, other=0.0, eviction_policy="evict_last"
    )

    num_chunks = tl.cdiv(pid, CHUNK)
    for g in tl.range(0, num_chunks):
        pid_lo = g * CHUNK
        pid_hi = tl.minimum(pid_lo + CHUNK, pid)
        while tl.atomic_add(flag_ptr, 0, sem="acquire") < pid_hi - 1:
            pass
        c0 = (num_panels - pid_hi) * BLOCK_N
        c1 = (num_panels - pid_lo) * BLOCK_N
        kcols = c0 + koffs
        col_mask = (kcols < c1) & (kcols < n)
        x2 = tl.load(
            x_ptr + (kcols * 2)[:, None] + tl.arange(0, 2)[None, :],
            mask=col_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        xpr, xpi = tl.split(x2)
        if TRANS == 0:
            a_off = (rows[:, None] + kcols[None, :] * LDA) * 2
        else:
            a_off = (kcols[None, :] + rows[:, None] * LDA) * 2
        m2 = row_mask[:, None] & col_mask[None, :]
        a2 = tl.load(
            a_ptr + a_off[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask=m2[:, :, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        ar, ai = tl.split(a2)
        if CONJ:
            ai = -ai
        sxr -= tl.sum(ar * xpr[None, :] - ai * xpi[None, :], axis=1)
        sxi -= tl.sum(ar * xpi[None, :] + ai * xpr[None, :], axis=1)

    outr = tl.sum(Xr * sxr[None, :] - Xi * sxi[None, :], axis=1)
    outi = tl.sum(Xr * sxi[None, :] + Xi * sxr[None, :], axis=1)
    tl.store(x_ptr + rows * 2, outr, mask=row_mask)
    tl.store(x_ptr + rows * 2 + 1, outi, mask=row_mask)
    tl.atomic_xchg(flag_ptr, pid, sem="release")


def ztrsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == torch.complex128 == x.dtype
    _check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=True)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    conj = 1 if trans == CUBLAS_OP_C else 0
    forward = _forward(uplo, trans)
    mode_key = _mode_key(uplo, trans_flag, unit) | (conj << 8)

    with torch_device_fn.device(A.device):
        A_real = torch.view_as_real(A)
        x_real = torch.view_as_real(x)
        if incx == 1 and lda == n:
            if unit:
                bb = 16 if n <= 64 else 32
            else:
                bb = 16 if n <= 512 else 32
            npanel = triton.cdiv(n, bb)
            flags = torch.full((1,), -1, dtype=torch.int32, device=A.device)
            lower_eff = 1 if (uplo == CUBLAS_FILL_MODE_LOWER) ^ (trans_flag == 1) else 0
            rowload = 1 if n <= _ZTRSV_ROWLOAD_MAX_N else 0
            chunk, nw = _ztrsv_fused_cfg(forward, n, bb, unit)
            if forward:
                ztrsv_fwd_fused_kernel[(npanel,)](
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
                    ROWLOAD=rowload,
                    num_warps=nw,
                )
            else:
                ztrsv_bwd_fused_kernel[(npanel,)](
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
                    ROWLOAD=rowload,
                    num_warps=nw,
                )
            return

        block_n = 8
        block_m = 64
        A_real = torch.view_as_real(A)
        x_real = torch.view_as_real(x)
        for start, end in _panel_ranges(n, block_n, forward):
            ztrsv_panel_kernel[(1,)](
                A_real,
                x_real,
                start,
                end,
                lda,
                incx,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                FORWARD=forward,
                CONJ=conj,
                BLOCK_N=block_n,
            )
            if forward:
                row_base = end
                row_count = n - end
            else:
                row_base = 0
                row_count = start
            if row_count > 0:
                grid = (triton.cdiv(row_count, block_m),)
                ztrsv_update_kernel[grid](
                    A_real,
                    x_real,
                    row_base,
                    row_count,
                    start,
                    end,
                    lda,
                    incx,
                    TRANS=trans_flag,
                    CONJ=conj,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                )
