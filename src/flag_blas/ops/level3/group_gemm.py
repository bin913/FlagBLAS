import logging

import torch
import triton
import triton.language as tl

from flag_blas.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@triton.jit
def grouped_launch(
    pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr
):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)
    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


def get_autotune_config(pre_hook=None):
    return [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 4},
            num_stages=4,
            num_warps=4,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4},
            num_stages=4,
            num_warps=4,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 4},
            num_stages=3,
            num_warps=4,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 512, "BLOCK_K": 32, "GROUP_M": 4},
            num_stages=4,
            num_warps=8,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
            pre_hook=pre_hook,
        ),
    ]


def get_autotune_config_tf32(pre_hook=None):
    return [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4},
            num_stages=3,
            num_warps=4,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=2,
            num_warps=8,
            pre_hook=pre_hook,
        ),
    ]


@libentry()
@libtuner(configs=get_autotune_config(), key=["M", "N", "K"])
@triton.jit
def grouped_bfgemm_kernel(
    M,
    N,
    K,
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_out_ptrs,
    group_m_sizes,
    group_n_sizes,
    group_k_sizes,
    group_ldas,
    group_ldbs,
    group_ldcs,
    group_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    total_grid = tl.num_programs(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_m_sizes + g)
        gn = tl.load(group_n_sizes + g)
        gk = tl.load(group_k_sizes + g)
        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles

        current_problem_end = last_problem_end + num_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            lda = tl.load(group_ldas + g)
            ldb = tl.load(group_ldbs + g)
            ldc = tl.load(group_ldcs + g)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            out_ptr = tl.load(group_out_ptrs + g).to(tl.pointer_type(tl.bfloat16))

            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx, tile_n_idx = grouped_launch(
                    tile_idx_in_gemm, gm, gn, BLOCK_M, BLOCK_N, GROUP_M
                )

                offs_am = tile_m_idx * BLOCK_M
                offs_bn = tile_n_idx * BLOCK_N

                a_ptrs = tl.make_block_ptr(
                    base=a_ptr,
                    shape=(gm, gk),
                    strides=(lda, 1),
                    offsets=(offs_am, 0),
                    block_shape=(BLOCK_M, BLOCK_K),
                    order=(1, 0),
                )
                b_ptrs = tl.make_block_ptr(
                    base=b_ptr,
                    shape=(gk, gn),
                    strides=(ldb, 1),
                    offsets=(0, offs_bn),
                    block_shape=(BLOCK_K, BLOCK_N),
                    order=(1, 0),
                )

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(gk, BLOCK_K)):
                    a = tl.load(a_ptrs, boundary_check=(0, 1))
                    b = tl.load(b_ptrs, boundary_check=(0, 1))
                    accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)
                    a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
                    b_ptrs = tl.advance(b_ptrs, (BLOCK_K, 0))

                offs_cm = tile_m_idx * BLOCK_M
                offs_cn = tile_n_idx * BLOCK_N

                out_ptrs = tl.make_block_ptr(
                    base=out_ptr,
                    shape=(gm, gn),
                    strides=(ldc, 1),
                    offsets=(offs_cm, offs_cn),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                if beta == 0.0:
                    accumulator = accumulator * alpha
                else:
                    c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.bfloat16))
                    c_ptrs = tl.make_block_ptr(
                        base=c_ptr,
                        shape=(gm, gn),
                        strides=(ldc, 1),
                        offsets=(offs_cm, offs_cn),
                        block_shape=(BLOCK_M, BLOCK_N),
                        order=(1, 0),
                    )
                    ori_c = tl.load(c_ptrs, boundary_check=(0, 1))
                    accumulator = ori_c * beta + accumulator * alpha

                c = accumulator.to(out_ptrs.dtype.element_ty)
                tl.store(out_ptrs, c, boundary_check=(0, 1))

                tile_idx += total_grid

        last_problem_end = current_problem_end


def group_bfgemm(
    group_out,
    d_a_ptrs,
    d_b_ptrs,
    d_c_ptrs,
    d_output_ptrs,
    d_m_sizes,
    d_n_sizes,
    d_k_sizes,
    d_ldas,
    d_ldbs,
    d_ldcs,
    group_size,
    M,
    N,
    K,
    alpha,
    beta,
    use_small_m=False,
):
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    grouped_bfgemm_kernel[(NUM_SMS,)](
        M,
        N,
        K,
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_output_ptrs,
        d_m_sizes,
        d_n_sizes,
        d_k_sizes,
        d_ldas,
        d_ldbs,
        d_ldcs,
        group_size,
        alpha=alpha,
        beta=beta,
    )

    return group_out


@libentry()
@libtuner(configs=get_autotune_config(), key=["M", "N", "K"])
@triton.jit
def grouped_hgemm_kernel(
    M,
    N,
    K,
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_out_ptrs,
    group_m_sizes,
    group_n_sizes,
    group_k_sizes,
    group_ldas,
    group_ldbs,
    group_ldcs,
    group_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    total_grid = tl.num_programs(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_m_sizes + g)
        gn = tl.load(group_n_sizes + g)
        gk = tl.load(group_k_sizes + g)
        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles

        current_problem_end = last_problem_end + num_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            lda = tl.load(group_ldas + g)
            ldb = tl.load(group_ldbs + g)
            ldc = tl.load(group_ldcs + g)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            out_ptr = tl.load(group_out_ptrs + g).to(tl.pointer_type(tl.float16))

            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx, tile_n_idx = grouped_launch(
                    tile_idx_in_gemm, gm, gn, BLOCK_M, BLOCK_N, GROUP_M
                )

                offs_am = tile_m_idx * BLOCK_M
                offs_bn = tile_n_idx * BLOCK_N

                a_ptrs = tl.make_block_ptr(
                    base=a_ptr,
                    shape=(gm, gk),
                    strides=(lda, 1),
                    offsets=(offs_am, 0),
                    block_shape=(BLOCK_M, BLOCK_K),
                    order=(1, 0),
                )
                b_ptrs = tl.make_block_ptr(
                    base=b_ptr,
                    shape=(gk, gn),
                    strides=(ldb, 1),
                    offsets=(0, offs_bn),
                    block_shape=(BLOCK_K, BLOCK_N),
                    order=(1, 0),
                )

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(gk, BLOCK_K)):
                    a = tl.load(a_ptrs, boundary_check=(0, 1))
                    b = tl.load(b_ptrs, boundary_check=(0, 1))
                    accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)
                    a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
                    b_ptrs = tl.advance(b_ptrs, (BLOCK_K, 0))

                offs_cm = tile_m_idx * BLOCK_M
                offs_cn = tile_n_idx * BLOCK_N

                out_ptrs = tl.make_block_ptr(
                    base=out_ptr,
                    shape=(gm, gn),
                    strides=(ldc, 1),
                    offsets=(offs_cm, offs_cn),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                if beta == 0.0:
                    accumulator = accumulator * alpha
                else:
                    c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
                    c_ptrs = tl.make_block_ptr(
                        base=c_ptr,
                        shape=(gm, gn),
                        strides=(ldc, 1),
                        offsets=(offs_cm, offs_cn),
                        block_shape=(BLOCK_M, BLOCK_N),
                        order=(1, 0),
                    )
                    ori_c = tl.load(c_ptrs, boundary_check=(0, 1))
                    accumulator = ori_c * beta + accumulator * alpha

                c = accumulator.to(out_ptrs.dtype.element_ty)
                tl.store(out_ptrs, c, boundary_check=(0, 1))

                tile_idx += total_grid

        last_problem_end = current_problem_end


def group_hgemm(
    group_out,
    d_a_ptrs,
    d_b_ptrs,
    d_c_ptrs,
    d_output_ptrs,
    d_m_sizes,
    d_n_sizes,
    d_k_sizes,
    d_ldas,
    d_ldbs,
    d_ldcs,
    group_size,
    M,
    N,
    K,
    alpha,
    beta,
    use_small_m=False,
):
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    grouped_hgemm_kernel[(NUM_SMS,)](
        M,
        N,
        K,
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_output_ptrs,
        d_m_sizes,
        d_n_sizes,
        d_k_sizes,
        d_ldas,
        d_ldbs,
        d_ldcs,
        group_size,
        alpha=alpha,
        beta=beta,
    )

    return group_out


@libentry()
@libtuner(configs=get_autotune_config_tf32(), key=["M", "N", "K"])
@triton.jit
def grouped_tf32gemm_kernel(
    M,
    N,
    K,
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_out_ptrs,
    group_m_sizes,
    group_n_sizes,
    group_k_sizes,
    group_ldas,
    group_ldbs,
    group_ldcs,
    group_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    total_grid = tl.num_programs(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_m_sizes + g)
        gn = tl.load(group_n_sizes + g)
        gk = tl.load(group_k_sizes + g)
        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles

        current_problem_end = last_problem_end + num_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            lda = tl.load(group_ldas + g)
            ldb = tl.load(group_ldbs + g)
            ldc = tl.load(group_ldcs + g)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float32))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float32))
            out_ptr = tl.load(group_out_ptrs + g).to(tl.pointer_type(tl.float32))

            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx, tile_n_idx = grouped_launch(
                    tile_idx_in_gemm, gm, gn, BLOCK_M, BLOCK_N, GROUP_M
                )

                offs_am = tile_m_idx * BLOCK_M
                offs_bn = tile_n_idx * BLOCK_N

                a_ptrs = tl.make_block_ptr(
                    base=a_ptr,
                    shape=(gm, gk),
                    strides=(lda, 1),
                    offsets=(offs_am, 0),
                    block_shape=(BLOCK_M, BLOCK_K),
                    order=(1, 0),
                )
                b_ptrs = tl.make_block_ptr(
                    base=b_ptr,
                    shape=(gn, gk),
                    strides=(ldb, 1),
                    offsets=(offs_bn, 0),
                    block_shape=(BLOCK_N, BLOCK_K),
                    order=(1, 0),
                )
                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(gk, BLOCK_K)):
                    a = tl.load(a_ptrs, boundary_check=(0, 1))
                    b = tl.load(b_ptrs, boundary_check=(0, 1))
                    accumulator = tl.dot(
                        a, b.T, acc=accumulator, input_precision="tf32"
                    )
                    a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
                    b_ptrs = tl.advance(b_ptrs, (0, BLOCK_K))

                offs_cm = tile_m_idx * BLOCK_M
                offs_cn = tile_n_idx * BLOCK_N

                out_ptrs = tl.make_block_ptr(
                    base=out_ptr,
                    shape=(gm, gn),
                    strides=(ldc, 1),
                    offsets=(offs_cm, offs_cn),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                if beta == 0.0:
                    accumulator = accumulator * alpha
                else:
                    c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float32))
                    c_ptrs = tl.make_block_ptr(
                        base=c_ptr,
                        shape=(gm, gn),
                        strides=(ldc, 1),
                        offsets=(offs_cm, offs_cn),
                        block_shape=(BLOCK_M, BLOCK_N),
                        order=(1, 0),
                    )
                    ori_c = tl.load(c_ptrs, boundary_check=(0, 1))
                    accumulator = ori_c * beta + accumulator * alpha

                c = accumulator.to(out_ptrs.dtype.element_ty)
                tl.store(out_ptrs, c, boundary_check=(0, 1))

                tile_idx += total_grid

        last_problem_end = current_problem_end


def group_tf32gemm(
    group_out,
    d_a_ptrs,
    d_b_ptrs,
    d_c_ptrs,
    d_output_ptrs,
    d_m_sizes,
    d_n_sizes,
    d_k_sizes,
    d_ldas,
    d_ldbs,
    d_ldcs,
    group_size,
    M,
    N,
    K,
    alpha,
    beta,
    use_small_m=False,
):
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    grouped_tf32gemm_kernel[(NUM_SMS,)](
        M,
        N,
        K,
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_output_ptrs,
        d_m_sizes,
        d_n_sizes,
        d_k_sizes,
        d_ldas,
        d_ldbs,
        d_ldcs,
        group_size,
        alpha=alpha,
        beta=beta,
    )

    return group_out


@libentry()
@libtuner(configs=get_autotune_config(), key=["M", "N", "K"])
@triton.jit
def grouped_mm_kernel(
    A,
    B,
    C,
    offs,
    num_groups: tl.constexpr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    total_grid = tl.num_programs(axis=0)
    tile_idx = tl.program_id(axis=0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    last_problem_end = 0
    group_start = 0
    group_end = 0

    for group_idx in tl.range(num_groups):
        group_end = tl.load(offs + group_idx).to(tl.int32)
        m = group_end - group_start
        num_m_tiles = tl.cdiv(m, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles

        current_problem_end = last_problem_end + num_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx, tile_n_idx = grouped_launch(
                    tile_idx_in_gemm, m, N, BLOCK_M, BLOCK_N, GROUP_M
                )

                offs_am = group_start + tile_m_idx * BLOCK_M
                offs_bn = tile_n_idx * BLOCK_N
                offs_bk = group_idx * K

                a_block_ptr = tl.make_block_ptr(
                    base=A,
                    shape=(M, K),
                    strides=(stride_am, stride_ak),
                    offsets=(offs_am, 0),
                    block_shape=(BLOCK_M, BLOCK_K),
                    order=(1, 0),
                )

                b_block_ptr = tl.make_block_ptr(
                    base=B,
                    shape=(num_groups * K, N),
                    strides=(stride_bk, stride_bn),
                    offsets=(offs_bk, offs_bn),
                    block_shape=(BLOCK_K, BLOCK_N),
                    order=(1, 0),
                )

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    a = tl.load(a_block_ptr, boundary_check=(0, 1))
                    b = tl.load(b_block_ptr, boundary_check=(0, 1))
                    accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)

                    a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
                    b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

                c = accumulator.to(C.dtype.element_ty)

                c_block_ptr = tl.make_block_ptr(
                    base=C,
                    shape=(M, N),
                    strides=(stride_cm, stride_cn),
                    offsets=(offs_am, offs_bn),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )

                if offs_am + BLOCK_M <= group_end:
                    tl.store(c_block_ptr, c, boundary_check=(0, 1))
                else:
                    offs_cm = offs_am + tl.arange(0, BLOCK_M)
                    offs_cn = offs_bn + tl.arange(0, BLOCK_N)
                    c_ptrs = (
                        C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
                    )
                    c_mask = (offs_cm[:, None] < group_end) & (offs_cn[None, :] < N)
                    tl.store(c_ptrs, c, mask=c_mask)

                tile_idx += total_grid

        last_problem_end = current_problem_end
        group_start = group_end


def group_mm(A: torch.Tensor, B: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 2
    assert B.dim() == 3
    M, K = A.shape

    num_groups, BK, N = B.shape
    strideBK, strideBN = B.stride(1), B.stride(2)

    assert num_groups == offs.numel()
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    C = A.new_empty(M, N)
    grouped_mm_kernel[(NUM_SMS,)](
        A,
        B,
        C,
        offs,
        num_groups,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        strideBK,
        strideBN,
        C.stride(0),
        C.stride(1),
    )

    return C
