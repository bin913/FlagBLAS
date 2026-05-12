import ctypes
from typing import Generator

import cupy as cp
import numpy as np
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from benchmark.performance_utils import Benchmark
from flag_blas.utils import shape_utils

# ── monkey-patch: cupy 14.0.1 未绑定 cublasGemmGroupedBatchedEx ──────────────
if not hasattr(cublas, "cublasGemmGroupedBatchedEx"):
    _libcublas = ctypes.CDLL("libcublas.so.12")

    def _cublasGemmGroupedBatchedEx_impl(
        handle,
        transa,
        transb,
        m_arr,
        n_arr,
        k_arr,
        alpha,
        a_array,
        a_type,
        lda,
        b_array,
        b_type,
        ldb,
        beta,
        c_array,
        c_type,
        ldc,
        group_count,
        group_size,
        compute_type,
    ):
        return _libcublas.cublasGemmGroupedBatchedEx(
            ctypes.c_void_p(handle),
            transa.ctypes.data_as(ctypes.c_void_p),
            transb.ctypes.data_as(ctypes.c_void_p),
            m_arr.ctypes.data_as(ctypes.c_void_p),
            n_arr.ctypes.data_as(ctypes.c_void_p),
            k_arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(alpha),
            ctypes.c_void_p(a_array),
            ctypes.c_int(a_type),
            lda.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(b_array),
            ctypes.c_int(b_type),
            ldb.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(beta),
            ctypes.c_void_p(c_array),
            ctypes.c_int(c_type),
            ldc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(group_count),
            group_size.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(compute_type),
        )

    cublas.cublasGemmGroupedBatchedEx = _cublasGemmGroupedBatchedEx_impl


GROUP_GEMM_CONFIGS = [
    (2048, 128, 1536),
    # (768, 128, 2048),
    # (2048, 128, 768),
    # (384, 128, 2048),
    # (2048, 128, 384),
    # (192, 128, 2048),
    # (2048, 128, 192),
    # (96, 128, 2048),
    # (2048, 64, 1536),
    # (768, 64, 2048),
    # (2048, 32, 1536),
    # (768, 32, 2048),
    # (2048, 16, 1536),
    # (768, 16, 2048),
    # (4096, 128, 384),
    # (192, 128, 4096),
    # (4096, 16, 3072),
    # (1536, 16, 4096),
    # (7168, 16, 4096),
    # (2048, 16, 7168),
    # (7168, 17, 4096),
    # (2048, 17, 7168),
    # (2048, 512, 256),
    # (128, 512, 2048),
    # (2048, 512, 128),
    # (64, 512, 2048),
    # (2048, 128, 1024),
    # (512, 128, 2048),
    # (2048, 64, 1024),
    # (512, 64, 2048),
]

def _get_m_values():
    """M 取值规范：1~32 全部，大于 32 取 2 的次幂，共 43 个值。"""
    return list(range(1, 33)) + [64, 128, 256, 512, 1024, 2048, 4096]


CUBLAS_OP_N = 0
CUDA_R_16F = 2
CUDA_R_16BF = 14
CUBLAS_COMPUTE_32F = 0
CUBLAS_COMPUTE_32F_FAST_16F = 74
CUBLAS_COMPUTE_32F_FAST_16BF = 75


def _build_offs_table(k, e, n, m_list):
    offs = []
    start_M = 0
    start_K = 0
    for g in range(e):
        mg = m_list[g]
        offs.append([mg, n, k, start_M, start_K, start_M])
        start_M += mg
        start_K += k
    return offs


def _compute_cpu_reference(group_A, group_B, group_C, offs_table, alpha, beta):
    """CPU float32 参考结果，用于验证 GPU 实现的正确性。"""
    ref_out = group_C.clone().to(torch.float32)
    A_f32 = group_A.to(torch.float32)
    B_f32 = group_B.to(torch.float32)
    for entry in offs_table:
        m_g, n_g, k_g, start_M, start_K, start_C = entry
        A_sub = A_f32[start_M : start_M + m_g, :k_g]
        B_sub = B_f32[start_K : start_K + k_g, :n_g]
        res = torch.matmul(A_sub, B_sub)
        if beta == 0.0:
            ref_out[start_C : start_C + m_g, :n_g] = alpha * res
        else:
            ref_out[start_C : start_C + m_g, :n_g] = (
                alpha * res + beta * ref_out[start_C : start_C + m_g, :n_g]
            )
    return ref_out.to(group_A.dtype)


def _setup_cublas_handle():
    """获取 cuBLAS handle 并配置 host-side pointer mode."""
    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    cublas.setMathMode(handle, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    return handle


def cublas_group_gemm_baseline(
    group_A,
    group_B,
    group_C,
    offs_table,
    alpha,
    beta,
    handle,
    **kwargs,
):
    """cuBLAS grouped GEMM via ``cublasGemmGroupedBatchedEx``.

    PyTorch 使用行优先，而 cuBLAS 使用列优先。
    行优先的 C = A @ B 等价于列优先的 C^T = B^T @ A^T。
    因此调用 cuBLAS 时需要交换 A/B 数据源，并修正对应的切片索引。
    """
    e = len(offs_table)
    if e == 0:
        return torch.empty_like(group_C)

    if group_A.dtype == torch.float16:
        cu_dtype = CUDA_R_16F
    else:
        cu_dtype = CUDA_R_16BF
    cu_compute = CUBLAS_COMPUTE_32F

    out = torch.empty_like(group_C)

    a_ptrs = []
    b_ptrs = []
    c_ptrs = []
    m_arr_list = []
    n_arr_list = []
    k_arr_list = []
    lda_list = []
    ldb_list = []
    ldc_list = []

    for entry in offs_table:
        mg, ng, kg, start_M, start_K, start_C = entry
        a_ptrs.append(group_B[start_K : start_K + kg, :].data_ptr())
        b_ptrs.append(group_A[start_M : start_M + mg, :].data_ptr())
        c_ptrs.append(out[start_C : start_C + mg, :].data_ptr())
        m_arr_list.append(ng)
        n_arr_list.append(mg)
        k_arr_list.append(kg)
        lda_list.append(ng)
        ldb_list.append(kg)
        ldc_list.append(ng)

    transa = np.array([CUBLAS_OP_N] * e, dtype=np.int32)
    transb = np.array([CUBLAS_OP_N] * e, dtype=np.int32)
    m_arr = np.array(m_arr_list, dtype=np.int32)
    n_arr = np.array(n_arr_list, dtype=np.int32)
    k_arr = np.array(k_arr_list, dtype=np.int32)
    lda_arr = np.array(lda_list, dtype=np.int32)
    ldb_arr = np.array(ldb_list, dtype=np.int32)
    ldc_arr = np.array(ldc_list, dtype=np.int32)
    batch = np.array([1] * e, dtype=np.int32)
    alpha_arr = np.full(e, alpha, dtype=np.float32)
    beta_arr = np.full(e, beta, dtype=np.float32)

    device = group_A.device
    d_a_ptrs = torch.tensor(a_ptrs, dtype=torch.int64, device=device)
    d_b_ptrs = torch.tensor(b_ptrs, dtype=torch.int64, device=device)
    d_c_ptrs = torch.tensor(c_ptrs, dtype=torch.int64, device=device)

    cublas.cublasGemmGroupedBatchedEx(
        handle,
        transa,
        transb,
        m_arr,
        n_arr,
        k_arr,
        alpha_arr.ctypes.data,
        d_a_ptrs.data_ptr(),
        cu_dtype,
        lda_arr,
        d_b_ptrs.data_ptr(),
        cu_dtype,
        ldb_arr,
        beta_arr.ctypes.data,
        d_c_ptrs.data_ptr(),
        cu_dtype,
        ldc_arr,
        e,
        batch,
        cu_compute,
    )

    return out


def gems_group_gemm_wrapper(
    group_A, group_B, group_C, offs_table, alpha, beta, **kwargs
):
    return flag_blas.group_gemm(
        group_A, group_B, group_C, offs_table, alpha=alpha, beta=beta
    )


class GroupGemmBenchmark(Benchmark):

    def __init__(self, *args, alpha=1.0, beta=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = _setup_cublas_handle()

        alpha_np = np.array(self.alpha, dtype=np.float32)
        beta_np = np.array(self.beta, dtype=np.float32)
        alpha_ptr = alpha_np.ctypes.data
        beta_ptr = beta_np.ctypes.data

        m_values = _get_m_values()
        scale = 1.0
        for k, e, n in GROUP_GEMM_CONFIGS:
            m_list = [m_values[i % len(m_values)] for i in range(e)]
            total_M = sum(m_list)
            total_K = e * k

            group_A = (
                torch.randn(total_M, k, dtype=cur_dtype, device=self.device) * scale
            )
            group_B = (
                torch.randn(total_K, n, dtype=cur_dtype, device=self.device) * scale
            )
            group_C = (
                torch.randn(total_M, n, dtype=cur_dtype, device=self.device) * scale
            )
            offs_table = _build_offs_table(k, e, n, m_list)

            yield group_A, group_B, group_C, offs_table, {
                "alpha": self.alpha,
                "beta": self.beta,
                "handle": handle,
                "alpha_ptr": alpha_ptr,
                "beta_ptr": beta_ptr,
            }

    def get_tflops(self, op, *args, **kwargs):
        offs_table = args[3]
        total_flops = 0
        for entry in offs_table:
            m_g, n_g, k_g = entry[0], entry[1], entry[2]
            total_flops += 2 * m_g * n_g * k_g
        return total_flops

    def get_gbps(self, args, latency):
        group_A, group_B, group_C = args[0], args[1], args[2]
        io_amount = (
            shape_utils.size_in_bytes(group_A)
            + shape_utils.size_in_bytes(group_B)
            + 2 * shape_utils.size_in_bytes(group_C)
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.group_gemm
def test_perf_group_gemm_bf16():
    bench = GroupGemmBenchmark(
        op_name="group_gemm",
        torch_op=cublas_group_gemm_baseline,
        gems_op=gems_group_gemm_wrapper,
        dtypes=[torch.bfloat16],
        alpha=1.0,
        beta=0.0,
    )
    bench.run()


@pytest.mark.group_gemm
def test_perf_group_gemm_fp16():
    bench = GroupGemmBenchmark(
        op_name="group_gemm",
        torch_op=cublas_group_gemm_baseline,
        gems_op=gems_group_gemm_wrapper,
        dtypes=[torch.float16],
        alpha=1.0,
        beta=0.0,
    )
    bench.run()

CORRECTNESS_CONFIGS = [
    (2048, 128, 1536),
    (768, 128, 2048),
    (2048, 16, 1536),
    (768, 16, 2048),
    (2048, 512, 256),
    (128, 512, 2048),
]


@pytest.mark.group_gemm
@pytest.mark.parametrize("k_en", CORRECTNESS_CONFIGS)
def test_correctness_group_gemm_fp16(k_en):
    """验证 fp16：cuBLAS == flag_blas == CPU float32 参考"""
    k, e, n = k_en
    device = flag_blas.device

    handle = _setup_cublas_handle()

    torch.manual_seed(42)
    np.random.seed(42)

    m_values = _get_m_values()
    m_list = [m_values[i % len(m_values)] for i in range(e)]
    total_M = sum(m_list)
    total_K = e * k

    group_A = torch.randn(total_M, k, dtype=torch.float16, device=device)
    group_B = torch.randn(total_K, n, dtype=torch.float16, device=device)
    group_C = torch.randn(total_M, n, dtype=torch.float16, device=device)
    offs_table = _build_offs_table(k, e, n, m_list)

    out_cublas = cublas_group_gemm_baseline(
        group_A, group_B, group_C, offs_table,
        alpha=1.0, beta=0.0, handle=handle,
    )

    out_flag = gems_group_gemm_wrapper(
        group_A, group_B, group_C, offs_table, alpha=1.0, beta=0.0,
    )

    cpu_ref = _compute_cpu_reference(group_A, group_B, group_C, offs_table, 1.0, 0.0)

    # cuBLAS vs flag_blas
    torch.testing.assert_close(out_cublas, out_flag, rtol=1e-2, atol=1e-2)
    # flag_blas vs CPU reference
    torch.testing.assert_close(out_flag.cpu(), cpu_ref.cpu(), rtol=1e-2, atol=1e-2)


@pytest.mark.group_gemm
@pytest.mark.parametrize("k_en", CORRECTNESS_CONFIGS)
def test_correctness_group_gemm_bf16(k_en):
    """验证 bf16：cuBLAS == flag_blas == CPU float32 参考"""
    k, e, n = k_en
    device = flag_blas.device

    handle = _setup_cublas_handle()

    torch.manual_seed(42)
    np.random.seed(42)

    m_values = _get_m_values()
    m_list = [m_values[i % len(m_values)] for i in range(e)]
    total_M = sum(m_list)
    total_K = e * k

    group_A = torch.randn(total_M, k, dtype=torch.bfloat16, device=device)
    group_B = torch.randn(total_K, n, dtype=torch.bfloat16, device=device)
    group_C = torch.randn(total_M, n, dtype=torch.bfloat16, device=device)
    offs_table = _build_offs_table(k, e, n, m_list)

    out_cublas = cublas_group_gemm_baseline(
        group_A, group_B, group_C, offs_table,
        alpha=1.0, beta=0.0, handle=handle,
    )

    out_flag = gems_group_gemm_wrapper(
        group_A, group_B, group_C, offs_table, alpha=1.0, beta=0.0,
    )

    cpu_ref = _compute_cpu_reference(group_A, group_B, group_C, offs_table, 1.0, 0.0)

    # cuBLAS vs flag_blas
    torch.testing.assert_close(out_cublas, out_flag, rtol=1e-2, atol=1e-2)
    # flag_blas vs CPU reference
    torch.testing.assert_close(out_flag.cpu(), cpu_ref.cpu(), rtol=1e-2, atol=1e-2)
