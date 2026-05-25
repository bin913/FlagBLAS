import ctypes
import random
from typing import Generator

import cupy as cp
import numpy as np
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from flag_blas.ops import CUBLAS_OP_N

from benchmark.performance_utils import Benchmark
from flag_blas.utils import shape_utils

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


GROUP_GEMM_SHAPES = [
    (2048, 128, 1536),
    (768, 128, 2048),
    (2048, 128, 768),
    (384, 128, 2048),
    (2048, 128, 384),
    (192, 128, 2048),
    (2048, 128, 192),
    (96, 128, 2048),
    (2048, 64, 1536),
    (768, 64, 2048),
    (2048, 32, 1536),
    (768, 32, 2048),
    (2048, 16, 1536),
    (768, 16, 2048),
    (4096, 128, 384),
    (192, 128, 4096),
    (4096, 16, 3072),
    (1536, 16, 4096),
    (7168, 16, 4096),
    (2048, 16, 7168),
    (7168, 17, 4096),
    (2048, 17, 7168),
    (2048, 512, 256),
    (128, 512, 2048),
    (2048, 512, 128),
    (64, 512, 2048),
    (2048, 128, 1024),
    (512, 128, 2048),
    (2048, 64, 1024),
    (512, 64, 2048),
]

SEED = 42
GROUP_GEMM_M_VALUES = list(range(1, 33)) + [64, 128, 256, 512, 1024, 2048, 4096]

CUDA_R_16F = 2
CUBLAS_COMPUTE_32F = 0


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
    e = len(offs_table)
    if e == 0:
        return torch.empty_like(group_C)

    cu_dtype = CUDA_R_16F

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
        CUBLAS_COMPUTE_32F,
    )

    return out


def gems_group_gemm_wrapper(
    group_A, group_B, group_C, offs_table, alpha, beta, **kwargs
):
    return flag_blas.group_hgemm(
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
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
        cublas.setMathMode(handle, 0)
        torch.backends.cuda.matmul.allow_tf32 = False

        alpha_np = np.array(self.alpha, dtype=np.float32)
        beta_np = np.array(self.beta, dtype=np.float32)
        alpha_ptr = alpha_np.ctypes.data
        beta_ptr = beta_np.ctypes.data

        scale = 1.0
        random.seed(SEED)
        for k, e, n in GROUP_GEMM_SHAPES:
            m_list = [random.choice(GROUP_GEMM_M_VALUES) for _ in range(e)]
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

    def validate_results(self, torch_result, gems_result, reduce_dim, tolerance=1e-2):
        torch_cpu = torch_result.cpu()
        gems_cpu = gems_result.cpu()

        try:
            flag_blas.testing.assert_close(
                gems_cpu,
                torch_cpu,
                torch_cpu.dtype,
                equal_nan=False,
                reduce_dim=reduce_dim,
                atol=tolerance,
            )
        except AssertionError:
            max_abs_diff = torch.max(torch.abs(torch_cpu - gems_cpu))
            max_rel_diff = torch.max(
                torch.abs((torch_cpu - gems_cpu) / (torch.abs(torch_cpu) + 1e-9))
            )
            raise AssertionError(
                f"Results differ beyond tolerance {tolerance}:\n"
                f"Max absolute difference: {max_abs_diff}\n"
                f"Max relative difference: {max_rel_diff}\n"
                f"Shape: {torch_cpu.shape}"
            )


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
    for cur_dtype in bench.dtypes:
        for A, B, C, offs, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_group_gemm_baseline(A, B, C.clone(), offs, **kwargs)
            gems_result = gems_group_gemm_wrapper(A, B, C.clone(), offs, **kwargs)
            bench.validate_results(torch_result, gems_result, 1, tolerance=1e-2)
    bench.run()
