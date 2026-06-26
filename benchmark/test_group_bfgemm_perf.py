import ctypes
import ctypes.util
import random
from typing import Generator

import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from flag_blas.ops import CUBLAS_OP_N

from benchmark.performance_utils import Benchmark
from flag_blas.utils import shape_utils


def load_cublas():
    lib_names = ["libcublas.so", "libcublas.so.12", "libcublas.so.11"]
    found_path = ctypes.util.find_library("cublas")
    if found_path:
        lib_names.insert(0, found_path)
    for name in lib_names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError("Unable to find libcublas.so on the system.")


_cublas = load_cublas()


def _cublasGemmGroupedBatchedEx(
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
    return _cublas.cublasGemmGroupedBatchedEx(
        ctypes.c_void_p(handle),
        ctypes.c_void_p(transa.data_ptr()),
        ctypes.c_void_p(transb.data_ptr()),
        ctypes.c_void_p(m_arr.data_ptr()),
        ctypes.c_void_p(n_arr.data_ptr()),
        ctypes.c_void_p(k_arr.data_ptr()),
        ctypes.c_void_p(alpha),
        ctypes.c_void_p(a_array),
        ctypes.c_int(a_type),
        ctypes.c_void_p(lda.data_ptr()),
        ctypes.c_void_p(b_array),
        ctypes.c_int(b_type),
        ctypes.c_void_p(ldb.data_ptr()),
        ctypes.c_void_p(beta),
        ctypes.c_void_p(c_array),
        ctypes.c_int(c_type),
        ctypes.c_void_p(ldc.data_ptr()),
        ctypes.c_int(group_count),
        ctypes.c_void_p(group_size.data_ptr()),
        ctypes.c_int(compute_type),
    )


cublas.cublasGemmGroupedBatchedEx = _cublasGemmGroupedBatchedEx


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

SEED = 50
CUDA_R_16BF = 14
CUBLAS_COMPUTE_32F = 0


def cublas_group_gemm(
    group_A,
    group_B,
    group_C,
    offs_table,
    transa,
    transb,
    cu_m_arr,
    cu_n_arr,
    cu_k_arr,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    a_cublas,
    b_cublas,
    c_cublas,
    alpha_arr,
    beta_arr,
    batch,
    cu_dtype,
    compute_type,
    out_cublas,
    handle,
    a_flag,
    b_flag,
    c_flag,
    out_flag_ptrs,
    sizes_flag,
    lds_flag,
    group_size,
    M,
    N,
    K,
    out_flag,
    alpha,
    beta,
    **kwargs,
):
    cublas.cublasGemmGroupedBatchedEx(
        handle,
        transa,
        transb,
        cu_m_arr,
        cu_n_arr,
        cu_k_arr,
        alpha_arr.data_ptr(),
        a_cublas.data_ptr(),
        cu_dtype,
        lda_cublas,
        b_cublas.data_ptr(),
        cu_dtype,
        ldb_cublas,
        beta_arr.data_ptr(),
        c_cublas.data_ptr(),
        cu_dtype,
        ldc_cublas,
        group_size,
        batch,
        compute_type,
    )
    return out_cublas


def gems_group_gemm_wrapper(
    group_A,
    group_B,
    group_C,
    offs_table,
    transa,
    transb,
    cu_m_arr,
    cu_n_arr,
    cu_k_arr,
    lda_cublas,
    ldb_cublas,
    ldc_cublas,
    a_cublas,
    b_cublas,
    c_cublas,
    alpha_arr,
    beta_arr,
    batch,
    cu_dtype,
    compute_type,
    out_cublas,
    handle,
    a_flag,
    b_flag,
    c_flag,
    out_flag_ptrs,
    sizes_flag,
    lds_flag,
    group_size,
    M,
    N,
    K,
    out_flag,
    alpha,
    beta,
    **kwargs,
):
    return flag_blas.group_bfgemm(
        out_flag,
        a_flag,
        b_flag,
        c_flag,
        out_flag_ptrs,
        sizes_flag,
        lds_flag,
        group_size,
        M,
        N,
        K,
        alpha=alpha,
        beta=beta,
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

        scale = 1.0
        random.seed(SEED)
        for k, e, n in GROUP_GEMM_SHAPES:
            m_list = [random.randint(1, 4096) for _ in range(e)]
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

            offs = []
            start_M = 0
            start_K = 0
            for g in range(e):
                mg = m_list[g]
                offs.append([mg, n, k, start_M, start_K, start_M])
                start_M += mg
                start_K += k

            out_cublas = torch.empty_like(group_C)
            out_flag = torch.empty_like(group_C)

            cu_a_ptrs = []
            cu_b_ptrs = []
            cu_c_ptrs = []
            cu_m_list = []
            cu_n_list = []
            cu_k_list = []
            cu_lda_list = []
            cu_ldb_list = []
            cu_ldc_list = []

            flag_a_ptrs = []
            flag_b_ptrs = []
            flag_c_ptrs = []
            flag_out_ptrs = []
            flag_sizes = []
            flag_lds = []

            for entry in offs:
                mg, ng, kg, start_M_offs, start_K_offs, start_C_offs = entry

                cu_a_ptrs.append(
                    group_B[start_K_offs : start_K_offs + kg, :].data_ptr()
                )
                cu_b_ptrs.append(
                    group_A[start_M_offs : start_M_offs + mg, :].data_ptr()
                )
                cu_c_ptrs.append(
                    out_cublas[start_C_offs : start_C_offs + mg, :].data_ptr()
                )
                cu_m_list.append(ng)
                cu_n_list.append(mg)
                cu_k_list.append(kg)
                cu_lda_list.append(ng)
                cu_ldb_list.append(kg)
                cu_ldc_list.append(ng)

                flag_a_ptrs.append(
                    group_A[start_M_offs : start_M_offs + mg, :].data_ptr()
                )
                flag_b_ptrs.append(
                    group_B[start_K_offs : start_K_offs + kg, :].data_ptr()
                )
                flag_c_ptrs.append(
                    group_C[start_C_offs : start_C_offs + mg, :].data_ptr()
                )
                flag_out_ptrs.append(
                    out_flag[start_C_offs : start_C_offs + mg, :].data_ptr()
                )
                flag_sizes += [mg, ng, kg]
                flag_lds += [kg, ng, ng]

            yield group_A, group_B, group_C, offs, {
                "transa": torch.tensor([CUBLAS_OP_N] * e, dtype=torch.int32),
                "transb": torch.tensor([CUBLAS_OP_N] * e, dtype=torch.int32),
                "cu_m_arr": torch.tensor(cu_m_list, dtype=torch.int32),
                "cu_n_arr": torch.tensor(cu_n_list, dtype=torch.int32),
                "cu_k_arr": torch.tensor(cu_k_list, dtype=torch.int32),
                "lda_cublas": torch.tensor(cu_lda_list, dtype=torch.int32),
                "ldb_cublas": torch.tensor(cu_ldb_list, dtype=torch.int32),
                "ldc_cublas": torch.tensor(cu_ldc_list, dtype=torch.int32),
                "a_cublas": torch.tensor(
                    cu_a_ptrs, dtype=torch.int64, device=self.device
                ),
                "b_cublas": torch.tensor(
                    cu_b_ptrs, dtype=torch.int64, device=self.device
                ),
                "c_cublas": torch.tensor(
                    cu_c_ptrs, dtype=torch.int64, device=self.device
                ),
                "alpha_arr": torch.full((e,), self.alpha, dtype=torch.float32),
                "beta_arr": torch.full((e,), self.beta, dtype=torch.float32),
                "batch": torch.tensor([1] * e, dtype=torch.int32),
                "cu_dtype": CUDA_R_16BF,
                "compute_type": CUBLAS_COMPUTE_32F,
                "out_cublas": out_cublas,
                "a_flag": torch.tensor(
                    flag_a_ptrs, dtype=torch.int64, device=self.device
                ),
                "b_flag": torch.tensor(
                    flag_b_ptrs, dtype=torch.int64, device=self.device
                ),
                "c_flag": torch.tensor(
                    flag_c_ptrs, dtype=torch.int64, device=self.device
                ),
                "out_flag_ptrs": torch.tensor(
                    flag_out_ptrs, dtype=torch.int64, device=self.device
                ),
                "sizes_flag": torch.tensor(
                    flag_sizes, dtype=torch.int32, device=self.device
                ),
                "lds_flag": torch.tensor(
                    flag_lds, dtype=torch.int32, device=self.device
                ),
                "group_size": e,
                "M": total_M,
                "N": n,
                "K": k,
                "out_flag": out_flag,
                "alpha": self.alpha,
                "beta": self.beta,
                "handle": handle,
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
def test_perf_group_gemm_bf16():
    bench = GroupGemmBenchmark(
        op_name="group_gemm",
        torch_op=cublas_group_gemm,
        gems_op=gems_group_gemm_wrapper,
        dtypes=[torch.bfloat16],
    )
    for cur_dtype in bench.dtypes:
        for A, B, C, offs, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_group_gemm(A, B, C.clone(), offs, **kwargs)
            gems_result = gems_group_gemm_wrapper(A, B, C.clone(), offs, **kwargs)
            bench.validate_results(torch_result, gems_result, 1, tolerance=1e-2)
    bench.run()
