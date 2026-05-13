import ctypes
import random

import cupy as cp
import numpy as np
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas

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

DTYPES = [torch.bfloat16, torch.float16]

M_MIN, M_MAX = 1, 4096

CUBLAS_OP_N = 0
CUDA_R_16F = 2
CUDA_R_16BF = 14
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


def _compute_reference(group_A, group_B, group_C, offs_table, alpha, beta):
    dtype = group_A.dtype
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

    return ref_out.to(dtype)


def cublas_group_gemm_reference(group_A, group_B, group_C, offs_table, alpha, beta):
    e = len(offs_table)
    if e == 0:
        return torch.empty_like(group_C)

    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    cublas.setMathMode(handle, 0)
    torch.backends.cuda.matmul.allow_tf32 = False

    if group_A.dtype == torch.float16:
        cu_dtype = CUDA_R_16F
    else:
        cu_dtype = CUDA_R_16BF

    out = group_C.clone()

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


@pytest.mark.group_gemm
@pytest.mark.parametrize("config_idx", range(len(GROUP_GEMM_CONFIGS)))
@pytest.mark.parametrize("dtype", DTYPES)
def test_accuracy_group_gemm(config_idx, dtype):
    k, e, n = GROUP_GEMM_CONFIGS[config_idx]
    device = flag_blas.device
    alpha, beta = 1.5, 0.5
    scale = k**-0.5

    m_list = [random.randint(M_MIN, M_MAX) for _ in range(e)]
    total_M = sum(m_list)
    total_K = e * k

    group_A = torch.randn(total_M, k, dtype=dtype, device=device) * scale
    group_B = torch.randn(total_K, n, dtype=dtype, device=device) * scale
    group_C = torch.randn(total_M, n, dtype=dtype, device=device) * scale

    offs_table = _build_offs_table(k, e, n, m_list)

    ref = _compute_reference(group_A, group_B, group_C, offs_table, alpha, beta)

    out = flag_blas.group_gemm(
        group_A, group_B, group_C, offs_table, alpha=alpha, beta=beta
    )

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.group_gemm
@pytest.mark.parametrize("config_idx", range(len(GROUP_GEMM_CONFIGS)))
@pytest.mark.parametrize("dtype", DTYPES)
def test_accuracy_group_gemm_cublas(config_idx, dtype):
    k, e, n = GROUP_GEMM_CONFIGS[config_idx]
    device = flag_blas.device
    alpha, beta = 1.5, 0.5
    scale = k**-0.5

    m_list = [random.randint(M_MIN, M_MAX) for _ in range(e)]
    total_M = sum(m_list)
    total_K = e * k

    group_A = torch.randn(total_M, k, dtype=dtype, device=device) * scale
    group_B = torch.randn(total_K, n, dtype=dtype, device=device) * scale
    group_C = torch.randn(total_M, n, dtype=dtype, device=device) * scale

    offs_table = _build_offs_table(k, e, n, m_list)

    out_cublas = cublas_group_gemm_reference(
        group_A, group_B, group_C, offs_table, alpha, beta
    )

    out_flag = flag_blas.group_gemm(
        group_A, group_B, group_C, offs_table, alpha=alpha, beta=beta
    )

    torch.testing.assert_close(out_flag, out_cublas, rtol=1e-2, atol=1e-2)


@pytest.mark.group_gemm
def test_group_gemm_alpha_zero():
    m, k, e, n = 16, 64, 4, 128
    dtype, device = torch.bfloat16, flag_blas.device
    A = torch.randn(e * m, k, dtype=dtype, device=device)
    B = torch.randn(e * k, n, dtype=dtype, device=device)
    C = torch.randn(e * m, n, dtype=dtype, device=device)
    C_orig = C.clone()
    m_list = [m] * e
    offs_table = _build_offs_table(k, e, n, m_list)

    out = flag_blas.group_gemm(A, B, C, offs_table, alpha=0.0, beta=2.0)

    torch.testing.assert_close(out, C_orig * 2.0, rtol=1e-2, atol=1e-2)


@pytest.mark.group_gemm
def test_group_gemm_beta_zero():
    m, k, e, n = 8, 32, 3, 64
    dtype, device = torch.bfloat16, flag_blas.device
    A = torch.randn(e * m, k, dtype=dtype, device=device)
    B = torch.randn(e * k, n, dtype=dtype, device=device)
    C_zeros = torch.zeros(e * m, n, dtype=dtype, device=device)
    m_list = [m] * e
    offs_table = _build_offs_table(k, e, n, m_list)

    ref = _compute_reference(A, B, C_zeros, offs_table, 1.0, 0.0)
    out = flag_blas.group_gemm(A, B, C_zeros, offs_table, alpha=1.0, beta=0.0)

    assert not torch.isnan(out).any()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.group_gemm
@pytest.mark.parametrize(
    "alpha,beta", [(1.0, 0.0), (2.0, 0.0), (2.0, 0.5), (0.0, 1.0), (0.5, 1.5)]
)
def test_group_gemm_alpha_beta(alpha, beta):
    m, k, e, n = 32, 128, 2, 128
    dtype, device = torch.bfloat16, flag_blas.device
    scale = k**-0.5
    A = torch.randn(e * m, k, dtype=dtype, device=device) * scale
    B = torch.randn(e * k, n, dtype=dtype, device=device) * scale
    C = torch.randn(e * m, n, dtype=dtype, device=device) * scale
    m_list = [m] * e
    offs_table = _build_offs_table(k, e, n, m_list)

    ref = _compute_reference(A, B, C, offs_table, alpha, beta)
    out = flag_blas.group_gemm(A, B, C, offs_table, alpha=alpha, beta=beta)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
