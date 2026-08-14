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

import ctypes
import ctypes.util
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.ops import (
    CUBLAS_DIAG_NON_UNIT,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
)
from flag_blas.utils import shape_utils

STBSV_SIZES = [
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    12288,
    16384,
]

STBSV_KS = [1, 4, 16, 64, 256]


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
    raise RuntimeError("Unable to find libcublas.so on this system")


_cublas = load_cublas()
_cublas_handle = None


def _get_cublas_handle():
    global _cublas_handle
    if _cublas_handle is None:
        handle = ctypes.c_void_p()
        status = _cublas.cublasCreate_v2(ctypes.byref(handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status code: {status}")
        _cublas_handle = handle.value
    status = _cublas.cublasSetStream_v2(
        ctypes.c_void_p(_cublas_handle),
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )
    if status != 0:
        raise RuntimeError(f"cublasSetStream_v2 failed with status code: {status}")
    return _cublas_handle


def cublas_stbsv_baseline(
    A,
    x,
    uplo,
    trans,
    diag,
    n,
    k,
    lda,
    incx,
    handle,
    c_func,
    **kwargs,
):
    if n == 0:
        return x
    status = c_func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(trans),
        ctypes.c_int(diag),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
    )
    if status != 0:
        raise RuntimeError(f"cublasStbsv_v2 failed with status code: {status}")
    return x


def gems_stbsv_wrapper(A, x, uplo, trans, diag, n, k, lda, incx, **kwargs):
    flag_blas.stbsv(uplo, trans, diag, n, k, A, lda, x, incx)
    return x


def _gems_wrapper(op):
    def _impl(A, x, uplo, trans, diag, n, k, lda, incx, **kwargs):
        op(uplo, trans, diag, n, k, A, lda, x, incx)
        return x

    return _impl


GEMS_TBSV_WRAPPERS = {
    "stbsv": gems_stbsv_wrapper,
    "dtbsv": _gems_wrapper(flag_blas.dtbsv),
    "ctbsv": _gems_wrapper(flag_blas.ctbsv),
    "ztbsv": _gems_wrapper(flag_blas.ztbsv),
}


def _make_triangular_banded(n, k, lda, uplo, dtype, device):
    if n == 0:
        return torch.zeros((n, lda), dtype=dtype, device=device).contiguous()

    A = torch.randn((n, lda), dtype=dtype, device=device) * 0.1
    diag_floor = 2.0 * (k + 1) + 1.0
    cols = torch.arange(lda, device=device).view(1, lda)
    j = torch.arange(n, device=device).view(n, 1)

    if uplo == CUBLAS_FILL_MODE_UPPER:
        valid = (cols >= torch.clamp(k - j, min=0)) & (cols <= k)
        diag_col = k
    else:
        valid = cols <= torch.clamp(n - 1 - j, max=k)
        diag_col = 0

    A = A.masked_fill(~valid, 0.0)
    A[:, diag_col] = diag_floor
    return A.contiguous()


def _stored_band_nnz(n, k):
    if n <= 0:
        return 0
    if k >= n - 1:
        return n * (n + 1) // 2
    return (k + 1) * (k + 2) // 2 + (n - k - 1) * (k + 1)


class StbsvBenchmark(Benchmark):
    def __init__(
        self,
        *args,
        uplo=CUBLAS_FILL_MODE_LOWER,
        trans=CUBLAS_OP_N,
        diag=CUBLAS_DIAG_NON_UNIT,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.trans = trans
        self.diag = diag
        self.ks = STBSV_KS

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in STBSV_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = _get_cublas_handle()

        if cur_dtype == torch.float32:
            c_func = _cublas.cublasStbsv_v2
        elif cur_dtype == torch.float64:
            c_func = _cublas.cublasDtbsv_v2
        elif cur_dtype == torch.complex64:
            c_func = _cublas.cublasCtbsv_v2
        elif cur_dtype == torch.complex128:
            c_func = _cublas.cublasZtbsv_v2
        else:
            raise ValueError(f"Unsupported dtype: {cur_dtype}")

        seen = set()
        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            for k_req in self.ks:
                k = min(k_req, max(0, n - 1))
                key = (n, k)
                if key in seen:
                    continue
                seen.add(key)
                lda = k + 1
                A = _make_triangular_banded(
                    n, k, lda, self.uplo, cur_dtype, self.device
                )
                x = torch.randn(n, dtype=cur_dtype, device=self.device)

                yield A, x.clone(), {
                    "uplo": self.uplo,
                    "trans": self.trans,
                    "diag": self.diag,
                    "n": n,
                    "k": k,
                    "lda": lda,
                    "incx": 1,
                    "handle": handle,
                    "c_func": c_func,
                }

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        k = kwargs.get("k", 0)
        # ~2 flops per stored band element (1 mul + 1 add) for the
        # off-diagonal updates, plus ~n divisions for the diagonal.
        # The off-diagonals dominate.
        nnz = _stored_band_nnz(n, k)
        return 2 * nnz

    def get_gbps(self, args, latency):
        A, x = args[0], args[1]
        n = x.numel()
        k = A.shape[-1] - 1
        stored = _stored_band_nnz(n, k)
        a_bytes = stored * A.element_size()
        # x is read and written exactly once for each unknown.
        io_amount = a_bytes + 2 * shape_utils.size_in_bytes(x)
        return io_amount * 1e-9 / (latency * 1e-3)


# --------------------------------------------------------------------------
# Top-level perf entry points
# --------------------------------------------------------------------------
def _run_tbsv_variant(op_name, dtype, uplo, trans):
    if (
        dtype in (torch.float64, torch.complex128)
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("fp64 is not supported on this device")
    bench = StbsvBenchmark(
        op_name=f"{op_name}_{uplo}_{trans}",
        torch_op=cublas_stbsv_baseline,
        gems_op=GEMS_TBSV_WRAPPERS[op_name],
        dtypes=[dtype],
        uplo=uplo,
        trans=trans,
        diag=CUBLAS_DIAG_NON_UNIT,
    )
    run_correctness_then_benchmark(bench)


TBSV_PERF_CASES = [
    pytest.param(
        "stbsv",
        torch.float32,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        marks=pytest.mark.stbsv,
    ),
    pytest.param(
        "stbsv",
        torch.float32,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        marks=pytest.mark.stbsv,
    ),
    pytest.param(
        "stbsv",
        torch.float32,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        marks=pytest.mark.stbsv,
    ),
    pytest.param(
        "stbsv",
        torch.float32,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        marks=pytest.mark.stbsv,
    ),
    pytest.param(
        "dtbsv",
        torch.float64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        marks=pytest.mark.dtbsv,
    ),
    pytest.param(
        "dtbsv",
        torch.float64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        marks=pytest.mark.dtbsv,
    ),
    pytest.param(
        "dtbsv",
        torch.float64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        marks=pytest.mark.dtbsv,
    ),
    pytest.param(
        "dtbsv",
        torch.float64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        marks=pytest.mark.dtbsv,
    ),
    pytest.param(
        "ctbsv",
        torch.complex64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        marks=pytest.mark.ctbsv,
    ),
    pytest.param(
        "ctbsv",
        torch.complex64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        marks=pytest.mark.ctbsv,
    ),
    pytest.param(
        "ctbsv",
        torch.complex64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        marks=pytest.mark.ctbsv,
    ),
    pytest.param(
        "ctbsv",
        torch.complex64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        marks=pytest.mark.ctbsv,
    ),
    pytest.param(
        "ctbsv",
        torch.complex64,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_C,
        marks=pytest.mark.ctbsv,
    ),
    pytest.param(
        "ctbsv",
        torch.complex64,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_C,
        marks=pytest.mark.ctbsv,
    ),
    pytest.param(
        "ztbsv",
        torch.complex128,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        marks=pytest.mark.ztbsv,
    ),
    pytest.param(
        "ztbsv",
        torch.complex128,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        marks=pytest.mark.ztbsv,
    ),
    pytest.param(
        "ztbsv",
        torch.complex128,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        marks=pytest.mark.ztbsv,
    ),
    pytest.param(
        "ztbsv",
        torch.complex128,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T,
        marks=pytest.mark.ztbsv,
    ),
    pytest.param(
        "ztbsv",
        torch.complex128,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_C,
        marks=pytest.mark.ztbsv,
    ),
    pytest.param(
        "ztbsv",
        torch.complex128,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_C,
        marks=pytest.mark.ztbsv,
    ),
]


@pytest.mark.parametrize("op_name,dtype,uplo,trans", TBSV_PERF_CASES)
def test_perf_tbsv(op_name, dtype, uplo, trans):
    _run_tbsv_variant(op_name, dtype, uplo, trans)
