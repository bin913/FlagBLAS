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

import pytest
import torch

from benchmark.gemm_perf_common import (
    Fp8GemmBenchmark,
    cublas_fp8gemm_baseline,
    gems_fp8gemm_wrapper,
)
from flag_blas.ops import CUBLAS_OP_N, CUBLAS_OP_T

@pytest.mark.fp8gemm
def test_perf_fp8gemm_nn():
    bench = Fp8GemmBenchmark(
        op_name="fp8gemm",
        torch_op=cublas_fp8gemm_baseline,
        gems_op=gems_fp8gemm_wrapper,
        dtypes=[torch.float8_e4m3fn],
        transa=CUBLAS_OP_N,
        transb=CUBLAS_OP_N,
    )
    # bench.run()


@pytest.mark.fp8gemm
def test_perf_fp8gemm_tn():
    bench = Fp8GemmBenchmark(
        op_name="fp8gemm_tn",
        torch_op=cublas_fp8gemm_baseline,
        gems_op=gems_fp8gemm_wrapper,
        dtypes=[torch.float8_e4m3fn],
        transa=CUBLAS_OP_T,
        transb=CUBLAS_OP_N,
    )
    # bench.run()


@pytest.mark.fp8gemm
def test_perf_fp8gemm_nt():
    bench = Fp8GemmBenchmark(
        op_name="fp8gemm_nt",
        torch_op=cublas_fp8gemm_baseline,
        gems_op=gems_fp8gemm_wrapper,
        dtypes=[torch.float8_e4m3fn],
        transa=CUBLAS_OP_N,
        transb=CUBLAS_OP_T,
    )
    # bench.run()


@pytest.mark.fp8gemm
def test_perf_fp8gemm_tt():
    bench = Fp8GemmBenchmark(
        op_name="fp8gemm_tt",
        torch_op=cublas_fp8gemm_baseline,
        gems_op=gems_fp8gemm_wrapper,
        dtypes=[torch.float8_e4m3fn],
        transa=CUBLAS_OP_T,
        transb=CUBLAS_OP_T,
    )
    # bench.run()
