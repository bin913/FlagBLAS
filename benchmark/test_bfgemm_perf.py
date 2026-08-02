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
    GemmBenchmark,
    cublas_bfgemm,
    cublas_bfgemm_reference,
    gems_bfgemm_wrapper,
)
from flag_blas.ops import CUBLAS_OP_N, CUBLAS_OP_T


@pytest.mark.bfgemm
def test_perf_bfgemm_nn():
    bench = GemmBenchmark(
        op_name="bfgemm",
        torch_op=cublas_bfgemm,
        gems_op=gems_bfgemm_wrapper,
        dtypes=[torch.bfloat16],
        transa=CUBLAS_OP_N,
        transb=CUBLAS_OP_N,
    )
    bench.init_user_config()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_bfgemm_reference(A, B, C.clone(), **kwargs)
            gems_result = gems_bfgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1e-4)
    bench.run()


@pytest.mark.bfgemm
def test_perf_bfgemm_tn():
    bench = GemmBenchmark(
        op_name="bfgemm_tn",
        torch_op=cublas_bfgemm,
        gems_op=gems_bfgemm_wrapper,
        dtypes=[torch.bfloat16],
        transa=CUBLAS_OP_T,
        transb=CUBLAS_OP_N,
    )
    bench.init_user_config()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_bfgemm_reference(A, B, C.clone(), **kwargs)
            gems_result = gems_bfgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1e-4)
    bench.run()


@pytest.mark.bfgemm
def test_perf_bfgemm_nt():
    bench = GemmBenchmark(
        op_name="bfgemm_nt",
        torch_op=cublas_bfgemm,
        gems_op=gems_bfgemm_wrapper,
        dtypes=[torch.bfloat16],
        transa=CUBLAS_OP_N,
        transb=CUBLAS_OP_T,
    )
    bench.init_user_config()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_bfgemm_reference(A, B, C.clone(), **kwargs)
            gems_result = gems_bfgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1e-3)
    bench.run()


@pytest.mark.bfgemm
def test_perf_bfgemm_tt():
    bench = GemmBenchmark(
        op_name="bfgemm_tt",
        torch_op=cublas_bfgemm,
        gems_op=gems_bfgemm_wrapper,
        dtypes=[torch.bfloat16],
        transa=CUBLAS_OP_T,
        transb=CUBLAS_OP_T,
    )
    bench.init_user_config()
    for cur_dtype in bench.dtypes:
        for A, B, C, kwargs in bench.get_input_iter(cur_dtype):
            torch_result = cublas_bfgemm_reference(A, B, C.clone(), **kwargs)
            gems_result = gems_bfgemm_wrapper(A, B, C.clone(), **kwargs)
            k = kwargs.get("k", 0)
            bench.validate_results(torch_result, gems_result, k, tolerance=1e-3)
    bench.run()
