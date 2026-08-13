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
from scipy.linalg import blas as cpu_blas

import flag_blas

from .accuracy_utils import SCALARS, blas_assert_close, to_cpu_blas_tensor

# Keep shapes small so this doubles as a smoke test for the Ascend NPU backend.
_ASCEND_AXPY_SHAPES = [(1024,), (5333,), (65536,)]


@pytest.fixture(autouse=True)
def _require_ascend_cpu_ref(request):
    if flag_blas.vendor_name != "ascend":
        pytest.skip("test_ascend_axpy only runs on the Ascend NPU backend")
    if request.config.getoption("--ref") != "cpu":
        pytest.skip("test_ascend_axpy only supports --ref cpu")


def _cpu_axpy_reference(n, alpha, x, incx, y, incy):
    if n == 0:
        return to_cpu_blas_tensor(y)

    # to_cpu_blas_tensor upcasts float32 -> float64 for the scipy BLAS reference.
    ref_x = to_cpu_blas_tensor(x)
    ref_y = to_cpu_blas_tensor(y)
    cpu_blas.daxpy(ref_x.numpy(), ref_y.numpy(), n=n, a=alpha, incx=incx, incy=incy)
    return ref_y


@pytest.mark.ascend
@pytest.mark.axpy
@pytest.mark.parametrize("shape", _ASCEND_AXPY_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
def test_ascend_axpy_saxpy(shape, alpha):
    n = shape[0]
    incx = 1
    incy = 1
    x = torch.randn(n, dtype=torch.float32, device=flag_blas.device)
    y = torch.randn(n, dtype=torch.float32, device=flag_blas.device)

    ref_y = _cpu_axpy_reference(n, alpha, x, incx, y, incy)

    flag_blas.ops.saxpy(n, alpha, x, incx, y, incy)

    blas_assert_close(y, ref_y, torch.float32)
