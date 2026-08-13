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

import logging
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry
from flag_blas.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

ScalarType = Union[float, int, complex, torch.Tensor]

_BLOCK_SIZE = 1024


@libentry()
@triton.jit
def saxpy_kernel(
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    n,
    INCX,
    INCY,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    x = tl.load(x_ptr + idx * INCX, mask=mask, other=0.0)
    y = tl.load(y_ptr + idx * INCY, mask=mask, other=0.0)

    tl.store(y_ptr + idx * INCY, alpha * x + y, mask=mask)


def saxpy(
    n: int, alpha: ScalarType, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int
) -> None:
    """Minimal Ascend NPU demo operator: y = alpha * x + y (float32)."""
    logger.debug("FLAG_BLAS ASCEND SAXPY")

    assert x.device == y.device, "x and y must be on the same device"
    assert x.dtype == torch.float32, "x must be float32"
    assert y.dtype == torch.float32, "y must be float32"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert incx > 0 and incy > 0, "incx and incy must be positive"

    if n <= 0:
        return

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)

    req_size_x = 1 + (n - 1) * incx
    req_size_y = 1 + (n - 1) * incy
    assert x.numel() >= req_size_x, "x is too short"
    assert y.numel() >= req_size_y, "y is too short"

    grid = (triton.cdiv(n, _BLOCK_SIZE),)

    with torch_device_fn.device(x.device):
        saxpy_kernel[grid](x, y, alpha, n, incx, incy, _BLOCK_SIZE)
