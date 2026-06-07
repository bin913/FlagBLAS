from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.attri_util import L1_STRIDE_SHAPES, L1_VECTOR_SHAPES
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.utils import shape_utils

PAIR_STRIDES = [(2, 2), (2, 3), (3, 2), (3, 3)]
ROTM_FLAGS = [-2.0, -1.5, -1.0, -0.5, 0.0, 1.0]


def torch_rotm_reference(x, y, param, n=None, incx=1, incy=1):
    if n is None:
        n = min(x.numel() // incx, y.numel() // incy)
    if n <= 0:
        return x, y

    flag = float(param[0].item())
    h11, h21, h12, h22 = param[1], param[2], param[3], param[4]
    x_view = x[0 : n * incx : incx]
    y_view = y[0 : n * incy : incy]
    old_x = x_view.clone()
    old_y = y_view.clone()

    if flag == -2.0:
        return x, y
    if flag < 0.0:
        x_view.copy_(h11 * old_x + h12 * old_y)
        y_view.copy_(h21 * old_x + h22 * old_y)
    elif flag == 0.0:
        x_view.copy_(old_x + h12 * old_y)
        y_view.copy_(h21 * old_x + old_y)
    elif flag == 1.0:
        x_view.copy_(h11 * old_x + old_y)
        y_view.copy_(-old_x + h22 * old_y)
    else:
        raise ValueError(f"Invalid rotm flag: {flag}")
    return x, y


def gems_rotm_wrapper(x, y, param, n=None, incx=1, incy=1):
    if x.dtype == torch.float32:
        flag_blas.ops.srotm(n, x, incx, y, incy, param)
    elif x.dtype == torch.float64:
        flag_blas.ops.drotm(n, x, incx, y, incy, param)
    else:
        raise TypeError(f"Unsupported dtype for rotm: {x.dtype}")
    return x, y


class RotmBenchmark(Benchmark):
    correctness_reference = "Torch reference"

    def __init__(self, *args, incx=1, incy=1, flag=-1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = list(self.metrics)
        self.to_bench_metrics = list(self.to_bench_metrics)
        self.incx = incx
        self.incy = incy
        self.flag = flag

    def set_more_metrics(self):
        return ["gbps"]

    def set_shapes(self, shape_file_path=None):
        self.shapes = L1_VECTOR_SHAPES[:4]
        self.shape_desc = "N"

    def set_more_shapes(self):
        return L1_VECTOR_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            n = shape[0]
            x = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            y = torch.randn(n * self.incy, dtype=cur_dtype, device=self.device)
            param = torch.tensor(
                [self.flag, 0.8, 0.2, -0.3, 0.7],
                dtype=cur_dtype,
                device=self.device,
            )
            yield x, y, param, {"n": n, "incx": self.incx, "incy": self.incy}

    def get_gbps(self, args, latency):
        x = args[0]
        n = x.numel() // self.incx
        element_size = x.element_size()
        io_amount = 4 * n * element_size
        return io_amount * 1e-9 / (latency * 1e-3)


class RotmStrideBenchmark(RotmBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = L1_STRIDE_SHAPES[:4]
        self.shape_desc = "N"

    def set_more_shapes(self):
        return L1_STRIDE_SHAPES


@pytest.mark.rotm
@pytest.mark.parametrize("flag", ROTM_FLAGS)
def test_perf_srotm(flag):
    run_correctness_then_benchmark(
        RotmBenchmark(
            op_name=f"srotm_flag{flag}",
            torch_op=torch_rotm_reference,
            gems_op=gems_rotm_wrapper,
            dtypes=[torch.float32],
            flag=flag,
        )
    )


@pytest.mark.rotm
@pytest.mark.parametrize("flag", ROTM_FLAGS)
def test_perf_drotm(flag):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    run_correctness_then_benchmark(
        RotmBenchmark(
            op_name=f"drotm_flag{flag}",
            torch_op=torch_rotm_reference,
            gems_op=gems_rotm_wrapper,
            dtypes=[torch.float64],
            flag=flag,
        )
    )


@pytest.mark.rotm
@pytest.mark.parametrize("incx,incy", PAIR_STRIDES)
def test_perf_srotm_stride(incx, incy):
    run_correctness_then_benchmark(
        RotmStrideBenchmark(
            op_name=f"srotm_stride_incx{incx}_incy{incy}",
            torch_op=torch_rotm_reference,
            gems_op=gems_rotm_wrapper,
            dtypes=[torch.float32],
            incx=incx,
            incy=incy,
        )
    )


@pytest.mark.rotm
@pytest.mark.parametrize("incx,incy", PAIR_STRIDES)
def test_perf_drotm_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    run_correctness_then_benchmark(
        RotmStrideBenchmark(
            op_name=f"drotm_stride_incx{incx}_incy{incy}",
            torch_op=torch_rotm_reference,
            gems_op=gems_rotm_wrapper,
            dtypes=[torch.float64],
            incx=incx,
            incy=incy,
        )
    )
