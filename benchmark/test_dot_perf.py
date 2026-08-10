from typing import Generator

import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from tests.accuracy_utils import DEFAULT_SHAPES

PAIR_STRIDES = [(2, 2), (2, 3), (3, 2), (3, 3)]


def cublas_dot(x, y, result, n=None, incx=1, incy=1, handle=None):
    if n is None:
        n = min(x.numel() // incx, y.numel() // incy)
    if n <= 0:
        result.zero_()
        return result
    if x.dtype == torch.float32:
        cublas.sdot(
            handle, n, x.data_ptr(), incx, y.data_ptr(), incy, result.data_ptr()
        )
    elif x.dtype == torch.float64:
        cublas.ddot(
            handle, n, x.data_ptr(), incx, y.data_ptr(), incy, result.data_ptr()
        )
    else:
        raise TypeError(f"Unsupported dtype for dot: {x.dtype}")
    return result


def gems_sdot_wrapper(x, y, result, n=None, incx=1, incy=1, handle=None):
    flag_blas.ops.sdot(n, x, incx, y, incy, result)
    return result


def gems_ddot_wrapper(x, y, result, n=None, incx=1, incy=1, handle=None):
    flag_blas.ops.ddot(n, x, incx, y, incy, result)
    return result


class DotBenchmark(Benchmark):
    def __init__(self, *args, incx=1, incy=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx
        self.incy = incy

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        self.shapes = DEFAULT_SHAPES
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
        for shape in self.shapes:
            n = shape[0]
            x = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            y = torch.randn(n * self.incy, dtype=cur_dtype, device=self.device)
            result = torch.zeros(1, dtype=cur_dtype, device=self.device)
            yield x, y, result, {
                "n": n,
                "incx": self.incx,
                "incy": self.incy,
                "handle": handle,
            }

    def get_gbps(self, args, latency):
        x, y = args[0], args[1]
        n = min(x.numel() // self.incx, y.numel() // self.incy)
        io_amount = 2 * n * x.element_size()
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return kwargs["n"]


@pytest.mark.dot
def test_perf_sdot():
    bench = DotBenchmark(
        op_name="sdot",
        torch_op=cublas_dot,
        gems_op=gems_sdot_wrapper,
        dtypes=[torch.float32],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dot
def test_perf_ddot():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = DotBenchmark(
        op_name="ddot",
        torch_op=cublas_dot,
        gems_op=gems_ddot_wrapper,
        dtypes=[torch.float64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dot
@pytest.mark.parametrize("incx,incy", PAIR_STRIDES)
def test_perf_sdot_stride(incx, incy):
    bench = DotBenchmark(
        op_name=f"sdot_stride_incx{incx}_incy{incy}",
        torch_op=cublas_dot,
        gems_op=gems_sdot_wrapper,
        dtypes=[torch.float32],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dot
@pytest.mark.parametrize("incx,incy", PAIR_STRIDES)
def test_perf_ddot_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = DotBenchmark(
        op_name=f"ddot_stride_incx{incx}_incy{incy}",
        torch_op=cublas_dot,
        gems_op=gems_ddot_wrapper,
        dtypes=[torch.float64],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)
