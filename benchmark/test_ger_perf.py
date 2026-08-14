from typing import Generator

import cupy as cp
import numpy as np
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.utils import shape_utils

GER_BENCH_OPS = {
    "sger": (torch.float32, cublas.sger, np.float32, 1e-5),
    "dger": (torch.float64, cublas.dger, np.float64, 1e-5),
    "cgeru": (torch.complex64, cublas.cgeru, np.complex64, 1e-5 + 2e-5j),
    "cgerc": (torch.complex64, cublas.cgerc, np.complex64, 1e-5 + 2e-5j),
    "zgeru": (torch.complex128, cublas.zgeru, np.complex128, 1e-5 + 2e-5j),
    "zgerc": (torch.complex128, cublas.zgerc, np.complex128, 1e-5 + 2e-5j),
}


def cublas_ger(
    A,
    x,
    y,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    incy,
    handle,
    alpha_ptr,
    op_name,
):
    func = GER_BENCH_OPS[op_name][1]
    func(
        handle,
        m,
        n,
        alpha_ptr,
        x.data_ptr(),
        incx,
        y.data_ptr(),
        incy,
        A.data_ptr(),
        lda_col,
    )
    return A


def flag_blas_ger_wrapper(
    A,
    x,
    y,
    m,
    n,
    alpha,
    A_row,
    lda_col,
    lda_row,
    incx,
    incy,
    handle,
    alpha_ptr,
    op_name,
):
    getattr(flag_blas.ops, op_name)(m, n, alpha, x, incx, y, incy, A_row, lda_row)
    return A_row


class GerBenchmark(Benchmark):
    def __init__(self, *args, ger_op_name, incx=1, incy=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ger_op_name = ger_op_name
        self.alpha = GER_BENCH_OPS[ger_op_name][3]
        self.incx = incx
        self.incy = incy
        if "gbps" not in self.metrics:
            self.metrics.append("gbps")

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        self.shapes = [
            (64, 64),
            (256, 256),
            (1024, 1024),
            (4096, 4096),
            (1024, 4096),
            (4096, 1024),
            (127, 255),
            (1023, 4095),
            (4095, 1023),
        ]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
        _, _, np_dtype, _ = GER_BENCH_OPS[self.ger_op_name]
        alpha_np = np.array(self.alpha, dtype=np_dtype)
        alpha_ptr = alpha_np.ctypes.data

        for m, n in self.shapes:
            A_col = torch.randn(n, m, dtype=cur_dtype, device=self.device).t()
            A_row = A_col.contiguous()
            x = torch.randn(m * self.incx, dtype=cur_dtype, device=self.device)
            y = torch.randn(n * self.incy, dtype=cur_dtype, device=self.device)

            yield A_col, x, y, {
                "m": m,
                "n": n,
                "alpha": self.alpha,
                "A_row": A_row,
                "lda_col": m,
                "lda_row": n,
                "incx": self.incx,
                "incy": self.incy,
                "handle": handle,
                "alpha_ptr": alpha_ptr,
                "op_name": self.ger_op_name,
            }

    def get_gbps(self, args, latency):
        A, x, y = args[0], args[1], args[2]
        m, n = A.shape
        element_size = A.element_size()
        io_amount = 2 * m * n * element_size + m * element_size + n * element_size
        if self.incx != 1:
            io_amount += shape_utils.size_in_bytes(x) - m * element_size
        if self.incy != 1:
            io_amount += shape_utils.size_in_bytes(y) - n * element_size
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return kwargs["n"]

    def clone_correctness_inputs(self, args, kwargs):
        A, x, y = args
        A_col_ref = A.t().contiguous().t()
        A_row_blas = A_col_ref.contiguous()

        ref_kwargs = dict(kwargs)
        ref_kwargs["A_row"] = A_row_blas.clone()

        blas_kwargs = dict(kwargs)
        blas_kwargs["A_row"] = A_row_blas

        return (
            (A_col_ref, x.clone(), y.clone()),
            ref_kwargs,
            (A.clone(), x.clone(), y.clone()),
            blas_kwargs,
        )


def _run_ger_benchmark(op_name):
    dtype = GER_BENCH_OPS[op_name][0]
    if dtype in (torch.float64, torch.complex128):
        if not flag_blas.runtime.device.support_fp64:
            pytest.skip("Device does not support float64")

    bench = GerBenchmark(
        op_name=op_name,
        torch_op=cublas_ger,
        gems_op=flag_blas_ger_wrapper,
        dtypes=[dtype],
        ger_op_name=op_name,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ger
def test_perf_sger():
    _run_ger_benchmark("sger")


@pytest.mark.ger
def test_perf_dger():
    _run_ger_benchmark("dger")


@pytest.mark.ger
def test_perf_cgeru():
    _run_ger_benchmark("cgeru")


@pytest.mark.ger
def test_perf_cgerc():
    _run_ger_benchmark("cgerc")


@pytest.mark.ger
def test_perf_zgeru():
    _run_ger_benchmark("zgeru")


@pytest.mark.ger
def test_perf_zgerc():
    _run_ger_benchmark("zgerc")
