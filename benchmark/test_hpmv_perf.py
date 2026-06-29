import ctypes
import ctypes.util
from typing import Generator

import cupy as cp
import pytest
import torch
from cupy_backends.cuda.libs import cublas

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER
from flag_blas.utils import shape_utils

HPMV_SIZES = [
    256,
    512,
    1024,
    2048,
    4096,
    6144,
    8192,
    12288,
    16384,
]


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


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


_CUBLAS_HPMV_FUNCS = {
    torch.complex64: (_cublas.cublasChpmv_v2, cuComplex),
    torch.complex128: (_cublas.cublasZhpmv_v2, cuDoubleComplex),
}


def _make_scalar(ctor, value):
    return ctor(value.real, value.imag)


def cublas_hpmv_baseline(
    AP,
    x,
    y,
    uplo,
    n,
    alpha,
    incx,
    beta,
    incy,
    handle,
    c_func,
    alpha_c,
    beta_c,
    **kwargs,
):
    if n == 0:
        return y
    status = c_func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(AP.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.byref(beta_c),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
    )
    if status != 0:
        raise RuntimeError(f"cublasXhpmv_v2 failed with status code: {status}")
    return y


def _gems_wrapper(op):
    def _impl(AP, x, y, uplo, n, alpha, incx, beta, incy, handle, **kwargs):
        op(uplo, n, alpha, AP, x, incx, beta, y, incy)
        return y

    return _impl


gems_chpmv_wrapper = _gems_wrapper(flag_blas.ops.chpmv)
gems_zhpmv_wrapper = _gems_wrapper(flag_blas.ops.zhpmv)


def _generate_packed_her(n, dtype, device):
    return torch.randn(n * (n + 1) // 2, dtype=dtype, device=device)


class HpmvBenchmark(Benchmark):
    def __init__(
        self,
        *args,
        uplo=CUBLAS_FILL_MODE_LOWER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.alpha = alpha
        self.beta = beta

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in HPMV_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

        if cur_dtype not in _CUBLAS_HPMV_FUNCS:
            raise ValueError(f"Unsupported dtype: {cur_dtype}")
        c_func, ctor = _CUBLAS_HPMV_FUNCS[cur_dtype]
        alpha_c = _make_scalar(ctor, self.alpha)
        beta_c = _make_scalar(ctor, self.beta)

        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            AP = _generate_packed_her(n, cur_dtype, self.device)
            x = torch.randn(n, dtype=cur_dtype, device=self.device)
            y = torch.randn(n, dtype=cur_dtype, device=self.device)

            yield AP, x, y, {
                "uplo": self.uplo,
                "n": n,
                "alpha": self.alpha,
                "incx": 1,
                "beta": self.beta,
                "incy": 1,
                "handle": handle,
                "c_func": c_func,
                "alpha_c": alpha_c,
                "beta_c": beta_c,
            }

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        return 8 * n * n

    def get_gbps(self, args, latency):
        AP, x, y = args[0], args[1], args[2]
        a_bytes = AP.numel() * AP.element_size()
        io_amount = (
            a_bytes + shape_utils.size_in_bytes(x) + 2 * shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return max(1, kwargs.get("n", 0))

    def clone_correctness_inputs(self, args, kwargs):
        AP, x, y = args
        ref_args = (AP, x, y.clone())
        blas_args = (AP, x, y.clone())
        return ref_args, kwargs, blas_args, kwargs


@pytest.mark.chpmv
def test_perf_chpmv():
    bench = HpmvBenchmark(
        op_name="chpmv",
        torch_op=cublas_hpmv_baseline,
        gems_op=gems_chpmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.chpmv
def test_perf_chpmv_upper():
    bench = HpmvBenchmark(
        op_name="chpmv_upper",
        torch_op=cublas_hpmv_baseline,
        gems_op=gems_chpmv_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_UPPER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zhpmv
def test_perf_zhpmv():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = HpmvBenchmark(
        op_name="zhpmv",
        torch_op=cublas_hpmv_baseline,
        gems_op=gems_zhpmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zhpmv
def test_perf_zhpmv_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = HpmvBenchmark(
        op_name="zhpmv_upper",
        torch_op=cublas_hpmv_baseline,
        gems_op=gems_zhpmv_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_UPPER,
        alpha=1.5 + 0.5j,
        beta=0.5 + 0.25j,
    )
    run_correctness_then_benchmark(bench)
