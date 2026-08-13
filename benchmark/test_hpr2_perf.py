import ctypes
import ctypes.util
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER
from flag_blas.utils import shape_utils

HPR2_SIZES = [
    64,
    96,
    127,
    128,
    129,
    160,
    191,
    192,
    193,
    224,
    255,
    256,
    257,
    320,
    383,
    384,
    385,
    448,
    511,
    512,
    513,
    640,
    767,
    768,
    769,
    896,
    1023,
    1024,
    1025,
    1280,
    1535,
    1536,
    1537,
    1792,
    2047,
    2048,
    2049,
    2304,
    2559,
    2560,
    2561,
    2816,
    3071,
    3072,
    3073,
    3328,
    3583,
    3584,
    3585,
    3840,
    4095,
    4096,
    4607,
    4608,
    4609,
    5119,
    5120,
    5121,
    5632,
    6143,
    6144,
    6145,
    7167,
    7168,
    7169,
    8191,
    8192,
]


def load_cublas():
    lib_names = ["libcublas.so.13"]
    found_path = ctypes.util.find_library("cublas")
    if found_path:
        lib_names.append(found_path)
    lib_names.extend(["libcublas.so", "libcublas.so.12", "libcublas.so.11"])
    for name in lib_names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError("Unable to find libcublas.so on this system")


_cublas = load_cublas()
_cublas_handle = None
CUBLAS_POINTER_MODE_HOST = 0


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def _configure_cublas_signatures():
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    for name in ("cublasChpr2_v2", "cublasZhpr2_v2"):
        func = getattr(_cublas, name)
        func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        func.restype = ctypes.c_int


_configure_cublas_signatures()
_CUBLAS_HPR2_FUNCS = {
    torch.complex64: (_cublas.cublasChpr2_v2, cuComplex),
    torch.complex128: (_cublas.cublasZhpr2_v2, cuDoubleComplex),
}


def _get_cublas_handle():
    global _cublas_handle
    if _cublas_handle is not None:
        return _cublas_handle
    _cublas_handle = ctypes.c_void_p()
    status = _cublas.cublasCreate_v2(ctypes.byref(_cublas_handle))
    if status != 0:
        raise RuntimeError(f"cublasCreate_v2 failed with status code: {status}")
    status = _cublas.cublasSetPointerMode_v2(_cublas_handle, CUBLAS_POINTER_MODE_HOST)
    if status != 0:
        raise RuntimeError(f"cublasSetPointerMode_v2 failed with status code: {status}")
    return _cublas_handle


def _make_scalar(ctor, value):
    return ctor(value.real, value.imag)


def cublas_hpr2_baseline(
    AP, x, y, uplo, n, alpha, incx, incy, handle, c_func, alpha_c, **kwargs
):
    if n == 0:
        return AP
    status = c_func(
        handle,
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
        ctypes.c_void_p(AP.data_ptr()),
    )
    if status != 0:
        raise RuntimeError(f"cublasXhpr2_v2 failed with status code: {status}")
    torch.cuda.synchronize(AP.device)
    return AP


def gems_chpr2_wrapper(AP, x, y, uplo, n, alpha, incx, incy, handle, **kwargs):
    flag_blas.ops.chpr2(uplo, n, alpha, x, incx, y, incy, AP)
    return AP


def gems_zhpr2_wrapper(AP, x, y, uplo, n, alpha, incx, incy, handle, **kwargs):
    flag_blas.ops.zhpr2(uplo, n, alpha, x, incx, y, incy, AP)
    return AP


class Hpr2Benchmark(Benchmark):
    DEFAULT_SHAPES = [(n,) for n in HPR2_SIZES]
    DEFAULT_SHAPE_DESC = "N"

    def __init__(self, *args, uplo=CUBLAS_FILL_MODE_LOWER, alpha=1.5 + 0.5j, **kwargs):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.alpha = alpha

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in HPR2_SIZES]
        return None

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        max_n = max(HPR2_SIZES)
        if any(len(shape) != 1 or shape[0] > max_n for shape in self.shapes):
            self.shapes = list(self.DEFAULT_SHAPES)
            self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = _get_cublas_handle()
        c_func, ctor = _CUBLAS_HPR2_FUNCS[cur_dtype]
        alpha_c = _make_scalar(ctor, self.alpha)
        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            AP = torch.randn(n * (n + 1) // 2, dtype=cur_dtype, device=self.device)
            x = torch.randn(n, dtype=cur_dtype, device=self.device)
            y = torch.randn(n, dtype=cur_dtype, device=self.device)
            yield AP, x, y, {
                "uplo": self.uplo,
                "n": n,
                "alpha": self.alpha,
                "incx": 1,
                "incy": 1,
                "handle": handle,
                "c_func": c_func,
                "alpha_c": alpha_c,
            }

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        return 4 * n * n

    def get_gbps(self, args, latency):
        AP, x, y = args[0], args[1], args[2]
        io_amount = (
            2 * shape_utils.size_in_bytes(AP)
            + shape_utils.size_in_bytes(x)
            + shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return max(1, kwargs.get("n", 0))

    def clone_correctness_inputs(self, args, kwargs):
        AP, x, y = args
        return (AP.clone(), x, y), kwargs, (AP.clone(), x, y), kwargs


@pytest.mark.chpr2
def test_perf_chpr2():
    bench = Hpr2Benchmark(
        op_name="chpr2",
        torch_op=cublas_hpr2_baseline,
        gems_op=gems_chpr2_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.chpr2
def test_perf_chpr2_upper():
    bench = Hpr2Benchmark(
        op_name="chpr2_upper",
        torch_op=cublas_hpr2_baseline,
        gems_op=gems_chpr2_wrapper,
        dtypes=[torch.complex64],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zhpr2
def test_perf_zhpr2():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support complex128")
    bench = Hpr2Benchmark(
        op_name="zhpr2",
        torch_op=cublas_hpr2_baseline,
        gems_op=gems_zhpr2_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zhpr2
def test_perf_zhpr2_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support complex128")
    bench = Hpr2Benchmark(
        op_name="zhpr2_upper",
        torch_op=cublas_hpr2_baseline,
        gems_op=gems_zhpr2_wrapper,
        dtypes=[torch.complex128],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)
