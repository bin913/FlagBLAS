import ctypes
import ctypes.util
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER
from flag_blas.utils import shape_utils

SPR_SIZES = [
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
    9215,
    9216,
    9217,
    10239,
    10240,
    10241,
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


_cublas = None
_CUBLAS_SPR_FUNCS = None
CUBLAS_POINTER_MODE_HOST = 0


def _configure_cublas_signatures():
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    for name in ("cublasSspr_v2", "cublasDspr_v2"):
        func = getattr(_cublas, name)
        func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        func.restype = ctypes.c_int


def _ensure_cublas():
    global _cublas, _CUBLAS_SPR_FUNCS
    if _cublas is None:
        _cublas = load_cublas()
        _configure_cublas_signatures()
        _CUBLAS_SPR_FUNCS = {
            torch.float32: (_cublas.cublasSspr_v2, ctypes.c_float),
            torch.float64: (_cublas.cublasDspr_v2, ctypes.c_double),
        }
    return _cublas


def cublas_spr_baseline(AP, x, uplo, n, incx, handle, c_func, alpha_c, **kwargs):
    if n == 0:
        return AP
    status = c_func(
        handle,
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.c_void_p(AP.data_ptr()),
    )
    if status != 0:
        raise RuntimeError(f"cublasXspr_v2 failed with status code: {status}")
    torch.cuda.synchronize(AP.device)
    return AP


def gems_sspr_wrapper(AP, x, uplo, n, alpha, incx, handle, **kwargs):
    flag_blas.ops.sspr(uplo, n, alpha, x, incx, AP)
    return AP


def gems_dspr_wrapper(AP, x, uplo, n, alpha, incx, handle, **kwargs):
    flag_blas.ops.dspr(uplo, n, alpha, x, incx, AP)
    return AP


def _generate_packed(n, dtype, device):
    return torch.randn(n * (n + 1) // 2, dtype=dtype, device=device)


class SprBenchmark(Benchmark):
    DEFAULT_SHAPES = [(n,) for n in SPR_SIZES]
    DEFAULT_SHAPE_DESC = "N"

    def __init__(self, *args, uplo=CUBLAS_FILL_MODE_LOWER, alpha=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.alpha = alpha

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in SPR_SIZES]
        return None

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        max_n = max(SPR_SIZES)
        if any(len(shape) != 1 or shape[0] > max_n for shape in self.shapes):
            self.shapes = list(self.DEFAULT_SHAPES)
            self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype) -> Generator:
        cublas = _ensure_cublas()
        handle = ctypes.c_void_p()
        status = cublas.cublasCreate_v2(ctypes.byref(handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status code: {status}")
        status = cublas.cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST)
        if status != 0:
            raise RuntimeError(
                f"cublasSetPointerMode_v2 failed with status code: {status}"
            )
        c_func, ctor = _CUBLAS_SPR_FUNCS[cur_dtype]
        alpha_c = ctor(self.alpha)
        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            AP = _generate_packed(n, cur_dtype, self.device)
            x = torch.randn(n, dtype=cur_dtype, device=self.device)
            yield AP, x, {
                "uplo": self.uplo,
                "n": n,
                "alpha": self.alpha,
                "incx": 1,
                "handle": handle,
                "c_func": c_func,
                "alpha_c": alpha_c,
            }

    def get_tflops(self, op, *args, **kwargs):
        return kwargs.get("n", 0) * kwargs.get("n", 0)

    def get_gbps(self, args, latency):
        AP, x = args[0], args[1]
        io_amount = 2 * shape_utils.size_in_bytes(AP) + shape_utils.size_in_bytes(x)
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return max(1, kwargs.get("n", 0))

    def clone_correctness_inputs(self, args, kwargs):
        AP, x = args
        return (AP.clone(), x), kwargs, (AP.clone(), x), kwargs


@pytest.mark.sspr
def test_perf_sspr():
    bench = SprBenchmark(
        op_name="sspr",
        torch_op=cublas_spr_baseline,
        gems_op=gems_sspr_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.sspr
def test_perf_sspr_upper():
    bench = SprBenchmark(
        op_name="sspr_upper",
        torch_op=cublas_spr_baseline,
        gems_op=gems_sspr_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dspr
def test_perf_dspr():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = SprBenchmark(
        op_name="dspr",
        torch_op=cublas_spr_baseline,
        gems_op=gems_dspr_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dspr
def test_perf_dspr_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = SprBenchmark(
        op_name="dspr_upper",
        torch_op=cublas_spr_baseline,
        gems_op=gems_dspr_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)
