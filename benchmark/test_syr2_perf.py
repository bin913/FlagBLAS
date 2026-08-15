import ctypes
import ctypes.util
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER
from flag_blas.utils import shape_utils

SYR2_SIZES = [
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


_cublas = None
_CUBLAS_SYR2_FUNCS = None
CUBLAS_POINTER_MODE_HOST = 0


def _configure_cublas_signatures():
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    for name in ("cublasSsyr2_v2", "cublasDsyr2_v2"):
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
            ctypes.c_int,
        ]
        func.restype = ctypes.c_int


def _ensure_cublas():
    global _cublas, _CUBLAS_SYR2_FUNCS
    if _cublas is None:
        _cublas = load_cublas()
        _configure_cublas_signatures()
        _CUBLAS_SYR2_FUNCS = {
            torch.float32: (_cublas.cublasSsyr2_v2, ctypes.c_float, False),
            torch.float64: (_cublas.cublasDsyr2_v2, ctypes.c_double, False),
        }
    return _cublas


def _make_scalar(ctor, is_complex, value):
    if is_complex:
        return ctor(value.real, value.imag)
    return ctor(value)


def cublas_syr2_baseline(
    A, x, y, uplo, n, incx, incy, lda, handle, c_func, alpha_c, **kwargs
):
    if n == 0:
        return A

    status = c_func(
        handle,
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(incy),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
    )
    if status != 0:
        raise RuntimeError(f"cublasXsyr2_v2 failed with status code: {status}")
    torch.cuda.synchronize(A.device)
    return A


def gems_ssyr2_wrapper(A, x, y, uplo, n, alpha, incx, incy, lda, handle, **kwargs):
    flag_blas.ops.ssyr2(uplo, n, alpha, x, incx, y, incy, A, lda)
    return A


def gems_dsyr2_wrapper(A, x, y, uplo, n, alpha, incx, incy, lda, handle, **kwargs):
    flag_blas.ops.dsyr2(uplo, n, alpha, x, incx, y, incy, A, lda)
    return A


def _generate_syr2_A(n, lda, dtype, device):
    A = torch.zeros((n, lda), dtype=dtype, device=device)
    if dtype.is_complex:
        A[:, :n] = torch.randn(n, n, dtype=dtype, device=device)
    else:
        A[:, :n] = torch.randn(n, n, dtype=dtype, device=device) * 0.1
    return A.contiguous()


class Syr2Benchmark(Benchmark):
    DEFAULT_SHAPES = [(n,) for n in SYR2_SIZES]
    DEFAULT_SHAPE_DESC = "N"

    def __init__(self, *args, uplo=CUBLAS_FILL_MODE_LOWER, alpha=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.alpha = alpha

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in SYR2_SIZES]
        return None

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        max_n = max(SYR2_SIZES)
        if any(
            len(shape) not in (1, 2)
            or shape[0] > max_n
            or (len(shape) == 2 and shape[1] > max_n)
            for shape in self.shapes
        ):
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
        if cur_dtype not in _CUBLAS_SYR2_FUNCS:
            raise ValueError(f"Unsupported dtype: {cur_dtype}")
        c_func, ctor, is_complex = _CUBLAS_SYR2_FUNCS[cur_dtype]
        alpha_c = _make_scalar(ctor, is_complex, self.alpha)

        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            lda = shape[1] if isinstance(shape, (tuple, list)) and len(shape) > 1 else n
            A = _generate_syr2_A(n, lda, cur_dtype, self.device)
            x = torch.randn(n, dtype=cur_dtype, device=self.device)
            y = torch.randn(n, dtype=cur_dtype, device=self.device)

            yield A, x, y, {
                "uplo": self.uplo,
                "n": n,
                "alpha": self.alpha,
                "incx": 1,
                "incy": 1,
                "lda": lda,
                "handle": handle,
                "c_func": c_func,
                "alpha_c": alpha_c,
            }

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        return 2 * n * n

    def get_gbps(self, args, latency):
        A, x, y = args[0], args[1], args[2]
        n = x.numel()
        a_bytes = n * (n + 1) // 2 * A.element_size()
        io_amount = (
            a_bytes * 2 + shape_utils.size_in_bytes(x) + shape_utils.size_in_bytes(y)
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return max(1, kwargs.get("n", 0))

    def clone_correctness_inputs(self, args, kwargs):
        A, x, y = args
        ref_args = (A.clone(), x, y)
        blas_args = (A.clone(), x, y)
        return ref_args, kwargs, blas_args, kwargs


@pytest.mark.ssyr2
def test_perf_ssyr2():
    bench = Syr2Benchmark(
        op_name="ssyr2",
        torch_op=cublas_syr2_baseline,
        gems_op=gems_ssyr2_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.ssyr2
def test_perf_ssyr2_upper():
    bench = Syr2Benchmark(
        op_name="ssyr2_upper",
        torch_op=cublas_syr2_baseline,
        gems_op=gems_ssyr2_wrapper,
        dtypes=[torch.float32],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dsyr2
def test_perf_dsyr2():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = Syr2Benchmark(
        op_name="dsyr2",
        torch_op=cublas_syr2_baseline,
        gems_op=gems_dsyr2_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_LOWER,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.dsyr2
def test_perf_dsyr2_upper():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = Syr2Benchmark(
        op_name="dsyr2_upper",
        torch_op=cublas_syr2_baseline,
        gems_op=gems_dsyr2_wrapper,
        dtypes=[torch.float64],
        uplo=CUBLAS_FILL_MODE_UPPER,
    )
    run_correctness_then_benchmark(bench)


# csyr2/zsyr2 benchmarks are intentionally not registered: strict CPU
# correctness has no direct SciPy/CPU BLAS reference for those variants.
