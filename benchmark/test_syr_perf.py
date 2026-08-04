import ctypes
import ctypes.util
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER
from flag_blas.utils import shape_utils

SYR_SIZES = [
    64,
    96,
    127,
    128,
    129,
    192,
    255,
    256,
    257,
    384,
    511,
    512,
    513,
    768,
    1023,
    1024,
    1025,
    1536,
    2048,
    3072,
    4096,
]


class _ComplexFloat(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]


class _ComplexDouble(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


def _load_cublas():
    names = ["libcublas.so.13"]
    found = ctypes.util.find_library("cublas")
    if found:
        names.append(found)
    names.extend(["libcublas.so", "libcublas.so.12", "libcublas.so.11"])
    for name in names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError("Unable to find libcublas.so")


_cublas = _load_cublas()
_cublas_handle = None
_CUBLAS_SYR_FUNCS = {
    torch.float32: (_cublas.cublasSsyr_v2, ctypes.c_float),
    torch.float64: (_cublas.cublasDsyr_v2, ctypes.c_double),
    torch.complex64: (_cublas.cublasCsyr_v2, _ComplexFloat),
    torch.complex128: (_cublas.cublasZsyr_v2, _ComplexDouble),
}


def _get_cublas_handle():
    global _cublas_handle
    if _cublas_handle is None:
        handle = ctypes.c_void_p()
        status = _cublas.cublasCreate_v2(ctypes.byref(handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status code: {status}")
        _cublas_handle = handle.value
    return _cublas_handle


def _make_scalar(ctor, value):
    if ctor in (_ComplexFloat, _ComplexDouble):
        value = complex(value)
        return ctor(value.real, value.imag)
    return ctor(value)


def cublas_syr_baseline(
    A, x, uplo, n, alpha, incx, lda, handle, c_func, alpha_c, **kwargs
):
    if n == 0:
        return A
    status = c_func(
        ctypes.c_void_p(handle),
        ctypes.c_int(uplo),
        ctypes.c_int(n),
        ctypes.byref(alpha_c),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_int(incx),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
    )
    if status != 0:
        raise RuntimeError(f"cublasXsyr_v2 failed with status code: {status}")
    return A


def _gems_wrapper(op):
    def _impl(A, x, uplo, n, alpha, incx, lda, **kwargs):
        op(uplo, n, alpha, x, incx, A, lda)
        return A

    return _impl


GEMS_SYR_WRAPPERS = {
    "ssyr": _gems_wrapper(flag_blas.ssyr),
    "dsyr": _gems_wrapper(flag_blas.dsyr),
    "csyr": _gems_wrapper(flag_blas.csyr),
    "zsyr": _gems_wrapper(flag_blas.zsyr),
}


class SyrBenchmark(Benchmark):
    DEFAULT_SHAPES = [(n,) for n in SYR_SIZES]
    DEFAULT_SHAPE_DESC = "N"

    def __init__(self, *args, uplo=CUBLAS_FILL_MODE_LOWER, alpha=0.75, **kwargs):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.alpha = alpha

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in SYR_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = _get_cublas_handle()
        c_func, ctor = _CUBLAS_SYR_FUNCS[cur_dtype]
        alpha_c = _make_scalar(ctor, self.alpha)
        for shape in self.shapes:
            n = shape[0]
            lda = n
            x = torch.randn(n, dtype=cur_dtype, device=self.device)
            A = torch.randn((n, lda), dtype=cur_dtype, device=self.device)
            yield A, x, {
                "uplo": self.uplo,
                "n": n,
                "alpha": self.alpha,
                "incx": 1,
                "lda": lda,
                "handle": handle,
                "c_func": c_func,
                "alpha_c": alpha_c,
            }

    def get_correctness_reduce_dim(self, args, kwargs):
        return max(1, kwargs["n"])

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        return n * (n + 1)

    def get_gbps(self, args, latency):
        A, x = args[0], args[1]
        n = x.numel()
        triangle_bytes = n * (n + 1) // 2 * A.element_size()
        io_amount = 2 * triangle_bytes + shape_utils.size_in_bytes(x)
        return io_amount * 1e-9 / (latency * 1e-3)


def _run_syr(op_name, dtype, uplo, alpha):
    if (
        dtype in (torch.float64, torch.complex128)
        and not flag_blas.runtime.device.support_fp64
    ):
        pytest.skip("fp64 is not supported on this device")
    bench = SyrBenchmark(
        op_name=f"{op_name}_{uplo}",
        torch_op=cublas_syr_baseline,
        gems_op=GEMS_SYR_WRAPPERS[op_name],
        dtypes=[dtype],
        uplo=uplo,
        alpha=alpha,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.parametrize(
    "op_name,dtype,alpha,uplo",
    [
        pytest.param(
            "ssyr", torch.float32, 0.75, CUBLAS_FILL_MODE_LOWER, marks=pytest.mark.ssyr
        ),
        pytest.param(
            "ssyr", torch.float32, 0.75, CUBLAS_FILL_MODE_UPPER, marks=pytest.mark.ssyr
        ),
        pytest.param(
            "dsyr",
            torch.float64,
            0.12345678901234568,
            CUBLAS_FILL_MODE_LOWER,
            marks=pytest.mark.dsyr,
        ),
        pytest.param(
            "dsyr",
            torch.float64,
            0.12345678901234568,
            CUBLAS_FILL_MODE_UPPER,
            marks=pytest.mark.dsyr,
        ),
        pytest.param(
            "csyr",
            torch.complex64,
            complex(0.5, -0.25),
            CUBLAS_FILL_MODE_LOWER,
            marks=pytest.mark.csyr,
        ),
        pytest.param(
            "csyr",
            torch.complex64,
            complex(0.5, -0.25),
            CUBLAS_FILL_MODE_UPPER,
            marks=pytest.mark.csyr,
        ),
        pytest.param(
            "zsyr",
            torch.complex128,
            complex(0.12345678901234568, -0.2345678912345679),
            CUBLAS_FILL_MODE_LOWER,
            marks=pytest.mark.zsyr,
        ),
        pytest.param(
            "zsyr",
            torch.complex128,
            complex(0.12345678901234568, -0.2345678912345679),
            CUBLAS_FILL_MODE_UPPER,
            marks=pytest.mark.zsyr,
        ),
    ],
)
def test_perf_syr(op_name, dtype, alpha, uplo):
    _run_syr(op_name, dtype, uplo, alpha)
