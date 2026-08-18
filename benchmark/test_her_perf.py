import ctypes
import ctypes.util
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.ops import CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER
from flag_blas.utils import shape_utils

HER_SIZES = [
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
CUBLAS_POINTER_MODE_HOST = 0


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


def _configure_cublas_signatures():
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int
    _cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas.cublasSetPointerMode_v2.restype = ctypes.c_int
    _cublas.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _cublas.cublasSetStream_v2.restype = ctypes.c_int
    for name in ("cublasCher_v2", "cublasZher_v2"):
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
        ]
        func.restype = ctypes.c_int


_configure_cublas_signatures()
_CUBLAS_HER_FUNCS = {
    torch.complex64: (_cublas.cublasCher_v2, ctypes.c_float),
    torch.complex128: (_cublas.cublasZher_v2, ctypes.c_double),
}


def _get_cublas_handle():
    global _cublas_handle
    if _cublas_handle is None:
        _cublas_handle = ctypes.c_void_p()
        status = _cublas.cublasCreate_v2(ctypes.byref(_cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status code: {status}")
        status = _cublas.cublasSetPointerMode_v2(
            _cublas_handle, CUBLAS_POINTER_MODE_HOST
        )
        if status != 0:
            raise RuntimeError(
                f"cublasSetPointerMode_v2 failed with status code: {status}"
            )
    status = _cublas.cublasSetStream_v2(
        _cublas_handle,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )
    if status != 0:
        raise RuntimeError(f"cublasSetStream_v2 failed with status code: {status}")
    return _cublas_handle


def cublas_her_baseline(
    A, x, uplo, n, alpha, incx, lda, handle, c_func, alpha_c, **kwargs
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
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(lda),
    )
    if status != 0:
        raise RuntimeError(f"cublasXher_v2 failed with status code: {status}")
    return A


def gems_cher_wrapper(A, x, uplo, n, alpha, incx, lda, handle, **kwargs):
    flag_blas.ops.cher(uplo, n, alpha, x, incx, A, lda)
    return A


def gems_zher_wrapper(A, x, uplo, n, alpha, incx, lda, handle, **kwargs):
    flag_blas.ops.zher(uplo, n, alpha, x, incx, A, lda)
    return A


GEMS_HER_WRAPPERS = {
    "cher": gems_cher_wrapper,
    "zher": gems_zher_wrapper,
}


def _generate_A(n, lda, dtype, device):
    A = torch.randn((n, lda), dtype=dtype, device=device)
    if n > 0:
        diag_real = A[:, :n].diagonal().real.clone()
        A[:, :n].diagonal().copy_(diag_real.to(dtype))
    return A.contiguous()


class HerBenchmark(Benchmark):
    DEFAULT_SHAPES = [(n,) for n in HER_SIZES]
    DEFAULT_SHAPE_DESC = "N"

    def __init__(self, *args, uplo=CUBLAS_FILL_MODE_LOWER, alpha=0.75, **kwargs):
        super().__init__(*args, **kwargs)
        self.uplo = uplo
        self.alpha = alpha

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        self.shapes = [(n,) for n in HER_SIZES]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = _get_cublas_handle()
        c_func, ctor = _CUBLAS_HER_FUNCS[cur_dtype]
        alpha_c = ctor(self.alpha)
        for shape in self.shapes:
            n = shape[0] if isinstance(shape, (tuple, list)) else shape
            lda = n
            A = _generate_A(n, lda, cur_dtype, self.device)
            x = torch.randn(n, dtype=cur_dtype, device=self.device)
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

    def get_tflops(self, op, *args, **kwargs):
        n = kwargs.get("n", 0)
        return 2 * n * n

    def get_gbps(self, args, latency):
        A, x = args[0], args[1]
        io_amount = 2 * shape_utils.size_in_bytes(A) + shape_utils.size_in_bytes(x)
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_correctness_reduce_dim(self, args, kwargs):
        return max(1, kwargs.get("n", 0))

    def clone_correctness_inputs(self, args, kwargs):
        A, x = args
        return (A.clone(), x.clone()), kwargs, (A.clone(), x.clone()), kwargs


def _run_her(op_name, dtype, uplo):
    if dtype == torch.complex128 and not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support complex128")
    bench = HerBenchmark(
        op_name=op_name,
        torch_op=cublas_her_baseline,
        gems_op=GEMS_HER_WRAPPERS[op_name],
        dtypes=[dtype],
        uplo=uplo,
    )
    run_correctness_then_benchmark(bench)


HER_PERF_CASES = [
    pytest.param(
        op_name,
        dtype,
        uplo,
        marks=getattr(pytest.mark, op_name),
        id=f"{op_name}-{uplo}",
    )
    for op_name, dtype in (
        ("cher", torch.complex64),
        ("zher", torch.complex128),
    )
    for uplo in (CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER)
]


@pytest.mark.parametrize("op_name,dtype,uplo", HER_PERF_CASES)
def test_perf_her(op_name, dtype, uplo):
    _run_her(op_name, dtype, uplo)
