import ctypes
import ctypes.util
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark

CUBLAS_POINTER_MODE_DEVICE = 1


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
    raise RuntimeError("Cannot find libcublas.so in the system")


_cublas = load_cublas()
_cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_cublas.cublasCreate_v2.restype = ctypes.c_int
_cublas.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
_cublas.cublasSetPointerMode_v2.restype = ctypes.c_int

for _rotg_name in (
    "cublasSrotg_v2",
    "cublasDrotg_v2",
    "cublasCrotg_v2",
    "cublasZrotg_v2",
):
    _rotg_func = getattr(_cublas, _rotg_name)
    _rotg_func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _rotg_func.restype = ctypes.c_int


def create_cublas_handle():
    handle = ctypes.c_void_p()
    status = _cublas.cublasCreate_v2(ctypes.byref(handle))
    if status != 0:
        raise RuntimeError(f"cuBLAS handle creation failed, error code: {status}")
    status = _cublas.cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_DEVICE)
    if status != 0:
        raise RuntimeError(f"cuBLAS pointer mode setup failed, error code: {status}")
    return handle


def as_cublas_handle(handle):
    if isinstance(handle, ctypes.c_void_p):
        return handle
    return ctypes.c_void_p(handle)


def cublas_rotg(a, b, c, s, handle=None):
    if handle is None:
        handle = create_cublas_handle()

    if a.dtype == torch.float32:
        func = _cublas.cublasSrotg_v2
    elif a.dtype == torch.float64:
        func = _cublas.cublasDrotg_v2
    elif a.dtype == torch.complex64:
        func = _cublas.cublasCrotg_v2
    elif a.dtype == torch.complex128:
        func = _cublas.cublasZrotg_v2
    else:
        raise TypeError(f"Unsupported dtype for rotg: {a.dtype}")

    status = func(
        as_cublas_handle(handle),
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c.data_ptr()),
        ctypes.c_void_p(s.data_ptr()),
    )
    if status != 0:
        raise RuntimeError(f"cuBLAS rotg execution failed, error code: {status}")
    return a, b, c, s


def gems_rotg(a, b, c, s, handle=None):
    if a.dtype == torch.float32:
        flag_blas.ops.srotg(a, b, c, s)
    elif a.dtype == torch.float64:
        flag_blas.ops.drotg(a, b, c, s)
    elif a.dtype == torch.complex64:
        flag_blas.ops.crotg(a, b, c, s)
    elif a.dtype == torch.complex128:
        flag_blas.ops.zrotg(a, b, c, s)
    else:
        raise TypeError(f"Unsupported dtype for rotg: {a.dtype}")
    return a, b, c, s


class RotgBenchmark(Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1,)]
        self.shape_desc = "scalar"

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = create_cublas_handle()
        cases = (
            [
                (1.0 + 2.0j, 3.0 + 4.0j),
                (0.0 + 0.0j, 1.0 + 0.0j),
                (0.0 + 1.0j, 0.0 + 0.0j),
                (-2.0 - 3.0j, -4.0 + 5.0j),
                (1e-20 + 0.0j, 1.0 + 0.0j),
                (1e20 + 1e20j, -1e20 + 1e20j),
            ]
            if cur_dtype.is_complex
            else [
                (3.0, 4.0),
                (4.0, 3.0),
                (-3.0, 4.0),
                (-3.0, -4.0),
                (0.0, -1.0),
                (1.0, 0.0),
                (0.0, 0.0),
                (1e-20, 1.0),
                (1.0, 1e-20),
                (1e20, -1e20),
            ]
        )
        for a_val, b_val in cases:
            a = torch.tensor([a_val], dtype=cur_dtype, device=self.device)
            b = torch.tensor([b_val], dtype=cur_dtype, device=self.device)
            real_dtype = (
                torch.float32
                if cur_dtype in (torch.float32, torch.complex64)
                else torch.float64
            )
            c = torch.zeros(1, dtype=real_dtype, device=self.device)
            s = torch.zeros(1, dtype=cur_dtype, device=self.device)
            yield a, b, c, s, {"handle": handle}


@pytest.mark.rotg
def test_perf_srotg():
    run_correctness_then_benchmark(
        RotgBenchmark(
            "srotg",
            torch_op=cublas_rotg,
            blas_op=gems_rotg,
            dtypes=[torch.float32],
        )
    )


@pytest.mark.rotg
def test_perf_drotg():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    run_correctness_then_benchmark(
        RotgBenchmark(
            "drotg",
            torch_op=cublas_rotg,
            blas_op=gems_rotg,
            dtypes=[torch.float64],
        )
    )


@pytest.mark.rotg
def test_perf_crotg():
    run_correctness_then_benchmark(
        RotgBenchmark(
            "crotg",
            torch_op=cublas_rotg,
            blas_op=gems_rotg,
            dtypes=[torch.complex64],
        )
    )


@pytest.mark.rotg
def test_perf_zrotg():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    run_correctness_then_benchmark(
        RotgBenchmark(
            "zrotg",
            torch_op=cublas_rotg,
            blas_op=gems_rotg,
            dtypes=[torch.complex128],
        )
    )
