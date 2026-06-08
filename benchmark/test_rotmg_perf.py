import ctypes
import ctypes.util
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark

ROTMG_CASES = [
    (-1.0, 2.0, 3.0, 4.0),
    (1.0, 0.0, 2.0, 3.0),
    (1.0, 2.0, 3.0, 0.0),
    (2.0, 1.0, 3.0, 1.0),
    (1.0, -2.0, 3.0, 4.0),
    (1.0, 2.0, 1.0, 1.0),
]
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


def normalize_rotmg_param(param):
    flag = param[0].item()
    if flag == -2.0:
        param[1:].zero_()
    elif flag == 0.0:
        param[1].zero_()
        param[4].zero_()
    elif flag == 1.0:
        param[2].zero_()
        param[3].zero_()


for _rotmg_name in ("cublasSrotmg_v2", "cublasDrotmg_v2"):
    _rotmg_func = getattr(_cublas, _rotmg_name)
    _rotmg_func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _rotmg_func.restype = ctypes.c_int


def create_cublas_handle():
    handle = ctypes.c_void_p()
    status = _cublas.cublasCreate_v2(ctypes.byref(handle))
    if status != 0:
        raise RuntimeError(f"cuBLAS handle creation failed, error code: {status}")
    status = _cublas.cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_DEVICE)
    if status != 0:
        raise RuntimeError(f"cuBLAS pointer mode setup failed, error code: {status}")
    return handle


def cublas_rotmg(d1, d2, x1, y1, param, handle=None):
    if handle is None:
        handle = create_cublas_handle()
    func = (
        _cublas.cublasSrotmg_v2
        if d1.dtype == torch.float32
        else _cublas.cublasDrotmg_v2
    )
    status = func(
        handle,
        ctypes.c_void_p(d1.data_ptr()),
        ctypes.c_void_p(d2.data_ptr()),
        ctypes.c_void_p(x1.data_ptr()),
        ctypes.c_void_p(y1.data_ptr()),
        ctypes.c_void_p(param.data_ptr()),
    )
    if status != 0:
        raise RuntimeError(f"cuBLAS rotmg execution failed, error code: {status}")
    normalize_rotmg_param(param)
    return d1, d2, x1, y1, param


def gems_rotmg_wrapper(d1, d2, x1, y1, param, handle=None):
    if d1.dtype == torch.float32:
        flag_blas.ops.srotmg(d1, d2, x1, y1, param)
    elif d1.dtype == torch.float64:
        flag_blas.ops.drotmg(d1, d2, x1, y1, param)
    else:
        raise TypeError(f"Unsupported dtype for rotmg: {d1.dtype}")
    return d1, d2, x1, y1, param


class RotmgBenchmark(Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1,)]
        self.shape_desc = "scalar"

    def get_input_iter(self, cur_dtype) -> Generator:
        handle = create_cublas_handle()
        for d1_val, d2_val, x1_val, y1_val in ROTMG_CASES:
            d1 = torch.tensor([d1_val], dtype=cur_dtype, device=self.device)
            d2 = torch.tensor([d2_val], dtype=cur_dtype, device=self.device)
            x1 = torch.tensor([x1_val], dtype=cur_dtype, device=self.device)
            y1 = torch.tensor([y1_val], dtype=cur_dtype, device=self.device)
            param = torch.zeros(5, dtype=cur_dtype, device=self.device)
            yield d1, d2, x1, y1, param, {"handle": handle}


@pytest.mark.rotmg
def test_perf_srotmg():
    run_correctness_then_benchmark(
        RotmgBenchmark(
            op_name="srotmg",
            torch_op=cublas_rotmg,
            blas_op=gems_rotmg_wrapper,
            dtypes=[torch.float32],
        )
    )


@pytest.mark.rotmg
def test_perf_drotmg():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    run_correctness_then_benchmark(
        RotmgBenchmark(
            op_name="drotmg",
            torch_op=cublas_rotmg,
            blas_op=gems_rotmg_wrapper,
            dtypes=[torch.float64],
        )
    )
