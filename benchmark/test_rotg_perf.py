from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark


def _real_rotg_reference(a, b, c, s):
    abs_a = torch.abs(a)
    abs_b = torch.abs(b)
    scale = abs_a + abs_b
    if scale.item() == 0.0:
        a.zero_()
        b.zero_()
        c.fill_(1.0)
        s.zero_()
        return a, b, c, s
    roe = a if abs_a.item() > abs_b.item() else b
    r = scale * torch.sqrt((a / scale) * (a / scale) + (b / scale) * (b / scale))
    if roe.item() < 0.0:
        r = -r
    c.copy_(a / r)
    s.copy_(b / r)
    z = torch.ones_like(b)
    if abs_a.item() > abs_b.item():
        z.copy_(s)
    elif c.item() != 0.0:
        z.copy_(1.0 / c)
    a.copy_(r)
    b.copy_(z)
    return a, b, c, s


def _complex_rotg_reference(a, b, c, s):
    abs_a = torch.abs(a)
    if abs_a.item() == 0.0:
        c.zero_()
        s.fill_(1.0 + 0.0j)
        a.copy_(b)
        return a, b, c, s
    abs_b = torch.abs(b)
    scale = abs_a + abs_b
    norm = scale * torch.sqrt(torch.abs(a / scale) ** 2 + torch.abs(b / scale) ** 2)
    alpha = a / abs_a
    c.copy_(abs_a / norm)
    s.copy_(alpha * torch.conj(b) / norm)
    a.copy_(alpha * norm)
    return a, b, c, s


def torch_rotg(a, b, c, s):
    a = a.clone()
    b = b.clone()
    c = c.clone()
    s = s.clone()
    if a.dtype.is_complex:
        return _complex_rotg_reference(a, b, c, s)
    return _real_rotg_reference(a, b, c, s)


def gems_rotg(a, b, c, s):
    a = a.clone()
    b = b.clone()
    c = c.clone()
    s = s.clone()
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
    correctness_reference = "Torch reference"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1,)]
        self.shape_desc = "scalar"

    def get_input_iter(self, cur_dtype) -> Generator:
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
            yield a, b, c, s, {}


@pytest.mark.rotg
def test_perf_srotg():
    run_correctness_then_benchmark(
        RotgBenchmark(
            "srotg",
            torch_op=torch_rotg,
            gems_op=gems_rotg,
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
            torch_op=torch_rotg,
            gems_op=gems_rotg,
            dtypes=[torch.float64],
        )
    )


@pytest.mark.rotg
def test_perf_crotg():
    run_correctness_then_benchmark(
        RotgBenchmark(
            "crotg",
            torch_op=torch_rotg,
            gems_op=gems_rotg,
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
            torch_op=torch_rotg,
            gems_op=gems_rotg,
            dtypes=[torch.complex128],
        )
    )
