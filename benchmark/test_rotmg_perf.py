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
    (1e-20, 2.0, 3.0, 4.0),
    (1e20, 2.0, 3.0, 4.0),
]


def torch_rotmg_reference(d1, d2, x1, y1, param):
    gam = 4096.0
    gamsq = gam * gam
    rgamsq = 1.0 / gamsq
    flag = -2.0
    h11 = h12 = h21 = h22 = 0.0

    d1_val = float(d1.item())
    d2_val = float(d2.item())
    x1_val = float(x1.item())
    y1_val = float(y1.item())

    if d1_val < 0.0:
        flag = -1.0
        d1_val = d2_val = x1_val = 0.0
    else:
        p2 = d2_val * y1_val
        if p2 != 0.0:
            p1 = d1_val * x1_val
            q2 = p2 * y1_val
            q1 = p1 * x1_val

            if abs(q1) > abs(q2):
                h21 = -y1_val / x1_val
                h12 = p2 / p1
                u = 1.0 - h12 * h21
                if u > 0.0:
                    flag = 0.0
                    d1_val = d1_val / u
                    d2_val = d2_val / u
                    x1_val = x1_val * u
                else:
                    flag = -1.0
                    d1_val = d2_val = x1_val = 0.0
            elif q2 < 0.0:
                flag = -1.0
                d1_val = d2_val = x1_val = 0.0
            else:
                flag = 1.0
                h11 = p1 / p2
                h22 = x1_val / y1_val
                u = 1.0 + h11 * h22
                temp = d2_val / u
                d2_val = d1_val / u
                d1_val = temp
                x1_val = y1_val * u

            if flag >= 0.0:
                while d1_val != 0.0 and (d1_val <= rgamsq or d1_val >= gamsq):
                    if flag == 0.0:
                        h11 = 1.0
                        h22 = 1.0
                    else:
                        h21 = -1.0
                        h12 = 1.0
                    flag = -1.0

                    if d1_val <= rgamsq:
                        d1_val = d1_val * gamsq
                        x1_val = x1_val / gam
                        h11 = h11 / gam
                        h12 = h12 / gam
                    else:
                        d1_val = d1_val / gamsq
                        x1_val = x1_val * gam
                        h11 = h11 * gam
                        h12 = h12 * gam

                while d2_val != 0.0 and (
                    abs(d2_val) <= rgamsq or abs(d2_val) >= gamsq
                ):
                    if flag == 0.0:
                        h11 = 1.0
                        h22 = 1.0
                    else:
                        h21 = -1.0
                        h12 = 1.0
                    flag = -1.0

                    if abs(d2_val) <= rgamsq:
                        d2_val = d2_val * gamsq
                        h21 = h21 / gam
                        h22 = h22 / gam
                    else:
                        d2_val = d2_val / gamsq
                        h21 = h21 * gam
                        h22 = h22 * gam

    d1.fill_(d1_val)
    d2.fill_(d2_val)
    x1.fill_(x1_val)
    param.zero_()
    param[0] = flag
    if flag < 0.0:
        param[1] = h11
        param[2] = h21
        param[3] = h12
        param[4] = h22
    elif flag == 0.0:
        param[2] = h21
        param[3] = h12
    else:
        param[1] = h11
        param[4] = h22
    return d1, d2, x1, y1, param


def gems_rotmg_wrapper(d1, d2, x1, y1, param):
    if d1.dtype == torch.float32:
        flag_blas.ops.srotmg(d1, d2, x1, y1, param)
    elif d1.dtype == torch.float64:
        flag_blas.ops.drotmg(d1, d2, x1, y1, param)
    else:
        raise TypeError(f"Unsupported dtype for rotmg: {d1.dtype}")
    return d1, d2, x1, y1, param


class RotmgBenchmark(Benchmark):
    correctness_reference = "Torch reference"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1,)]
        self.shape_desc = "scalar"

    def get_input_iter(self, cur_dtype) -> Generator:
        for d1_val, d2_val, x1_val, y1_val in ROTMG_CASES:
            d1 = torch.tensor([d1_val], dtype=cur_dtype, device=self.device)
            d2 = torch.tensor([d2_val], dtype=cur_dtype, device=self.device)
            x1 = torch.tensor([x1_val], dtype=cur_dtype, device=self.device)
            y1 = torch.tensor([y1_val], dtype=cur_dtype, device=self.device)
            param = torch.zeros(5, dtype=cur_dtype, device=self.device)
            yield d1, d2, x1, y1, param, {}


@pytest.mark.rotmg
def test_perf_srotmg():
    run_correctness_then_benchmark(
        RotmgBenchmark(
            op_name="srotmg",
            torch_op=torch_rotmg_reference,
            gems_op=gems_rotmg_wrapper,
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
            torch_op=torch_rotmg_reference,
            gems_op=gems_rotmg_wrapper,
            dtypes=[torch.float64],
        )
    )
