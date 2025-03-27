import torch
from torch import nn
from torch.autograd.function import InplaceFunction

import cocotb
from cocotb.triggers import Timer
import numpy as np

from mase_cocotb.runner import mase_runner


# custom autograd-safe clamp & round
class MyClamp(InplaceFunction):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class MyRound(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


my_clamp = MyClamp.apply
my_round = MyRound.apply


# fixed-point quantization with clamping
def quantize(x, bits, bias):
    thresh = 2 ** (bits - 1)
    scale = 2 ** bias
    return my_clamp(my_round(x.mul(scale)), -thresh, thresh - 1).div(scale)


class RReLUVerificationCase:
    bitwidth = 32
    bias = 4
    num = 16
    negative_slope = 4.25  # must match NEGATIVE_SLOPE in hardware (e.g., 8'd32 in Q4)

    def __init__(self, samples=2):
        self.inputs, self.outputs = [], []
        for _ in range(samples):
            i, o = self.single_run()
            self.inputs.append(i)
            self.outputs.append(o)
        self.samples = samples

    def single_run(self):
        # Random inputs in range [-5, 10]
        xs = torch.rand(self.num) * 15 - 5
        xs = quantize(xs, self.bitwidth, self.bias)

        # Emulate hardware RReLU: apply deterministic fixed slope (0.25)
        ys = torch.where(xs > 0, xs, xs * self.negative_slope)
        ys = quantize(ys, self.bitwidth, self.bias)
        return xs, ys

    def get_dut_parameters(self):
        return {
            "DATA_IN_0_PARALLELISM_DIM_0": self.num,
            "DATA_OUT_0_PARALLELISM_DIM_0": self.num,
            "DATA_IN_0_PRECISION_0": self.bitwidth,
            "DATA_OUT_0_PRECISION_0": self.bitwidth,
        }

    def get_dut_input(self, i):
        inputs = self.inputs[i]
        return (inputs * (2 ** self.bias)).int().numpy().tolist()

    def get_dut_output(self, i):
        outputs = self.outputs[i]
        return (outputs * (2 ** self.bias)).int().numpy().tolist()


@cocotb.test()
async def cocotb_test_fixed_rrelu(dut):
    """Test integer-based RReLU"""
    test_case = RReLUVerificationCase(samples=100)

    for i in range(test_case.samples):
        x = test_case.get_dut_input(i)
        y = test_case.get_dut_output(i)

        dut.data_in_0.value = x
        await Timer(2, units="ns")
        dut_out = [v.signed_integer for v in dut.data_out_0.value]
        # print(f"Cycle {i}:")
        # print(f" Input     : {x}")
        # print(f" DUT Output: {dut_out}")
        # print(f" Expected  : {y}")
        for j, (a, b) in enumerate(zip(dut_out, y)):
            assert abs(a - b) <= 1, f"Mismatch at sample {i}, index {j}: got {a}, expected {b}"


def test_fixed_rrelu():
    tb = RReLUVerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])


if __name__ == "__main__":
    test_fixed_rrelu()