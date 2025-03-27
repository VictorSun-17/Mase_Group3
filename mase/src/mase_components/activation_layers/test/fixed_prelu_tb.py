import torch
from torch import nn
from torch.autograd.function import InplaceFunction

import cocotb
from cocotb.triggers import Timer
import numpy as np

from mase_cocotb.runner import mase_runner


# snippets
class MyClamp(InplaceFunction):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class MyRound(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


my_clamp = MyClamp.apply
my_round = MyRound.apply


def quantize(x, bits, bias):
    thresh = 2 ** (bits - 1)
    scale = 2**bias
    return my_clamp(my_round(x.mul(scale)), -thresh, thresh - 1).div(scale)


class VerificationCase:
    bitwidth = 32
    bias = 4
    num = 16
    alpha = 0.25  # should match Verilog ALPHA

    def __init__(self, samples=2):
        self.alpha_param = torch.tensor([self.alpha], requires_grad=True)
        self.m = nn.PReLU(num_parameters=1)
        self.m.weight.data = self.alpha_param  # assign fixed alpha

        self.inputs, self.outputs = [], []
        for _ in range(samples):
            i, o = self.single_run()
            self.inputs.append(i)
            self.outputs.append(o)
        self.samples = samples

    def single_run(self):
        xs = torch.rand(self.num).float()
        r1, r2 = 4, -4
        xs = (r1 - r2) * xs + r2
        xs = quantize(xs, self.bitwidth, self.bias)
        return xs, self.m(xs)

    def get_dut_parameters(self):
        return {
            "DATA_IN_0_PARALLELISM_DIM_0": self.num,
            "DATA_OUT_0_PARALLELISM_DIM_0": self.num,
            "DATA_IN_0_PRECISION_0": self.bitwidth,
            "DATA_OUT_0_PRECISION_0": self.bitwidth,
        }

    def get_dut_input(self, i):
        inputs = self.inputs[i]
        shifted_integers = (inputs * (2**self.bias)).int()
        return shifted_integers.numpy().tolist()

    def get_dut_output(self, i):
        outputs = self.outputs[i]
        shifted_integers = self.convert_to_fixed(outputs)
        return shifted_integers


    def convert_to_fixed(self, x):
        return (x * (2**self.bias)).round().int().numpy().tolist()



def convert_unsigned_to_signed(val, bitwidth=32):
    if val >= 2**(bitwidth - 1):
        val -= 2**bitwidth
    return val


import numpy as np

@cocotb.test()
async def cocotb_test_fixed_prelu(dut):
    """Test integer based PReLU"""
    test_case = VerificationCase(samples=100)

    for i in range(test_case.samples):
        x = test_case.get_dut_input(i)
        y = test_case.get_dut_output(i)

        dut.data_in_0.value = x
        await Timer(2, units="ns")

        dut_out = [x.integer for x in dut.data_out_0.value]
        dut_out = [convert_unsigned_to_signed(v, bitwidth=test_case.bitwidth) for v in dut_out]
        dut_out = np.array(dut_out)

        print(f"DUT output:  {dut_out}")
        print(f"Expected:    {y}")

        
        diffs = np.abs(dut_out - np.array(y))
        assert np.all(diffs <= 1), f"Mismatch >1 at sample {i}, diffs = {diffs}"




def test_fixed_prelu():
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])


if __name__ == "__main__":
    test_fixed_prelu()
