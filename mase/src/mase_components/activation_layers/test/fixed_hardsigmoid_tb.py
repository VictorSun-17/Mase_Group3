import torch
import torch.nn as nn
import numpy as np
import cocotb
from cocotb.triggers import Timer
from mase_cocotb.runner import mase_runner


def quantize(x, bits, bias):
    scale = 2**bias
    max_val = 2**(bits - 1) - 1
    min_val = -2**(bits - 1)
    return torch.clamp((x * scale).round(), min_val, max_val) / scale


class HardsigmoidDefined(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        out[x <= -3.0] = 0.0
        out[x >= 3.0] = 1.0
        mask = (x > -3.0) & (x < 3.0)
        out[mask] = x[mask] / 6.0 + 0.5
        return out


class VerificationCase:
    bitwidth = 8
    bias = 4
    num = 16

    def __init__(self, samples=50):
        self.model = HardsigmoidDefined()
        self.inputs, self.outputs = [], []
        for _ in range(samples):
            x, y = self.single_run()
            self.inputs.append(x)
            self.outputs.append(y)
        self.samples = samples

    def single_run(self):
        x = (torch.rand(self.num) - 0.5) * 12
        x = quantize(x, self.bitwidth, self.bias)
        y = self.model(x)
        return x, quantize(y, self.bitwidth, self.bias)

    def get_dut_parameters(self):
        return {
            "DATA_IN_0_PRECISION_0": self.bitwidth,
            "DATA_IN_0_PRECISION_1": self.bias,
            "DATA_IN_0_TENSOR_SIZE_DIM_0": self.num,
            "DATA_IN_0_TENSOR_SIZE_DIM_1": 1,
            "DATA_IN_0_PARALLELISM_DIM_0": self.num,
            "DATA_IN_0_PARALLELISM_DIM_1": 1,
            "DATA_OUT_0_PRECISION_0": self.bitwidth,
            "DATA_OUT_0_PRECISION_1": self.bias,
            "DATA_OUT_0_TENSOR_SIZE_DIM_0": self.num,
            "DATA_OUT_0_TENSOR_SIZE_DIM_1": 1,
            "DATA_OUT_0_PARALLELISM_DIM_0": self.num,
            "DATA_OUT_0_PARALLELISM_DIM_1": 1,
            "INPLACE": 0
        }

    def get_dut_input(self, i):
        return (self.inputs[i] * (2 ** self.bias)).int().tolist()

    def get_dut_output(self, i):
        return (self.outputs[i] * (2 ** self.bias)).int().tolist()


def convert_unsigned_to_signed(val, bitwidth=8):
    if val >= 2 ** (bitwidth - 1):
        val -= 2 ** bitwidth
    return val


@cocotb.test()
async def cocotb_test_fixed_hardsigmoid(dut):
    test_case = VerificationCase(samples=50)

    for i in range(test_case.samples):
        x = test_case.get_dut_input(i)
        y = test_case.get_dut_output(i)

        dut.data_in_0.value = x
        dut.data_in_0_valid.value = 1
        dut.data_out_0_ready.value = 1

        await Timer(2, units="ns")

        dut_out = [convert_unsigned_to_signed(v.integer, test_case.bitwidth) for v in dut.data_out_0.value]
        diffs = np.abs(np.array(dut_out) - np.array(y))
        if not np.all(diffs <= 1):
            print(f"\n Sample {i} mismatch:")
            print(f"Input    : {test_case.inputs[i].tolist()}")
            print(f"DUT out  : {dut_out}")
            print(f"Expected : {y}")
        assert np.all(diffs <= 1), f"Mismatch >1 at sample {i}"


def test_fixed_hardsigmoid():
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])


if __name__ == "__main__":
    test_fixed_hardsigmoid()
