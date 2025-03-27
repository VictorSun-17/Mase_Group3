import logging
from math import ceil
from pathlib import Path
from os import makedirs

import torch
import torch.nn as nn
import numpy as np
import cocotb
from cocotb.triggers import Timer
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, ErrorThresholdStreamMonitor
from mase_cocotb.runner import mase_runner
from mase_cocotb.z_qlayers import quantize_to_int as q2i
from mase_cocotb.matrix_tools import (
    gen_random_matrix_input,
    rebuild_matrix,
    batched
)
from mase_cocotb.utils import (
    bit_driver,
    sign_extend_t,
    verilator_str_param,
)

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


class ResBlockTB(Testbench):
    def __init__(self, dut, samples=1):
        super().__init__(dut, dut.clk, dut.rst)
        self.samples = samples
        
        # Register all parameters that will be used
        self.assign_self_params([
            "DATA_OUT_0_PRECISION_0",
            "DATA_OUT_0_PRECISION_1",
            "IN_C",
            "GROUP_CHANNELS",
            "OUT_C",
            "KERNEL_Y",
            "KERNEL_X",
            "STRIDE",
            "PADDING_Y",
            "PADDING_X",
            "HAS_BIAS",
            "IN_WIDTH",
            "IN_FRAC_WIDTH",
            "WEIGHT_PRECISION_0",
            "WEIGHT_PRECISION_1",
            "BIAS_PRECISION_0",
            "BIAS_PRECISION_1",
            # For matrix input formatting used by group_norm_2d
            "TOTAL_DIM0",
            "TOTAL_DIM1",
            "COMPUTE_DIM0",
            "COMPUTE_DIM1",
            # For output formatting from convolution
            "UNROLL_OUT_C",
        ])

        # Input data driver (for group_norm_2d input)
        self.data_in_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )

        # Weight and bias drivers (for convolution)
        self.weight_driver = StreamDriver(
            dut.clk, dut.weight, dut.weight_valid, dut.weight_ready
        )
        self.bias_driver = StreamDriver(
            dut.clk, dut.bias, dut.bias_valid, dut.bias_ready
        )

        # Output monitor (for final convolution output)
        self.output_monitor = ErrorThresholdStreamMonitor(
            dut.clk,
            dut.data_out_0,
            dut.data_out_0_valid,
            dut.data_out_0_ready,
            width=self.get_parameter("DATA_OUT_0_PRECISION_0"),
            signed=True,
            error_bits=1,
            check=True
        )

        # PyTorch model (GroupNorm -> Mish -> Conv2d)
        self.model = nn.Sequential(
            nn.GroupNorm(
                num_groups=self.get_parameter("IN_C") // self.get_parameter("GROUP_CHANNELS"),
                num_channels=self.get_parameter("IN_C"),
                affine=False,
            ),
            nn.Mish(),  # Mish activation function
            nn.Conv2d(
                in_channels=self.get_parameter("IN_C"),
                out_channels=self.get_parameter("OUT_C"),
                kernel_size=(self.get_parameter("KERNEL_Y"), self.get_parameter("KERNEL_X")),
                stride=self.get_parameter("STRIDE"),
                padding=(self.get_parameter("PADDING_Y"), self.get_parameter("PADDING_X")),
                bias=self.get_parameter("HAS_BIAS") == 1
            )
        )

        torch.manual_seed(0)
        for p in self.model.parameters():
            if p is not None:
                p.data.normal_()

    def generate_data(self):
        # Configuration for quantization
        cfg = {
            "in_width": self.get_parameter("IN_WIDTH"),
            "in_frac": self.get_parameter("IN_FRAC_WIDTH"),
            "out_width": self.get_parameter("DATA_OUT_0_PRECISION_0"),
            "out_frac": self.get_parameter("DATA_OUT_0_PRECISION_1"),
            "weight_width": self.get_parameter("WEIGHT_PRECISION_0"),
            "weight_frac": self.get_parameter("WEIGHT_PRECISION_1"),
            "bias_width": self.get_parameter("BIAS_PRECISION_0"),
            "bias_frac": self.get_parameter("BIAS_PRECISION_1"),
        }

        # Get parameters for matrix input generation
        total_tup = (self.get_parameter("TOTAL_DIM0"), self.get_parameter("TOTAL_DIM1"))
        compute_tup = (self.get_parameter("COMPUTE_DIM0"), self.get_parameter("COMPUTE_DIM1"))
        in_width_tup = (self.get_parameter("IN_WIDTH"), self.get_parameter("IN_FRAC_WIDTH"))
        total_channels = self.get_parameter("IN_C")

        # Generate input beats for each channel and each sample
        inputs = []
        for _ in range(total_channels * self.samples):
            inputs.extend(gen_random_matrix_input(*total_tup, *compute_tup, *in_width_tup))

        # Reconstruct the input tensor for the PyTorch model
        depth_dim0 = total_tup[0] // compute_tup[0]
        depth_dim1 = total_tup[1] // compute_tup[1]
        beats_per_channel = depth_dim0 * depth_dim1

        batches = batched(inputs, beats_per_channel)
        matrix_list = [rebuild_matrix(b, *total_tup, *compute_tup) for b in batches]

        # Stack along channel dimension: shape (channels, height, width)
        x = torch.stack(matrix_list, dim=0)
        # Add batch dimension and convert to float for GroupNorm
        x = x.unsqueeze(0).float()

        # Run the PyTorch model
        with torch.no_grad():
            y = self.model(x)

        # Quantize outputs, weights, and biases
        y_q = q2i(y, cfg["out_width"], cfg["out_frac"])
        w_q = q2i(self.model[2].weight, cfg["weight_width"], cfg["weight_frac"])
        if self.model[2].bias is not None:
            b_q = q2i(self.model[2].bias, cfg["bias_width"], cfg["bias_frac"])
        else:
            b_q = torch.zeros(self.get_parameter("OUT_C"), dtype=torch.int)

        # Convert expected output tensor into a list of beats
        y_list = y_q.permute(0, 2, 3, 1).reshape(-1, self.get_parameter("UNROLL_OUT_C")).tolist()
        w_list = w_q.reshape(-1).tolist()
        b_list = b_q.reshape(-1).tolist()

        # Return:
        # - x_list: the integer input beats (driver input)
        # - w_list, b_list: weight and bias for the convolution
        # - y_list: expected output beats
        return inputs, w_list, b_list, y_list

    async def run_test(self):
        await self.reset()
        self.output_monitor.ready.value = 1

        x_list, w_list, b_list, y_list = self.generate_data()
        logger.info(f"Input beats: {x_list}")
        logger.info(f"Weight: {w_list}")
        logger.info(f"Bias: {b_list}")
        logger.info(f"Expected output: {y_list}")

        self.data_in_driver.load_driver(x_list)
        self.weight_driver.load_driver(w_list)
        self.bias_driver.load_driver(b_list)
        self.output_monitor.load_monitor(y_list)

        await Timer(500, "us")
        assert self.output_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test_resblock(dut):
    tb = ResBlockTB(dut, samples=1)
    await tb.run_test()


from mase_cocotb.runner import mase_runner
import pytest

@pytest.mark.dev
def get_resblock_config():
    config = {
        "DATA_IN_0_PRECISION_0": 8,
        "DATA_IN_0_PRECISION_1": 4,
        "WEIGHT_PRECISION_0": 8,
        "WEIGHT_PRECISION_1": 4,
        "BIAS_PRECISION_0": 8,
        "BIAS_PRECISION_1": 4,
        "IN_X": 3,
        "IN_Y": 2,
        "IN_C": 4,
        "UNROLL_IN_C": 4,
        "KERNEL_X": 2,
        "KERNEL_Y": 2,
        "OUT_C": 4,
        "UNROLL_KERNEL_OUT": 4,
        "UNROLL_OUT_C": 4,
        "BIAS_SIZE": 4,
        "STRIDE": 1,
        "PADDING_Y": 1,
        "PADDING_X": 2,
        "HAS_BIAS": 1,
        "DATA_OUT_0_PRECISION_0": 8,
        "DATA_OUT_0_PRECISION_1": 4,
        # Parameters for group_norm_2d
        "TOTAL_DIM0": 4,
        "TOTAL_DIM1": 4,
        "COMPUTE_DIM0": 2,
        "COMPUTE_DIM1": 2,
        "GROUP_CHANNELS": 2,
        "IN_WIDTH": 8,
        "IN_FRAC_WIDTH": 4,
        "OUT_WIDTH": 8,
        "OUT_FRAC_WIDTH": 4,
        "ISQRT_LUT_POW": 5,
    }

    # Automatically calculate SLIDING_NUM based on convolution parameters
    in_y, in_x = config["IN_Y"], config["IN_X"]
    ky, kx = config["KERNEL_Y"], config["KERNEL_X"]
    py, px = config["PADDING_Y"], config["PADDING_X"]
    stride = config["STRIDE"]
    out_height = ceil((in_y - ky + 2 * py + 1) / stride)
    out_width = ceil((in_x - kx + 2 * px + 1) / stride)
    config["SLIDING_NUM"] = out_height * out_width

    return config

def test_resblock():
    mase_runner(
        module="resblock",  # Verilog top-level module name
        seed=42,
        trace=True,
        module_param_list=[
            get_resblock_config(),
        ]
    )

if __name__ == "__main__":
    test_resblock()
