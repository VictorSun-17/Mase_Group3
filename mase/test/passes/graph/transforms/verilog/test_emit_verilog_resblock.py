import sys, os, pytest
import toml

import torch
torch.manual_seed(0)

from chop.ir.graph.mase_graph import MaseGraph

from chop.passes.graph.analysis import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    report_node_type_analysis_pass,
)

from chop.passes.graph.transforms import (
    emit_verilog_top_transform_pass,
    emit_internal_rtl_transform_pass,
    emit_bram_transform_pass,
    emit_cocotb_transform_pass,
    quantize_transform_pass,
)

import chop.passes as passes
import chop.actions as actions
from chop.tools import get_logger, set_excepthook

from mase_components import get_module_dependencies
from mase_components.helper.generate_memory import generate_sv_lut

import operator
from functools import partial

logger = get_logger(__name__)
logger.setLevel("DEBUG")
set_excepthook()

os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]


RESBLOCK_CUSTOM_OPS = {
    "modules": {
        "toolchain": "INTERNAL_RTL",
        "module": "resblock",
        "dependence_files": get_module_dependencies(
                ""
        )
    },
    "functions": {},
}


def swish(x):
    return x*torch.nn.Sigmoid(x)

class ResnetBlock(torch.nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    def forward(self, x):
        x = self.norm1(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = swish(x)
        x = torch.unflatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)



def resblock_module_level_quantize(model, model_config, q_config):
    return model

def resblock_updata_metadata(mg, q_config):
    return mg, {}

def emit_verilog_resblock(
    config,
    q_config,
    config_sequence_length,
    wait_count=15,
    wait_unit="ms",
    max_parallelism=4,
):
    # * Get model and quantize self attention, linear and layer norm layers
    model = ResnetBlock()
    model = resblock_module_level_quantize(model, config, q_config)
    logger.info(f"Quantized Resblock model: {model}")

    # * Trace the model
    mg = MaseGraph(model, custom_ops = RESBLOCK_CUSTOM_OPS)
    mg, _ = passes.init_metadata_analysis_pass(mg)

    mg, _ = passes.report_graph_analysis_pass(mg, pass_args={"file_name": "bert.txt"})

    # * Add metadata analysis passes
    mg, _ = passes.add_common_metadata_analysis_pass(
        mg,
        pass_args={
            "dummy_in": {
                "input_ids": torch.randn(
                    (1, config_sequence_length, config.hidden_size)
                )
            },
            "add_value": False,
        },
    )

    mg, _ = resblock_updata_metadata(mg, q_config)

    mg, _ = passes.add_hardware_metadata_analysis_pass(
        mg,
        pass_args={
            "max_parallelism": [max_parallelism] * 4,
        },
    )

    # * Save the metadata to a file for debugging
    mg, _ = passes.report_node_meta_param_analysis_pass(
        mg,
        pass_args={
            "which": ["common", "hardware"],
            "save_path": "llama_graph_meta_params.txt",
        },
    )

    mg, _ = passes.emit_verilog_top_transform_pass(mg)
    mg, _ = passes.emit_bram_transform_pass(mg)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg)
    mg, _ = passes.emit_cocotb_transform_pass(
        mg,
        pass_args={
            "wait_time": wait_count,
            "wait_unit": wait_unit,
        },
    )
    mg, _ = passes.emit_vivado_project_transform_pass(mg)

     # Temporary: fix data coherency checks
    os.environ["COCOTB_RESOLVE_X"] = "ZEROS"

    actions.simulate(
        skip_build=False, skip_test=False, gui=False, waves=False, simulator="questa"
    )
