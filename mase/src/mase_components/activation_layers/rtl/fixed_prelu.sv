`timescale 1ns / 1ps

module fixed_prelu #(
    parameter DATA_IN_0_PRECISION_0 = 8,  
    parameter DATA_IN_0_PRECISION_1 = 4,  
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 4,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1,

    // alpha = 0.25 -> 0.25 * 2^4 = 4 (Q4 format)
    parameter signed [DATA_IN_0_PRECISION_0-1:0] ALPHA = 8'sd4
) (
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);

  initial begin
    assert (DATA_IN_0_PRECISION_0 == DATA_OUT_0_PRECISION_0)
    else $error("PReLU: Precision mismatch");
    assert (DATA_IN_0_PRECISION_1 == DATA_OUT_0_PRECISION_1)
    else $error("PReLU: Fractional bits mismatch");
  end

  for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; i++) begin : prelu
    always_comb begin
      if ($signed(data_in_0[i]) <= 0) begin
        // rounding fixed-point multiplication: (x * alpha + 2^(bias - 1)) >> bias
        data_out_0[i] = ($signed(data_in_0[i]) * $signed(ALPHA) + (1 <<< DATA_IN_0_PRECISION_1 - 1)) >>> DATA_IN_0_PRECISION_1;
      end else begin
        data_out_0[i] = data_in_0[i];
      end
    end
  end

  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready  = data_out_0_ready;

endmodule
