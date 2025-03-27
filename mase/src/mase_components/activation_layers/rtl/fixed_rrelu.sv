`timescale 1ns / 1ps

module fixed_rrelu #(
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 3,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1,

    parameter INPLACE = 0,

    // Added: fixed slope (Qm.n format, same as data, e.g., 3.5 bits = 8-bit)
    parameter [DATA_IN_0_PRECISION_0-1:0] NEGATIVE_SLOPE = 8'd34 // ~0.265 in Q3.5 format
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
    else $error("RReLU: Input and output precisions must match");

    assert (DATA_IN_0_PRECISION_1 == DATA_OUT_0_PRECISION_1)
    else $error("RReLU: Input and output Q point must match");
  end

  localparam TOTAL_PARALLELISM = DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1;

  for (genvar i = 0; i < TOTAL_PARALLELISM; i++) begin : RReLU
    always_comb begin
      if ($signed(data_in_0[i]) <= 0) begin
        // Apply negative slope (fixed-point mult, result width = 2x original)
        logic signed [2*DATA_IN_0_PRECISION_0-1:0] product;
        product = $signed(data_in_0[i]) * $signed(NEGATIVE_SLOPE);
        // Shift right by fractional bits to get final fixed-point result
        data_out_0[i] = product[DATA_IN_0_PRECISION_0+DATA_IN_0_PRECISION_1-1 -: DATA_IN_0_PRECISION_0];
      end else begin
        data_out_0[i] = data_in_0[i];
      end
    end
  end

  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready  = data_out_0_ready;

endmodule