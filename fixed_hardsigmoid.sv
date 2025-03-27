`timescale 1ns / 1ps

module fixed_hardsigmoid #(
    /* verilator lint_off UNUSEDPARAM */
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

    parameter INPLACE = 0
)(
    input logic clk,
    input logic rst,

    input  logic signed [DATA_IN_0_PRECISION_0-1:0]  data_in_0 [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1 - 1:0],
    output logic signed [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1 - 1:0],

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);

  localparam signed [DATA_IN_0_PRECISION_0-1:0] CONST_NEG_3 = -3 <<< DATA_IN_0_PRECISION_1;
  localparam signed [DATA_IN_0_PRECISION_0-1:0] CONST_POS_3 =  3 <<< DATA_IN_0_PRECISION_1;
  localparam signed [DATA_IN_0_PRECISION_0-1:0] ONE_Q       =  1 <<< DATA_IN_0_PRECISION_1;
  localparam signed [DATA_IN_0_PRECISION_0-1:0] HALF_Q      =  1 <<< (DATA_IN_0_PRECISION_1 - 1);

  // 1/6 ≈ 43 / 256
  localparam int MUL_NUM   = 43;
  localparam int MUL_SHIFT = 8;

  for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; i++) begin : hardsigmoid_loop
    always_comb begin
      if (data_in_0[i] <= CONST_NEG_3) begin
        data_out_0[i] = '0;
      end else if (data_in_0[i] >= CONST_POS_3) begin
        data_out_0[i] = ONE_Q;
      end else begin
        data_out_0[i] = ((data_in_0[i] * MUL_NUM + (1 << (MUL_SHIFT - 1))) >>> MUL_SHIFT) + HALF_Q;
      end
    end
  end

  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready  = data_out_0_ready;

endmodule
