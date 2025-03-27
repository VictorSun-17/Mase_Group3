`timescale 1ns/1ps

module fixed_swish #(
    parameter DATA_IN_0_PRECISION_0         = 8,
    parameter DATA_IN_0_PRECISION_1         = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0   = 10,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1   = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0   = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1   = 1,

    parameter IN_0_DEPTH = $rtoi($ceil(
        DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0
    )),

    parameter DATA_OUT_0_PRECISION_0        = 8,
    parameter DATA_OUT_0_PRECISION_1        = 4,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0  = 10,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1  = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0  = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1  = 1,

    parameter IN_PLACE = 0
)(
    input  logic rst,
    input  logic clk,

    // Input parallel data vector x
    input  logic [DATA_IN_0_PRECISION_0-1:0]
                data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],

    // Output parallel data vector: swish(x) = x * sigmoid(x)
    output logic [DATA_OUT_0_PRECISION_0+DATA_OUT_0_PRECISION_0-1:0]
                data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],

    // Handshake signals
    input  logic data_in_0_valid,
    output logic data_in_0_ready,

    output logic data_out_0_valid,
    input  logic data_out_0_ready
);


    logic [DATA_IN_0_PRECISION_0-1:0] sigmoid_data [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];
    logic sigmoid_data_valid;
    logic sigmoid_data_ready;

    logic data_in_0_ready_sigmoid;

    fixed_sigmoid #(
        .DATA_IN_0_PRECISION_0       (DATA_IN_0_PRECISION_0),
        .DATA_IN_0_PRECISION_1       (DATA_IN_0_PRECISION_1),
        .DATA_IN_0_TENSOR_SIZE_DIM_0 (DATA_IN_0_TENSOR_SIZE_DIM_0),
        .DATA_IN_0_TENSOR_SIZE_DIM_1 (DATA_IN_0_TENSOR_SIZE_DIM_1),
        .DATA_IN_0_PARALLELISM_DIM_0 (DATA_IN_0_PARALLELISM_DIM_0),
        .DATA_IN_0_PARALLELISM_DIM_1 (DATA_IN_0_PARALLELISM_DIM_1),
        .IN_0_DEPTH                  (IN_0_DEPTH),

        // Assuming sigmoid output uses the same precision as input
        .DATA_OUT_0_PRECISION_0      (DATA_IN_0_PRECISION_0),
        .DATA_OUT_0_PRECISION_1      (DATA_IN_0_PRECISION_1),
        .DATA_OUT_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
        .DATA_OUT_0_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1),
        .DATA_OUT_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
        .DATA_OUT_0_PARALLELISM_DIM_1(DATA_IN_0_PARALLELISM_DIM_1)
    ) sigmoid_inst (
        .clk             (clk),
        .rst             (rst),

        .data_in_0       (data_in_0),
        .data_in_0_valid (data_in_0_valid),
        .data_in_0_ready (data_in_0_ready_sigmoid),

        .data_out_0      (sigmoid_data),
        .data_out_0_valid(sigmoid_data_valid),
        .data_out_0_ready(sigmoid_data_ready)
    );

    logic [DATA_IN_0_PRECISION_0-1:0] ff_data [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];
    logic ff_data_valid;
    logic ff_data_ready;

    logic data_in_0_ready_fifo;

    unpacked_fifo #(
        .DEPTH     (IN_0_DEPTH),
        .DATA_WIDTH(DATA_IN_0_PRECISION_0),
        .IN_NUM    (DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1)
    ) x_buffer (
        .clk           (clk),
        .rst           (rst),

        .data_in       (data_in_0),
        .data_in_valid (data_in_0_valid),
        .data_in_ready (data_in_0_ready_fifo),

        .data_out      (ff_data),
        .data_out_valid(ff_data_valid),
        .data_out_ready(ff_data_ready)
    );

    assign data_in_0_ready = data_in_0_ready_fifo & data_in_0_ready_sigmoid;


    fixed_vector_mult #(
        .IN_WIDTH    (DATA_IN_0_PRECISION_0),
        .WEIGHT_WIDTH(DATA_IN_0_PRECISION_0),
        .IN_SIZE     (DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1)
    ) swish_mult (
        .clk           (clk),
        .rst           (rst),

        // x input from FIFO
        .data_in       (ff_data),
        .data_in_valid (ff_data_valid),
        .data_in_ready (ff_data_ready),

        // Weight input: sigmoid(x)
        .weight        (sigmoid_data),
        .weight_valid  (sigmoid_data_valid),
        .weight_ready  (sigmoid_data_ready),

        // Output: swish(x)
        .data_out      (data_out_0),
        .data_out_valid(data_out_0_valid),
        .data_out_ready(data_out_0_ready)
    );

endmodule
