`timescale 10ns / 1ps

module resblock #( 
    parameter DATA_IN_0_PRECISION_0 =8, //Integer part of input feature map
    parameter DATA_IN_0_PRECISION_1 = 4, // Fractional part of input feature map
    parameter WEIGHT_PRECISION_0    = 8, //Integer part of weights
    parameter WEIGHT_PRECISION_1    = 4, //Fractional part of weights
    parameter BIAS_PRECISION_0      = 8, //Integer part of bias
    parameter BIAS_PRECISION_1      = 4, //Fractional part of bias

//Input feature map size (3x2) with 4 channels
    parameter IN_X    = 3,
    parameter IN_Y   = 2,
    parameter IN_C = 4,
    parameter UNROLL_IN_C = 4,//Number of input channels processed in parallel

//2x2 convolution kernel
    parameter KERNEL_X = 2,
    parameter KERNEL_Y = 2,
    parameter OUT_C = 4, //Output channels

    parameter UNROLL_KERNEL_OUT = 4, //Number of kernel calculations performed in parallel
    parameter UNROLL_OUT_C = 4, //Number of output channels processed in parallel

    parameter SLIDING_NUM = 8, //Number of sliding window iterations

    parameter BIAS_SIZE = UNROLL_OUT_C, //Bias array size
    parameter STRIDE    = 1, //Convolution stride

    parameter PADDING_Y = 1, //Zero-padding on edges
    parameter PADDING_X = 2,
    parameter HAS_BIAS  = 1, //Includes bias term in computation

    parameter DATA_OUT_0_PRECISION_0 = 8, //Integer part of output feature map
    parameter DATA_OUT_0_PRECISION_1 = 4,  //Fractional part of output feature map

    parameter TOTAL_DIM0     = 4,
    parameter TOTAL_DIM1     = 4,
    parameter COMPUTE_DIM0   = 2,
    parameter COMPUTE_DIM1   = 2,
    parameter GROUP_CHANNELS = 2,

    // Data widths
    parameter IN_WIDTH       = 8,
    parameter IN_FRAC_WIDTH  = 4,
    parameter OUT_WIDTH      = 8,
    parameter OUT_FRAC_WIDTH = 4,

    // Inverse Sqrt LUT
    parameter ISQRT_LUT_MEMFILE = "",
    parameter ISQRT_LUT_POW     = 5
)(
    input clk,
    input rst,

    // input  [DATA_IN_0_PRECISION_0 - 1:0] data_in_0      [UNROLL_IN_C - 1 : 0],
    // input                                data_in_0_valid,
    // output                               data_in_0_ready,

    input  logic [IN_WIDTH-1:0] data_in_0 [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic                data_in_0_valid,
    output logic                data_in_0_ready,

    input  [WEIGHT_PRECISION_0-1:0] weight      [UNROLL_KERNEL_OUT * UNROLL_OUT_C -1:0],
    input                           weight_valid,
    output                          weight_ready,

    input  [BIAS_PRECISION_0-1:0] bias      [BIAS_SIZE-1:0],
    input                         bias_valid,
    output                        bias_ready,

    output [DATA_OUT_0_PRECISION_0 - 1:0] data_out_0[COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    output                                data_out_0_valid,
    input                                 data_out_0_ready
);

    // // Signals for inter-module communication
    logic [IN_WIDTH-1:0] norm_out_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
    logic norm_out_valid;
    logic norm_out_ready;

    // logic [OUT_WIDTH-1:0] conv_out_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
    // logic conv_out_valid;
    // logic conv_out_ready;

    // // Instantiate Group Normalization Module First
    // group_norm_2d norm_inst (
    //     .clk(clk),
    //     .rst(rst),
    //     .in_data(/* Provide input data here */),
    //     .in_valid(/* Provide valid signal */),
    //     .in_ready(/* Capture ready signal */),
    //     .out_data(norm_out_data),
    //     .out_valid(norm_out_valid),
    //     .out_ready(norm_out_ready)
    // );
    
    group_norm_2d #(
    // Dimensions
    .TOTAL_DIM0(TOTAL_DIM0),
    .TOTAL_DIM1(TOTAL_DIM1),
    .COMPUTE_DIM0(COMPUTE_DIM0),
    .COMPUTE_DIM1(COMPUTE_DIM1),
    .GROUP_CHANNELS(GROUP_CHANNELS),

    // Data widths
    .IN_WIDTH(IN_WIDTH),
    .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
    .OUT_WIDTH(OUT_WIDTH),
    .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),

    // Inverse Sqrt LUT
    .ISQRT_LUT_MEMFILE(ISQRT_LUT_MEMFILE),
    .ISQRT_LUT_POW(ISQRT_LUT_POW)
    ) norm_0(
    .clk(clk),
    .rst(rst),

    .in_data(data_in_0),
    .in_valid(data_in_0_valid),
    .in_ready(data_in_0_ready),

    .out_data(norm_out_data),
    .out_valid(norm_out_valid),
    .out_ready(norm_out_ready)
    );


    logic [DATA_OUT_0_PRECISION_0 - 1:0] mish_out_0[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
    logic                            mish_out_0_valid;
    logic                            mish_out_0_ready;

    fixed_mish #(
        /* verilator lint_off UNUSEDPARAM */
        .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
        .DATA_IN_0_PRECISION_1(DATA_IN_0_PRECISION_1),
        .DATA_IN_0_PARALLELISM_DIM_0(COMPUTE_DIM0),
        .DATA_IN_0_PARALLELISM_DIM_1(COMPUTE_DIM1),
        .DATA_OUT_0_PRECISION_0(DATA_IN_0_PRECISION_0),
        .DATA_OUT_0_PRECISION_1(DATA_IN_0_PRECISION_1),
        .DATA_OUT_0_PARALLELISM_DIM_0(COMPUTE_DIM0),
        .DATA_OUT_0_PARALLELISM_DIM_1(COMPUTE_DIM1)
    
    ) mish_0(
        /* verilator lint_off UNUSEDSIGNAL */
        .rst,
        .clk,
        .data_in_0(norm_out_data),
        .data_out_0(mish_out_0),
    
        .data_in_0_valid(norm_out_valid),
        .data_in_0_ready(norm_out_ready),
        .data_out_0_valid(mish_out_0_valid),
        .data_out_0_ready(mish_out_0_ready)
    );



     //Instantiate Convolution Module Following Group Normalization
     convolution #(
      
     .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
     .DATA_IN_0_PRECISION_1(DATA_IN_0_PRECISION_1),
     .WEIGHT_PRECISION_0(WEIGHT_PRECISION_0),
     .WEIGHT_PRECISION_1(WEIGHT_PRECISION_1),
     .BIAS_PRECISION_0(BIAS_PRECISION_0),
     .BIAS_PRECISION_1(BIAS_PRECISION_1),

     .IN_X(IN_X),
     .IN_Y(IN_Y),
     .IN_C(IN_C),
     .UNROLL_IN_C(UNROLL_IN_C),

     .KERNEL_X(KERNEL_X),
     .KERNEL_Y(KERNEL_Y),
     .OUT_C(OUT_C),

     .UNROLL_KERNEL_OUT(UNROLL_KERNEL_OUT),
     .UNROLL_OUT_C(UNROLL_OUT_C),

     .SLIDING_NUM(SLIDING_NUM),

     .BIAS_SIZE(BIAS_SIZE),
     .STRIDE(STRIDE),

     .PADDING_Y(PADDING_Y),
     .PADDING_X(PADDING_X),
     .HAS_BIAS(HAS_BIAS),

     .DATA_OUT_0_PRECISION_0(DATA_OUT_0_PRECISION_0),
     .DATA_OUT_0_PRECISION_1(DATA_OUT_0_PRECISION_1)
 ) conv_0 (
     .clk(clk),
     .rst(rst),

     .data_in_0(mish_out_0),
     .data_in_0_valid(mish_out_valid),
     .data_in_0_ready(mish_out_ready),

     .weight(weight),
     .weight_valid(weight_valid),
     .weight_ready(weight_ready),

     .bias(bias),
     .bias_valid(bias_valid),
     .bias_ready(bias_ready),

    .data_out_0(data_out_0),
     .data_out_0_valid(data_out_0_valid),
     .data_out_0_ready(data_out_0_ready)
     );
endmodule
