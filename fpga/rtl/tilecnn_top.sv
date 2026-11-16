// tilecnn_top.sv -- SKELETON. Top level of the tile classifier accelerator.
// Input: one 96x96x3 tile as int8 (= pixel - 128), streamed in. Output: 3 int8 logits (EXW-suspect, other, healthy).
// Layer sequence (see docs/ARCHITECTURE.md and results/*/fpga_export/model_spec.json):
//   conv3x3/2 -> 4 x [depthwise 3x3 -> pointwise 1x1] -> avgpool 6x6 -> FC.
//
// TODO (Yoseph): layer sequencer FSM, ping-pong activation buffers (M10K), weight ROMs, HPS register interface
// (see fpga/docs/interface_spec.md), cycle counter for benchmarking.
module tilecnn_top (
  input  logic              clk,
  input  logic              rst_n,
  input  logic              start,
  input  logic              in_valid,
  input  logic signed [7:0] in_data,
  output logic              busy,
  output logic              done,
  output logic signed [7:0] logits [3]
);
  conv_engine u_conv (
    .clk, .rst_n, .start, .mode(2'd0), .in_h(9'd96), .in_w(9'd96), .in_ch(9'd3), .out_ch(9'd16),
    .stride(2'd2), .busy(), .done()
  );
  avgpool_fc u_head (.clk, .rst_n, .start, .logits, .done);
  assign busy = 1'b0;
endmodule
