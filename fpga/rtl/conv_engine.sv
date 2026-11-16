// conv_engine.sv -- SKELETON. Shared engine for CONV_2D (3x3 and 1x1) and DEPTHWISE_CONV_2D.
//
// TODO (Yoseph):
//   1. Line buffers / window generator with SAME padding (padding contributes ZERO after zero-point subtraction;
//      pad_before = total/2 with total = max((out-1)*stride + k - in, 0), exactly as exw.golden._same_pad).
//   2. N parallel mac_unit lanes (one per output channel group); weights from ROM/M10K loaded from L*_w.mem.
//   3. requantize per output channel (mult/shift from L*_mult.mem / L*_shift.mem), `single_round` = 0 here.
//   4. Control FSM driven by model_spec.json (layer sizes, stride, zero points, clamp range).
// Acceptance test: tb_tilecnn_top reproduces vectors/tile_XX_L<k>.mem for the layer, bit for bit.
module conv_engine #(
  parameter int MAX_CH = 256
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  input  logic [1:0]  mode,            // 0 = conv 3x3, 1 = conv 1x1, 2 = depthwise 3x3
  input  logic [8:0]  in_h, in_w,
  input  logic [8:0]  in_ch, out_ch,
  input  logic [1:0]  stride,
  output logic        busy,
  output logic        done
);
  // placeholder so the project elaborates; replace with the real datapath
  assign busy = 1'b0;
  assign done = start;
endmodule
