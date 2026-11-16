`timescale 1ns/1ps
// SKELETON: end-to-end test. Feed vectors/tile_00_input.mem, dump every layer output to out/tile_00_L<k>.mem
// (same format: one 2-digit hex byte per line), then run `python scripts/compare_rtl_outputs.py`.
// The last layer must equal vectors/tile_00_logits.mem.
module tb_tilecnn_top;
  logic clk = 0, rst_n = 0, start = 0, in_valid = 0, busy, done;
  logic signed [7:0] in_data, logits [3];
  logic [7:0] tile_mem [0:96*96*3-1];

  tilecnn_top dut (.clk(clk), .rst_n(rst_n), .start(start), .in_valid(in_valid), .in_data(in_data),
                   .busy(busy), .done(done), .logits(logits));
  always #5 clk = ~clk;

  initial begin
    $readmemh("vectors/tile_00_input.mem", tile_mem);
    repeat (4) @(posedge clk); rst_n = 1;
    // TODO: stream tile_mem into the DUT, wait for done, $writememh the layer buffers into out/
    $display("tb_tilecnn_top: skeleton only (logits = %0d %0d %0d)", logits[0], logits[1], logits[2]);
    $finish;
  end
endmodule
