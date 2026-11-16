// avgpool_fc.sv -- SKELETON. AVERAGE_POOL_2D (6x6 -> 1x1) followed by FULLY_CONNECTED (256 -> 3).
//
// Pool, as in exw.golden: acc = sum of the 36 int8 values per channel; avg = (acc + 18)/36 for acc > 0 and
// -((-acc + 18)/36) otherwise (round half away from zero, truncating division). Input and output scale are equal.
// FC: acc32 = sum_i (x_i - zp_in) * w_ci + bias_c ; requantize with single_round = 1 ; NO ReLU clamp (-128..127).
//
// TODO (Yoseph): implement with one mac_unit per class (3 lanes) and the requantize block.
module avgpool_fc (
  input  logic              clk,
  input  logic              rst_n,
  input  logic              start,
  output logic signed [7:0] logits [3],
  output logic              done
);
  assign logits[0] = '0;
  assign logits[1] = '0;
  assign logits[2] = '0;
  assign done      = start;
endmodule
