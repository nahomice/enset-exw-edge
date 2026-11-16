// mac_unit.sv -- one INT8 multiply-accumulate lane: acc += (a - zp_in) * w, int32 accumulator.
// `load_bias` initialises the accumulator with the layer bias (int32) before the first product of an output value.
module mac_unit #(
  parameter int ACC_W = 32
) (
  input  logic                     clk,
  input  logic                     rst_n,
  input  logic                     load_bias,
  input  logic signed [ACC_W-1:0]  bias,
  input  logic                     en,
  input  logic signed [7:0]        a,         // activation (int8, as stored)
  input  logic signed [7:0]        w,         // weight (int8)
  input  logic signed [7:0]        zp_in,     // input zero point of the layer
  output logic signed [ACC_W-1:0]  acc
);
  logic signed [8:0] a_off;
  assign a_off = a - zp_in;                   // range -255 .. 255

  always_ff @(posedge clk) begin
    if (!rst_n)          acc <= '0;
    else if (load_bias)  acc <= bias;
    else if (en)         acc <= acc + a_off * w;
  end
endmodule
