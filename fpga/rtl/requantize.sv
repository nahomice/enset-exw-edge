// requantize.sv -- INT32 accumulator -> INT8 activation, bit-exact with exw.golden.requantize.
//
//   single_round = 0 : conv / depthwise layers  (SaturatingRoundingDoublingHighMul, then RoundingDivideByPOT)
//   single_round = 1 : final per-channel FULLY_CONNECTED layer (one rounding: round-half-away of acc*mult / 2^(31-shift))
//
// REFERENCE implementation for simulation: 64-bit combinational arithmetic, written for clarity not timing.
// Pipelining and narrowing the multiplier to DSP-friendly widths is a later step; keep tb_requantize passing.
// `shift` follows the TFLite convention: shift > 0 is a left shift of acc, shift < 0 a right shift after the multiply.
module requantize (
  input  logic signed [31:0] acc,
  input  logic signed [31:0] mult,
  input  logic signed [7:0]  shift,
  input  logic               single_round,
  input  logic signed [7:0]  zp_out,
  input  logic signed [7:0]  clamp_lo,
  input  logic signed [7:0]  clamp_hi,
  output logic signed [7:0]  y
);

  function automatic longint signed srdhm(input longint signed x, input longint signed m);
    longint signed ab, nudge, num, mag;
    ab    = x * m;
    nudge = (ab >= 0) ? (64'sd1 <<< 30) : (64'sd1 - (64'sd1 <<< 30));
    num   = ab + nudge;
    mag   = (num < 0) ? -num : num;
    mag   = mag >>> 31;                       // truncating division by 2^31 (toward zero)
    return (num < 0) ? -mag : mag;
  endfunction

  function automatic longint signed rdbpot(input longint signed x, input int e);
    longint signed mask, rem, thr;
    mask = (64'sd1 <<< e) - 1;
    rem  = x & mask;
    thr  = (mask >>> 1) + ((x < 0) ? 1 : 0);
    return (x >>> e) + ((rem > thr) ? 1 : 0); // round half away from zero
  endfunction

  function automatic longint signed single_r(input longint signed a, input longint signed m, input int total);
    longint signed prod, mag, r;
    prod = a * m;
    mag  = (prod < 0) ? -prod : prod;
    r    = (mag + (64'sd1 <<< (total - 1))) >>> total;
    return (prod < 0) ? -r : r;
  endfunction

  longint signed x, scaled, v;
  int left, right;

  always_comb begin
    left  = (shift > 0) ? int'(shift) : 0;
    right = (shift < 0) ? int'(-shift) : 0;
    x = longint'(acc) <<< left;
    if (x >  64'sd2147483647)  x =  64'sd2147483647;      // saturate to int32 like the C++ reference
    if (x < -64'sd2147483648)  x = -64'sd2147483648;
    if (single_round) scaled = single_r(longint'(acc), longint'(mult), 31 - int'(shift));
    else              scaled = rdbpot(srdhm(x, longint'(mult)), right);
    v = scaled + longint'(zp_out);
    if (v < longint'(clamp_lo)) v = longint'(clamp_lo);
    if (v > longint'(clamp_hi)) v = longint'(clamp_hi);
    y = v[7:0];
  end
endmodule
