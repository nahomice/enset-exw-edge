`timescale 1ns/1ps
// Self-checking test of mac_unit: random products against a behavioural model.
module tb_mac_unit;
  logic clk = 0, rst_n = 0, load_bias = 0, en = 0;
  logic signed [31:0] bias, acc;
  logic signed [7:0]  a, w, zp;
  mac_unit dut (.clk(clk), .rst_n(rst_n), .load_bias(load_bias), .bias(bias), .en(en),
                .a(a), .w(w), .zp_in(zp), .acc(acc));
  always #5 clk = ~clk;

  integer i, errors;
  longint signed model;

  initial begin
    errors = 0; zp = -128; bias = 1000; a = 0; w = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(negedge clk); load_bias = 1; @(negedge clk); load_bias = 0;
    model = 1000;
    for (i = 0; i < 2000; i = i + 1) begin
      a = $urandom_range(0, 255) - 128;
      w = $urandom_range(0, 255) - 128;
      en = 1;
      @(negedge clk);
      model = model + (integer'(a) - integer'(zp)) * integer'(w);
      @(negedge clk); en = 0;
      if (acc !== model[31:0]) begin
        errors = errors + 1;
        if (errors <= 5) $display("MISMATCH i=%0d rtl=%0d model=%0d", i, acc, model);
      end
    end
    $display("tb_mac_unit: %0d mismatches -> %s", errors, (errors == 0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
