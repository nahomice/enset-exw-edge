`timescale 1ns/1ps
// Checks rtl/requantize.sv against vectors/requant_cases.txt (made by scripts/make_requant_vectors.py from the golden model).
// Run from fpga/sim:  vsim -c -do run_sim.do
module tb_requantize;
  logic signed [31:0] acc, mult;
  logic signed [7:0]  shift, zp, lo, hi, y;
  logic               single;
  requantize dut (.acc(acc), .mult(mult), .shift(shift), .single_round(single),
                  .zp_out(zp), .clamp_lo(lo), .clamp_hi(hi), .y(y));

  integer fd, r, n, errors;
  integer a_i, m_i, s_i, sg_i, z_i, l_i, h_i, e_i;

  initial begin
    fd = $fopen("vectors/requant_cases.txt", "r");
    if (fd == 0) begin
      $display("FATAL: vectors/requant_cases.txt not found (run: make requant-vectors)");
      $finish;
    end
    n = 0; errors = 0;
    while (!$feof(fd)) begin
      r = $fscanf(fd, "%d %d %d %d %d %d %d %d\n", a_i, m_i, s_i, sg_i, z_i, l_i, h_i, e_i);
      if (r == 8) begin
        acc = a_i; mult = m_i; shift = s_i; single = sg_i[0]; zp = z_i; lo = l_i; hi = h_i;
        #1;
        n = n + 1;
        if (integer'(y) != e_i) begin
          errors = errors + 1;
          if (errors <= 10)
            $display("MISMATCH case %0d: acc=%0d mult=%0d shift=%0d single=%0d -> rtl %0d, golden %0d",
                     n, a_i, m_i, s_i, sg_i, integer'(y), e_i);
        end
      end
    end
    $display("tb_requantize: %0d cases, %0d mismatches -> %s", n, errors, (errors == 0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
