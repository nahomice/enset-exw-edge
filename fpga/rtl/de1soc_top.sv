// de1soc_top.sv -- board wrapper for the Terasic DE1-SoC. Currently a heartbeat LED so the project builds and
// the board can be programmed; the accelerator is attached through Platform Designer (fpga/hps/).
module de1soc_top (
  input  logic CLOCK_50,
  input  logic KEY0_N,       // active-low push button, used as reset
  output logic LED0
);
  logic [25:0] cnt;
  always_ff @(posedge CLOCK_50) begin
    if (!KEY0_N) cnt <= '0;
    else         cnt <= cnt + 1'b1;
  end
  assign LED0 = cnt[25];     // ~0.75 Hz blink at 50 MHz
endmodule
