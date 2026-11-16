# ModelSim / Questa (Intel FPGA edition, installed with Quartus 23.1). Run from the fpga/sim directory:
#   vsim -c -do run_sim.do
# Needs vectors/requant_cases.txt first:  make requant-vectors   (from the repo root)
if {[file exists work]} { vdel -lib work -all }
vlib work
vlog -sv ../rtl/requantize.sv ../rtl/mac_unit.sv ../rtl/conv_engine.sv ../rtl/avgpool_fc.sv ../rtl/tilecnn_top.sv \
        ../tb/tb_requantize.sv ../tb/tb_mac_unit.sv ../tb/tb_tilecnn_top.sv
foreach tb {tb_requantize tb_mac_unit} {
  vsim -c work.$tb -do "run -all; quit -sim"
}
quit -f
