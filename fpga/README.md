# fpga/ (Terasic DE1-SoC, Cyclone V 5CSEMA5F31C6, Quartus Prime 23.1)

Role: office batch accelerator and hardware benchmark, **not** the field device. The golden model (`src/exw/golden.py`)
is the specification; the RTL is correct when it reproduces the exported vectors bit for bit.

```
fpga/
  rtl/        SystemVerilog: requantize.sv and mac_unit.sv (done, bit-exact reference), conv_engine.sv, avgpool_fc.sv,
              tilecnn_top.sv (skeletons with TODO lists), de1soc_top.sv (board wrapper)
  tb/         tb_requantize.sv, tb_mac_unit.sv (self-checking), tb_tilecnn_top.sv (skeleton)
  sim/        run_sim.do (ModelSim/Questa), vectors/ (golden test vectors), out/ (RTL dumps for comparison)
  quartus/    enset_exw.qpf/.qsf, constraints/*.sdc  (compile, resource and Fmax reports)
  hls/        Intel HLS feasibility test (toy_mac), decision record
  hps/        ARM side: register header, host program skeleton, ARM INT8 baseline plan, Platform Designer system
  docs/       interface_spec.md, resource_budget.md, benchmark_log.csv
```

## Work order
1. `fpga/hls/`: one-hour test whether Intel HLS works for Cyclone V on 23.1; record the decision.
2. `make smoke` (or train on real data), then `python scripts/export_fpga.py` and `make requant-vectors`; copy the small
   `vectors/*.mem` you need into `fpga/sim/vectors/` (that folder is the only place `.mem` files are committed).
3. Simulate `tb_requantize` and `tb_mac_unit` (run `vsim -c -do run_sim.do` in `fpga/sim`).
4. Build `conv_engine`, then `avgpool_fc`, then `tilecnn_top`; after each, `scripts/compare_rtl_outputs.py`.
5. Compile in Quartus; log resources and Fmax in `docs/resource_budget.md`.
6. HPS: ARM baseline first, then the accelerator link (`hps/`), then energy with the inline DC meter.

Rule: no speed-up is claimed until it is measured on the same tiles with the same protocol
(`hps/arm_baseline/README.md`). Nothing in this folder has been compiled or simulated yet except by reading.
