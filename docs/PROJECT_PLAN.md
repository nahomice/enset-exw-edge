# Project plan and milestones

Mid-year defense = **simulation results**. Week numbers count from the start of implementation; map them to calendar
dates once the defense date is fixed. **[EDIT: defense date]**

| Milestone | Target | Deliverable (checkable) | Status |
|---|---|---|---|
| **M0 Repo and pipeline** | week 1-2 | repo, synthetic data, tiling, grouped splits, tile CNN, INT8, golden model, 13 passing tests | code done; run `make smoke` |
| **M1 Data request** | week 1-4 | message sent to EAA / partner offices; first Tier-1 healthy/background photos from field edge; capture protocol agreed | open |
| **M2 Simulation results** | week 4-8 | FP32 vs INT8 on synthetic + any real tiles; golden == TFLite reference; heat maps; lab-CPU latency; param/ops count | after M0 |
| **M3 FPGA simulation** | week 6-12 | RTL for conv/depthwise/pool/FC; ModelSim/Questa testbench reproduces golden test vectors; resource estimate | needs HLS-vs-SystemVerilog decision |
| **M4 Mid-year defense** | week 12 | slides, demo (heat map on a scene), honest limits slide | |
| M5 Phone app | after defense | Android app on Samsung Galaxy A13, TFLite INT8, tile loop + overlay | |
| M6 Board and energy | after defense | DE1-SoC runs: ARM INT8 baseline, FPGA accelerator, inline DC meter energy | |
| M7 User needs study | in parallel | short interviews with agricultural office staff (who uses it, what output they need) | |

## Decisions still open
1. **HLS or SystemVerilog.** Intel HLS Compiler support for Cyclone V on Quartus 23.1 is unverified (HLS Standard was
   dropped after Quartus 19.1 to my knowledge). Test with a 20-line example **this week**; fallback is SystemVerilog.
2. **Real data source and permission** (EAA / partners): who owns the images, labeling expert, ethics/permission.
3. **Tile size vs plant size:** choose 96 px only after seeing real photo resolution (see DATA_PROTOCOL).
4. **QAT** only if INT8 loses more than 2 percentage points macro-F1 (`quant.max_drop_pp`).

## Division of work (suggested, edit)
- Nahom: data pipeline, training, quantization, evaluation, phone benchmark.
- Yoseph: FPGA design from the golden model, DE1-SoC baselines, energy measurement.
- Both: data collection requests, reviews of each other's pull requests, defense slides.
