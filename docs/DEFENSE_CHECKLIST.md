# Mid-year defense checklist (simulation results)

Show only what is true and checkable.

- [ ] Problem and users: agricultural office staff / extension workers (not farmers directly), wide-view photo, flag areas.
- [ ] Pipeline diagram: tiles -> INT8 CNN -> heat map -> risk score.
- [ ] Data honesty slide: Tier 0 synthetic vs real; what is NOT claimed.
- [ ] Leakage controls: grouped splits, train-only calibration, validation-only threshold.
- [ ] Model: ~50 k parameters, ops list, INT8 size.
- [ ] FP32 vs INT8 table (macro-F1, farm sens/spec with CI) -- labelled synthetic.
- [ ] Golden model == TFLite reference: 0 mismatches over N tiles; +-2 LSB for optimized kernels.
- [ ] Heat-map demo figure (`results/.../figures`).
- [ ] Lab-CPU latency per tile and per image (FP32 vs INT8); phone and board numbers marked "planned".
- [ ] FPGA: RTL simulation passes the exported test vectors (if M3 reached); resource estimate for Cyclone V.
- [ ] Limitations and next steps: real data, QAT if needed, phone app, board energy.
- [ ] Likely questions rehearsed: why not just a phone? why FPGA? what if no real data? cost? who uses it?
