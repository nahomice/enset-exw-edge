# Risk register

| # | Risk | Likelihood | Impact | Mitigation | Owner | Status |
|---|---|---|---|---|---|---|
| R1 | No real wide-view EXW images in time | high | high | request EAA/partners now; Tier-0 synthetic keeps pipeline and hardware work moving; claim only what the data supports | both | open |
| R2 | Plants too small in the photo for the tile size | medium | high | capture protocol (distance, resolution); tile-size ablation X1; notebook 02 to check | Nahom | open |
| R3 | Intel HLS unavailable for Cyclone V on 23.1 | medium | medium | one-hour test; SystemVerilog fallback already scaffolded | Yoseph | open |
| R4 | FPGA design does not fit or is slower than the ARM/phone | medium | medium | report honestly; the thesis claim is measured comparison, not speed-up | Yoseph | open |
| R5 | INT8 loses more than 2 pp | low | medium | more calibration tiles, then QAT | Nahom | open |
| R6 | Look-alikes (other diseases, drought, weeds) cause false alarms | high | high | "other condition" class, healthy-farm images, farm-level threshold on validation | Nahom | open |
| R7 | Domain shift between farms/phones | high | high | grouped splits, held-out site/style test, multiple phones if possible | Nahom | open |
| R8 | Data permission or labelling expert unavailable | medium | high | permission record (`data/schema/permission_template.md`) before collecting | both | open |
| R9 | Board or toolchain access limited (lab hours, licence) | medium | medium | simulate first; book lab time early | Yoseph | open |
