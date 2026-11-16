# Resource budget (Cyclone V 5CSEMA5F31C6)

Device limits: **verify from the Terasic DE1-SoC datasheet and Intel device handbook** before quoting. The proposal used
about 85 K LE (32 070 ALMs), 87 DSP blocks and about 4 450 Kb of embedded memory.

## Model needs (from the golden model)
| Item | Value | How computed |
|---|---|---|
| Parameters | about 49 800 (int8 weights, about 49.8 KB) | `count_params` |
| Largest activation map | 48 x 48 x 32 = 73 728 bytes | after the first pointwise layer |
| MACs per tile | fill in | sum over layers of out_h * out_w * k * k * in_ch * out_ch |

## Quartus results (fill after every compile)
| Date | Commit | MAC lanes | Fmax (MHz) | ALMs | DSP | M10K bits | Cycles/tile (sim) | Notes |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
