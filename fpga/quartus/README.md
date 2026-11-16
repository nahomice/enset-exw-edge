# fpga/quartus/

Quartus Prime Lite 23.1 project for the DE1-SoC.

| File | Purpose |
|---|---|
| `enset_exw.qpf` / `.qsf` | project and settings (device, sources, pins) |
| `constraints/enset_exw.sdc` | clock constraints |
| `output_files/` | build products (git-ignored): `.sof` bitstream and reports |

Steps: open `enset_exw.qpf` -> Processing -> Start Compilation -> Tools -> Programmer (USB-Blaster) -> load the `.sof`.

Before first use: **check the three pin assignments** against the DE1-SoC user manual. A wrong pin can fight with
another board function.

After every compile copy the resource lines (ALMs, DSP blocks, M10K bits, Fmax at 85C slow model) into
`fpga/docs/resource_budget.md`. Resource and speed claims in the thesis come from those reports, not from estimates.
Platform Designer (Qsys) system for the HPS-FPGA link goes in `fpga/hps/` (add the generated `.qsys` here when it exists).
