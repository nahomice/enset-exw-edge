# fpga/hps/ -- ARM side of the DE1-SoC (HPS) and the HPS-FPGA link

```
hps/
  README.md
  driver/fpga_regs.h        register map shared by C code and RTL (mirror of fpga/docs/interface_spec.md)
  driver/run_batch.c        skeleton: load tiles to DDR3, start accelerator, read logits, time it
  arm_baseline/README.md    INT8 inference on the Cortex-A9 only (the fair CPU baseline)
  platform_designer/        add the generated .qsys / .sopcinfo here when it exists
```

Work order
1. Boot the board image from Terasic (Linux on the SD card); confirm `ssh` over Ethernet.
2. ARM baseline first: run the same INT8 `.tflite` and the same tiles on the A9; this is the number the FPGA must beat
   (or not: report it honestly either way).
3. Platform Designer: HPS + lightweight AXI bridge (control registers) + a DMA/mSGDMA or HPS-to-FPGA bridge for tiles.
4. Replace the placeholder in `driver/run_batch.c` with real memory mapping (`/dev/mem` + `mmap`).
