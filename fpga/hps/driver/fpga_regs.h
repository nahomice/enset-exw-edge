/* fpga_regs.h -- control register map of the tile-CNN accelerator (PROPOSAL, finalise with the RTL).
 * Base address: lightweight HPS-to-FPGA bridge, 0xFF200000 on Cyclone V SoC (check the Platform Designer address map).
 * All registers are 32 bit. Keep in sync with fpga/docs/interface_spec.md. */
#ifndef FPGA_REGS_H
#define FPGA_REGS_H

#define EXW_LWH2F_BASE   0xFF200000u
#define EXW_REG_VERSION  0x00u   /* RO  RTL version (0x0001_0000 = v1.0) */
#define EXW_REG_CTRL     0x04u   /* RW  bit0 START (self-clearing), bit1 SOFT_RESET */
#define EXW_REG_STATUS   0x08u   /* RO  bit0 BUSY, bit1 DONE, bit2 ERROR */
#define EXW_REG_TILE_ADDR 0x0Cu  /* RW  physical DDR3 address of the first int8 tile (96*96*3 bytes each) */
#define EXW_REG_N_TILES  0x10u   /* RW  number of tiles in the batch */
#define EXW_REG_OUT_ADDR 0x14u   /* RW  physical DDR3 address for results (3 int8 logits per tile, padded to 4 bytes) */
#define EXW_REG_CYCLES   0x18u   /* RO  clock cycles of the last batch (for latency and energy-per-tile) */

#define EXW_CTRL_START       (1u << 0)
#define EXW_CTRL_SOFT_RESET  (1u << 1)
#define EXW_STATUS_BUSY      (1u << 0)
#define EXW_STATUS_DONE      (1u << 1)
#define EXW_STATUS_ERROR     (1u << 2)

#endif
