/* run_batch.c -- SKELETON host program for the HPS (ARM Linux). Not yet compiled or run.
 * Build on the board: gcc -O2 -o run_batch run_batch.c
 * Usage: ./run_batch tiles.bin n_tiles     (tiles.bin = n_tiles * 96*96*3 int8 values, already "pixel - 128")
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "fpga_regs.h"

#define TILE_BYTES (96 * 96 * 3)

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s tiles.bin n_tiles\n", argv[0]); return 1; }
  int n = atoi(argv[2]);
  (void)n;
  /* TODO 1: open("/dev/mem"), mmap the lightweight bridge at EXW_LWH2F_BASE (page-aligned, 4 KB). */
  /* TODO 2: reserve a physically contiguous DDR3 buffer (reserved-memory in the device tree) and copy the tiles in. */
  /* TODO 3: write TILE_ADDR, N_TILES, OUT_ADDR; set CTRL.START; poll STATUS.DONE with a timeout. */
  /* TODO 4: read CYCLES, copy logits out, print ms per tile = CYCLES / f_clk / n; compare with the golden logits. */
  fprintf(stderr, "not implemented yet (tile size %d bytes)\n", TILE_BYTES);
  return 2;
}
