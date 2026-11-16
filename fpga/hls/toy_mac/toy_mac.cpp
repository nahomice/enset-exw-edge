// toy_mac.cpp -- 16-element INT8 dot product as an HLS component. Only purpose: test the tool flow.
// (May need small edits for the installed HLS version; the header names below are from Intel HLS Standard.)
#include <stdio.h>
#include "HLS/hls.h"

component int dot16(const signed char *a, const signed char *b) {
  int acc = 0;
  #pragma unroll 16
  for (int i = 0; i < 16; i++) acc += (int)a[i] * (int)b[i];
  return acc;
}

int main() {
  signed char a[16], b[16];
  int ref = 0;
  for (int i = 0; i < 16; i++) { a[i] = (signed char)(i * 7 - 50); b[i] = (signed char)(90 - i * 11); ref += a[i] * b[i]; }
  int got = dot16(a, b);
  printf("dot16 = %d, expected %d -> %s\n", got, ref, got == ref ? "PASS" : "FAIL");
  return got == ref ? 0 : 1;
}
