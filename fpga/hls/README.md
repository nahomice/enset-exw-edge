# fpga/hls/ -- Intel HLS feasibility test (do this first, takes about an hour)

Open question: is the Intel HLS Compiler usable for **Cyclone V** with **Quartus 23.1**? To my knowledge the Standard
edition of Intel HLS stopped at Quartus 19.1 and the Pro edition does not target Cyclone V, but I could not verify this
from a primary source. Settle it with a toy component.

1. Check the installation: `which i++` and `i++ --version` in the Quartus/HLS environment, or look for an `hls` folder
   in the Quartus install directory.
2. Compile the toy for the host (x86): `i++ toy_mac/toy_mac.cpp -o toy_x86 && ./toy_x86`  (prints PASS).
3. Compile for the device: `i++ -march=5CSEMA5F31C6 toy_mac/toy_mac.cpp -o toy_fpga --simulator none`.
4. Write the result below (version, command, exact error if any) and decide.

## Decision record
| Date | Quartus / HLS version | Command | Result | Decision |
|---|---|---|---|---|
| | | | | HLS / SystemVerilog |

If HLS is not available or not usable: the RTL in `fpga/rtl/` (SystemVerilog) is the primary path and nothing else
changes; the golden model and test vectors are language independent.
