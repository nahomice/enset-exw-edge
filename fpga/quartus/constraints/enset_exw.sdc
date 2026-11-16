# 50 MHz board clock (20 ns). Tighten only after the design closes timing; record achieved Fmax in docs/resource_budget.md.
create_clock -name CLOCK_50 -period 20.000 [get_ports CLOCK_50]
derive_pll_clocks
derive_clock_uncertainty
set_false_path -from [get_ports KEY0_N]
