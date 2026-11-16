# docs/slides/

Defense decks (Beamer `.tex` + `.pdf`). Keep one folder per defense: `midyear/`, `final/`.

Mid-year storyline (simulation results), 10-12 minutes:
1. Problem and users (agricultural offices; wide-view photos; flag areas for inspection).
2. System: phone primary, FPGA as office batch accelerator and benchmark.
3. Data plan and honesty: Tier 0 synthetic now, real tiers requested; what is not claimed.
4. Pipeline: tiles -> INT8 CNN -> heat map -> risk score; leakage controls.
5. Model and INT8: ~50 k parameters, ops, golden model bit-exact with TFLite reference.
6. Results (synthetic, labelled): tile and farm metrics, INT8 vs FP32, heat map, lab-CPU latency.
7. FPGA simulation status: requantize/MAC verified, layers in progress.
8. Plan to final defense, risks, questions.

Use `docs/RESULTS_TEMPLATE.md` for the numbers and `docs/DEFENSE_CHECKLIST.md` before presenting.
