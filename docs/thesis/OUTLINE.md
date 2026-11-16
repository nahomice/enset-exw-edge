# Thesis outline mapped to repo artifacts

Every chapter's evidence comes from a file or script in this repo, so nothing in the text is untraceable.

| Chapter | Content | Evidence in the repo |
|---|---|---|
| 1 Introduction | EXW problem, users (agricultural offices), objectives | `docs/user_needs/`, proposal |
| 2 Literature review | EXW, wide-view/drone detection, edge CNNs, quantization, FPGA CNN accelerators | `docs/references.md` |
| 3 Methodology | data tiers, capture, tile rule, splits, model, INT8, golden model, farm decision | `docs/DATA_PROTOCOL.md`, `docs/ARCHITECTURE.md`, `configs/` |
| 4 Implementation | Python pipeline, phone app, FPGA design | `src/exw/`, `android/`, `fpga/` |
| 5 Results | tile and farm metrics, INT8 vs FP32, shift test, latency, energy, resources | `results/*/report.json`, `docs/EXPERIMENT_LOG.md`, `fpga/docs/` |
| 6 Discussion | limits (synthetic vs real, plant size vs tile size), user needs, cost | `docs/RISKS.md`, `docs/RESULTS_TEMPLATE.md` |
| 7 Conclusion | what was shown, what was not, future work | |

## Figures and tables to prepare (file names in `docs/figures/`)
pipeline diagram; tile grid + boxes (`notebooks/02`); heat-map examples (`results/.../figures`); farm ROC (`notebooks/03`);
FP32 vs INT8 table; latency table (lab CPU, ARM, phone, FPGA); FPGA resource table; user-needs summary.
