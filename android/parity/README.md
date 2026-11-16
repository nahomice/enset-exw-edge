# parity/

Checks that the app and the Python reference make the same decision on the same image.

1. `python android/parity/make_parity_case.py --tflite <model> --image <photo>` writes `case_<name>.json` (tile grid size, per-tile P(EXW), risk, clusters).
2. Open the same photo in the app; compare grid size (must be identical), risk R (should match to 2 decimals), clusters.
3. Small differences in individual probabilities are expected (JPEG decoding differences between PIL and Android,
   +-2 LSB kernel differences). The **flagged / not flagged decision** and the cluster count must agree on the
   parity set (at least 10 images, record the result in `docs/EXPERIMENT_LOG.md`).
