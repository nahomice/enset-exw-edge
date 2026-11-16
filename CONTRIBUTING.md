# Working agreement

- `main` is always green (`pytest -q`). Work on a branch `name/topic`, open a pull request, the other student reviews.
- One config file per experiment in `configs/`; record each result in `docs/EXPERIMENT_LOG.md` (commit, config, seed, numbers).
- Never commit real images, models (`*.keras`, `*.tflite`) or Quartus build products; they are in `.gitignore`.
- Code style: `ruff check src tests scripts`. New logic needs a test; anything touching the INT8 path must keep
  `tests/test_golden.py` passing.
- Issues use the task template (goal, checkable "done when", milestone, owner).
