.PHONY: install data train eval export smoke test lint clean check-env npy-dataset ablate requant-vectors compare-rtl notebooks
CONFIG ?= configs/default.yaml

install:
	pip install -e ".[tf,dev]"
check-env:
	python scripts/check_env.py
data:
	python scripts/prepare_data.py --config $(CONFIG)
train:
	python scripts/train.py --config $(CONFIG)
eval:
	python scripts/evaluate.py --config $(CONFIG)
export:
	python scripts/export_fpga.py --config $(CONFIG)
smoke: data train eval export
npy-dataset:
	python scripts/make_npy_dataset.py --out data_npy
ablate:
	bash scripts/run_ablation.sh
requant-vectors:
	python scripts/make_requant_vectors.py --config $(CONFIG)
compare-rtl:
	python scripts/compare_rtl_outputs.py --config $(CONFIG)
test:
	pytest -q
lint:
	ruff check src tests scripts
clean:
	rm -rf data/synthetic data/processed results/smoke
