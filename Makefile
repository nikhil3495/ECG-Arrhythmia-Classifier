# Reproduce everything with `make all` (Linux/macOS; LiteRT converter wheels are not published for Windows).
SHELL := /bin/bash
PY ?= .venv/bin/python
VARIANTS := tiny small base

.PHONY: setup data train export benchmark report app test lint format all clean

setup:  ## create .venv and install pinned dependencies
	@if command -v uv >/dev/null 2>&1; then \
		uv venv -q --python 3.12 .venv && \
		uv pip install --python $(PY) -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match && \
		uv pip install --python $(PY) --no-deps -e . ; \
	else \
		python3.12 -m venv .venv && \
		$(PY) -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu && \
		$(PY) -m pip install --no-deps -e . ; \
	fi

data:  ## download MIT-BIH, preprocess beats, write the data card
	$(PY) -m ecg_edge.data --config configs/default.yaml

train:  ## train tiny/small/base inter-patient, pick best by val macro-F1, train it intra-patient too
	for v in $(VARIANTS); do $(PY) -m ecg_edge.train --config configs/$$v.yaml --protocol inter || exit 1; done
	$(PY) scripts/select_best.py
	$(PY) -m ecg_edge.train --config $$($(PY) scripts/select_best.py --print-config) --protocol intra
	$(PY) -m ecg_edge.evaluate

export:  ## LiteRT FP32 / int8 (static, full-integer) / int8 dynamic-range export + parity checks
	$(PY) -m ecg_edge.quantize_export

benchmark:  ## size, accuracy/F1 and single-beat latency for every variant x runtime, then plots
	$(PY) -m ecg_edge.benchmark
	$(PY) -m ecg_edge.plots

report:  ## inject results into README/REPORT, build demo assets + Pages site, execute notebook
	$(PY) scripts/make_demo_assets.py
	$(PY) scripts/update_docs.py
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/results_walkthrough.ipynb

app:  ## run the Gradio demo locally (int8 LiteRT model only)
	$(PY) app/app.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m black --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m black .

all: setup data train export benchmark report test

clean:
	rm -rf checkpoints exports .pytest_cache .ruff_cache
