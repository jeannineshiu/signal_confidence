PYTHON ?= python
SPLIT ?= test

.PHONY: install sample run reproduce test lint check

install:
	$(PYTHON) -m pip install -r requirements.txt

# One-time: rebuild data/sample/ and the price caches from the raw Kaggle file.
# Not needed to reproduce results — both outputs are committed.
sample:
	$(PYTHON) scripts/build_sample.py

# Full pipeline. Calls the LLM only for headlines missing from the signal cache,
# bounded by LLM_MAX_CALLS / LLM_MAX_COST_USD.
run:
	$(PYTHON) -m sigconf.pipeline --split $(SPLIT)

# Same as run, but a cache miss is an error: no network, no API key, no spend.
reproduce:
	$(PYTHON) -m sigconf.pipeline --split $(SPLIT) --offline

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

check: lint test
