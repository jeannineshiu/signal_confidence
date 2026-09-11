PYTHON ?= python
SPLIT ?= test

.PHONY: install sample run reproduce logprob test lint check

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

# Same as run (+ the logprob analysis), but a cache miss is an error: no network,
# no API key, no spend. Regenerates every figure and number in the README.
reproduce:
	$(PYTHON) -m sigconf.pipeline --split $(SPLIT) --offline
	$(PYTHON) -m sigconf.experiments.logprob_vs_verbalized --split $(SPLIT) --offline

# Secondary analysis (PLAN.md §8): token-probability vs verbalised confidence.
logprob:
	$(PYTHON) -m sigconf.experiments.logprob_vs_verbalized --split $(SPLIT)

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

check: lint test
