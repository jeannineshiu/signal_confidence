# PLAN — LLM Signal Confidence & Calibration

Drafted 2026-09-11. `SPEC_signal_confidence.md` sets the requirements (the *what*). This file covers the *how* and *in what order*.
If the two conflict, the spec wins. Update this plan instead of letting it drift.

---

## 0. Decisions to confirm before Phase 1

| # | Decision | Recommendation | Alternative / note |
|---|----------|----------------|--------------------|
| D1 | LLM provider + model | **OpenAI `gpt-4o-mini`** through the plain `openai` SDK (no LangChain), `temperature=0`, `seed=42` | This follows `aws-ai-agent` (Pydantic schema, same API key) and keeps dependencies minimal. About 150 calls, so roughly $0.02 per full run. Only `signals/client.py` changes if you switch provider. |
| D2 | Headline dataset | **Kaggle "Daily Financial News for 6000+ Stocks"** (`miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests`), file `analyst_ratings_processed.csv`: columns title / date / stock, **minute-level timestamps** (author says UTC-4), 2009–2020, listed as **CC0** on Kaggle. The author notes the headlines themselves are Benzinga's property. | FNSPID (Hugging Face `Zihan1004/FNSPID`, 1999–2023, **CC BY-NC 4.0**). Its news table is one 5.7 GB CSV, which is too heavy for a 150-row slice. |
| D3 | Label anchor rule | **Timestamp-aware:** the entry close must come strictly *after* publication (see §1) | Taken literally, spec §4 ("close D → close D+1") lets after-close headlines use a close that happened before the news was out. That stays inside spec §2's no-look-ahead principle but flatters the signal. The stricter rule is in the spirit of §2. |
| D4 | Git | `git init` **inside** `signal_confidence/` | `$HOME` is itself a git repo (branch `jeannine`), so without this, commits would land in the home repo. System git is 2.23, so use `git init` then `git checkout -b main` (`init -b` is not available). |

---

## 1. Pre-registered analysis decisions

Write these into code (`sigconf/config.py`) and into the README **before the test split is ever scored**.
Changing any of them after seeing test results burns the test set. That's the whole point of the split.

| Topic | Decision |
|-------|----------|
| Universe | 2 tickers, picked in Phase 1 by headline coverage (candidate: AAPL plus one contrasting name). **150 headlines** in total, seeded sample (`SEED=42`) from a fixed date window. |
| Independence | **At most one headline per ticker per trading day.** Same-day headlines share a label, so keeping several would inflate the effective n. |
| Split | **Chronological.** `dev` = earliest 30% (~45) is used *only* to iterate on the prompt. `test` = latest 70% (~105) is run **once** with the frozen prompt. Every README number comes from `test`. |
| Anchor day `t0` | The first trading day whose 16:00 ET close is **strictly after** `published_at`. A pre-16:00 headline on trading day D gives `t0 = D`. After close, a weekend/holiday, **or a missing time-of-day (00:00:00)** gives the next trading day. The last rule is conservative and matches the dataset author's own backtesting advice. |
| Label | `r = close[t0+1] / close[t0] − 1` (the trading day after `t0`). `up = 1` if `r ≥ +0.10%`, `down = 0` if `r ≤ −0.10%`, otherwise **flat**. |
| Dead-band | `|r| < 0.0010` is flat. Flat items are excluded from directional metrics but counted in coverage. The threshold is fixed a priori, never tuned. |
| Prices | `yfinance` daily bars, `auto_adjust=True` passed explicitly (split- and dividend-adjusted), cached to CSV and committed. |
| Confidence meaning | The prompt defines confidence as *"the probability that the price moves in the stated direction over the next trading day"*, with a meaningful range of 0.5–1.0. The schema keeps [0, 1] per spec. **Values below 0.5 are not clamped.** They go into their own `[0.0, 0.5)` bin and are reported as "incoherent confidence". |
| Neutral | Excluded from accuracy, Brier and ECE. Counted in coverage. |
| Bins | `[0.0,0.5) [0.5,0.6) [0.6,0.7) [0.7,0.8) [0.8,0.9) [0.9,1.0]`: left-closed, with the last bin closed on both ends. Edges are explicit literals, not `linspace`. Empty bins are left out of ECE and the plot. |
| Brier | `mean((c − hit)²)` over directional predictions. This equals `mean((P(up) − y)²)` with `P(up) = c` if bullish, `1 − c` if bearish, and a test checks the identity. Baseline: constant 0.5 gives **0.25**. Also report the Brier skill score `1 − B/0.25`. |
| ECE | `Σ_b (n_b/N)·|acc_b − conf_b|`, where `conf_b` is the bin's **mean** confidence (not the bin centre). |
| Inference | Wilson 95% CI on every accuracy. Exact two-sided binomial p-value against 0.5. Bootstrap 95% CI for Brier and ECE (10 000 resamples, seeded). **ECE null distribution:** simulate outcomes `~ Bernoulli(c)` for the *same* confidences, so a perfectly calibrated model at this n gives the reference ECE. With ~60 points, even a perfect model shows ECE > 0, and this is how to say whether the observed ECE is more than noise. |
| No P&L | Strategy returns are deliberately not computed, so nothing suggests the signal is tradable. |

### Coverage funnel

Every sampled item ends in exactly one state, assigned in this order:

```
sampled ─→ api_error ─→ parse_failure ─→ neutral ─→ flat_label ─→ scored (bullish/bearish vs up/down)
```

`metrics.json` and the README carry the full funnel. An accuracy is always quoted as
*"x% (95% CI a–b) on n = k scored items = y% of N = 105 test headlines"*.

---

## 2. Architecture

Spec §7 names the modules `data/ signals/ eval/ report/`. They live under a package `sigconf/` for two reasons: a top-level `data/` package would collide with the data-file directory, and `import eval` would shadow the builtin.

```
signal_confidence/
├── SPEC_signal_confidence.md
├── PLAN.md
├── README.md                    # RESULTS block auto-generated between markers
├── Makefile                     # install | sample | run | test | lint
├── requirements.txt             # pinned; runtime + pytest + ruff
├── pyproject.toml               # ruff + pytest config only (pythonpath = ".")
├── .env.example                 # OPENAI_API_KEY, LLM_MODEL, LLM_MAX_CALLS, LLM_MAX_COST_USD
├── .github/workflows/ci.yml     # ruff + pytest, no secrets, no real LLM calls
├── scripts/
│   └── build_sample.py          # one-time: raw Kaggle CSV → data/sample/headlines.csv (seeded)
├── data/
│   ├── raw/                     # gitignored — Kaggle download
│   ├── sample/headlines.csv     # committed 150-row slice: id, ticker, published_at, headline, split
│   └── cache/
│       ├── prices_<TICKER>.csv  # committed
│       └── signals.jsonl        # committed raw LLM responses, keyed (headline_id, model, prompt_sha)
├── sigconf/
│   ├── config.py                # paths, SEED, DEAD_BAND, BINS, env-driven caps
│   ├── pipeline.py              # ENTRY POINT: python -m sigconf.pipeline [--split] [--offline]
│   ├── data/
│   │   ├── headlines.py         # load sample, dedupe per ticker-day, split
│   │   ├── prices.py            # yfinance fetch + CSV cache (cache-first)
│   │   └── labels.py            # anchor_day(), next_day_label() — pure functions
│   ├── signals/
│   │   ├── schema.py            # Pydantic Signal
│   │   ├── prompt.py            # PROMPT_VERSION, SYSTEM, USER_TEMPLATE, changelog docstring
│   │   ├── client.py            # LLMClient Protocol + OpenAIClient
│   │   ├── budget.py            # CostGuard (pre-flight + in-flight)
│   │   └── generate.py          # validate → retry once → status; cache read/write
│   ├── eval/
│   │   ├── scoring.py           # join signals × labels → funnel + scored frame
│   │   ├── calibration.py       # assign_bins, reliability_table, ece
│   │   ├── brier.py             # brier_score, brier_skill_score
│   │   ├── baselines.py         # accuracy, always_up, random band, wilson_interval, binomial_p
│   │   └── uncertainty.py       # bootstrap_ci, ece_null_distribution
│   └── report/
│       ├── figures.py           # calibration_curve.png, baselines.png
│       └── summary.py           # results/metrics.json + README block rendering
├── results/metrics.json         # committed
├── img/                         # committed figures
└── tests/                       # see §4
```

### Data flow

```
data/sample/headlines.csv ──┐
yfinance → prices cache ────┴─→ labels ─→ labeled frame
                                             │
             prompt + CostGuard + cache ─→ signals ─┤
                                             ▼
                            scoring (funnel) ─→ eval ─→ results/metrics.json
                                                          ├─→ img/*.png
                                                          └─→ README RESULTS block
```

### Key interfaces

```python
class Signal(BaseModel):
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=300)   # "short"; not enforced as one sentence

class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> Completion: ...   # text, input_tokens, output_tokens

def generate_signal(client, item, guard) -> SignalRecord: ...
    # status ∈ {ok, parse_failure, api_error}; attempts; raw_text; usage; prompt_sha; model
```

- **Structured output:** request OpenAI's JSON-schema response format, but **validate with Pydantic in our own code** (`Signal.model_validate_json`). That keeps the retry path ours and testable with a fake client.
- **Retry:** if attempt 1 fails with `ValidationError` or a JSON decode error, attempt 2 re-sends the same prompt plus a one-line correction naming the error. If that also fails, the item becomes `parse_failure` and its raw text is stored. Transient HTTP errors go to the SDK's own `max_retries`. Anything that still fails becomes `api_error`, which is separate from `parse_failure` and also counted.
- **Prompt content:** ticker and headline only. **No date and no price information.** Dates make it easier for the model to recall what happened next (see Risks), and a test enforces this.
- **Cost guard** (env: `LLM_MAX_CALLS`, default 400; `LLM_MAX_COST_USD`, default 1.00):
  - *Pre-flight*, before the batch: `pending × 2 attempts × worst-case cost per call` must fit under the cap, or the run aborts before any call. Worst case uses a conservative input-token estimate (`len(chars)/3` + overhead) and `max_tokens=150` for output. Cached items cost 0.
  - *In-flight*: tally the real `usage` tokens. Before each call, check that the remaining budget covers one worst-case call, and stop if it doesn't.
  - Prices live in one constant with a `# verified YYYY-MM-DD` comment. Re-check `gpt-4o-mini` pricing at implementation time.
- **Reproducibility:** LLMs aren't deterministic even at temperature 0, so the **committed `signals.jsonl` is the source of truth**. Its key includes `prompt_sha`, so editing the prompt causes cache misses and old results can't be reused silently. `make run` calls the API only for missing entries. `--offline` makes a cache miss an error, and CI and the reproduction check use it.
- **README can't drift:** `report/summary.py` renders the numbers block between `<!-- RESULTS:START -->` and `<!-- RESULTS:END -->`, and a test asserts the committed README block equals `render(metrics.json)`. Interpretive sentences such as "over-confident by N pp" are written by hand, and only after checking the underlying rows.

---

## 3. Phases

Every phase ends the same way: `ruff` clean, `pytest` green, one commit. Don't start the next phase with red tests.

### Phase 0: Scaffold (~0.5 day)
- `git init` + `git checkout -b main` (D4). Set up `.gitignore` (`.env`, `data/raw/`, `__pycache__/`, `.venv/`, `.pytest_cache/`) and `.env.example`.
- Create a conda env `signal-confidence` (Python 3.11), matching how `aws-ai-agent` is set up.
- Pin `requirements.txt`: `openai pydantic pandas numpy yfinance matplotlib python-dotenv pytest ruff`. **Don't add scipy** (Wilson and the exact binomial are a few lines each, and tested) **or LangChain**.
- Set up `pyproject.toml` (ruff rules `E F I B UP`, line length 100; pytest `pythonpath=["."]`), the `Makefile`, and the CI workflow.
- `tests/conftest.py`: an autouse fixture deletes `OPENAI_API_KEY`, so an accidental real call fails loudly instead of spending money.
- **Exit:** `make lint test` passes with one smoke test, and CI goes green on the first push.

### Phase 1: Data and labels (~1 day)
- Download `analyst_ratings_processed.csv` into `data/raw/` (Kaggle CLI or browser).
- **Check the data before trusting it** (a throwaway script whose findings go into `notes/`):
  - Timestamp format and offsets. The author says "UTC-4". If January rows also show `-04:00`, the offset is a fixed label and winter times are off by an hour. Count the rows within ±1 h of 16:00 ET that could change anchor day, then exclude or document them.
  - The share of rows at exactly `00:00:00` (time unknown, so the next trading day is used).
  - Headlines per ticker per year, and the mix of headline types (analyst actions vs news vs "stocks moving" lists).
- Choose the 2 tickers and the date window. Write `scripts/build_sample.py`: filter → one headline per ticker-day → seeded sample of 150 → chronological split column → `data/sample/headlines.csv`. Commit the slice.
- Implement `prices.py` (cache-first) and `labels.py` (pure functions), test-first (§4).
- **Manual audit:** print 10 random labeled rows (`published_at`, `t0`, `t0+1`, both closes, `r`, label) and check 3 of them against a price chart by hand. Record them in `notes/`.
- **Exit:** labeled sample with up/down/flat counts and base rate printed, and the no-look-ahead invariant test passes on the real sample.

### Phase 2: Evaluation harness (~1.5 days), built before any LLM call
The harness is the deliverable. Building it against synthetic data first means the metrics can't be shaped around the real results.
- `calibration.py`, `brier.py`, `baselines.py`, `uncertainty.py`, `scoring.py`, per §1 and §2.
- Synthetic sanity checks, written as tests:
  - (a) A **perfectly calibrated** forecaster (`hit ~ Bernoulli(c)`): observed ECE falls inside its own null distribution, and the curve hugs the diagonal.
  - (b) An **over-confident** forecaster (reports 0.9, true rate 0.6): ECE ≈ 0.30, and the curve sits below the diagonal.
  - (c) A constant 0.5 forecaster gives Brier = 0.25 exactly.
- `report/figures.py`:
  - `img/calibration_curve.png`: points at (bin-mean confidence, empirical accuracy) with Wilson error bars, the `y = x` reference, `n` next to each point, and a confidence histogram panel underneath. The histogram shows how few distinct values the LLM actually uses.
  - `img/baselines.png`: LLM vs random vs always-up accuracy with 95% CIs, `n` on each bar, and the random-guess 95% band shaded.
  - Use the matplotlib `Agg` backend and fixed rcParams.
- `report/summary.py`: `metrics.json` plus the README block renderer.
- **Exit:** `python -m sigconf.pipeline --synthetic` produces both figures and `metrics.json` from a fake signals file.

### Phase 3: Signal generation (~1 day)
- Build `schema.py`, `prompt.py`, `client.py`, `budget.py`, `generate.py` and the JSONL cache, test-first with a `FakeClient` (§4).
- **Dev run only:** `make run SPLIT=dev` (~45 calls, cents). Look at:
  - neutral rate
  - confidence distribution (how many distinct values?)
  - parse failures
  - a sample of `reasoning` strings
- Iterate on the prompt **on dev only**. Bump `PROMPT_VERSION` each time and record the changelog in the `prompt.py` docstring.
- Decision rule, fixed now: if the dev neutral rate is > 60%, the only allowed fix is a prompt change evaluated on dev. Test items are never re-sampled.
- Freeze the prompt, commit, and tag `prompt-frozen`.
- **Exit:** the dev funnel looks sane, and there are zero unexplained parse failures.

### Phase 4: Test run and report (~0.5 day)
- Run the test split **once** with `make run`, then commit `signals.jsonl`, `metrics.json` and `img/`.
- Before writing any interpretation, read the rows: the highest-confidence misses, and what is in the `<0.5` bin (if anything). Then write the plain-English readings.
- **Exit:** every spec §6 output exists and is regenerated by `make run`.

### Phase 5: README and definition of done (~0.5 day)
- Write the README in exactly the spec §8 order (1 framing → 2 headline result with baseline and n → 3 figure plus reading → 4 Brier and ECE plus meaning → 5 limitations and next steps → 6 reproduction command, data source and licence).
- **Fresh-clone reproduction check:** clone into a temp dir, create a fresh env, `pip install -r requirements.txt`, then `make run` with **no API key** (`--offline`). `diff` the resulting `metrics.json` against the committed one: it must be identical.
- CI green. Run through the DoD checklist (§6).

Total ≈ 5 working days.

---

## 4. Test plan (priority: eval math > labels > signals > report)

**`eval/` (hardest)**
- Hand-computed fixture: confidences `[0.55, 0.65, 0.65, 0.85, 0.95]`, hits `[1, 0, 1, 1, 0]`
  - Brier = (0.2025 + 0.4225 + 0.1225 + 0.0225 + 0.9025) / 5 = **0.3345**
  - ECE = (1·0.45 + 2·0.15 + 1·0.15 + 1·0.95) / 5 = **0.37** (bins [.5,.6) n1 · [.6,.7) n2 acc .5 · [.8,.9) n1 · [.9,1] n1)
- Bin edges: 0.5 → `[.5,.6)`; 0.6 → `[.6,.7)`; 0.9 → `[.9,1.0]`; 1.0 → `[.9,1.0]`; 0.4999 → `[0,.5)`.
- Empty bins are left out and bin weights sum to 1.
- Brier identity: the `(c, hit)` and `(P(up), y)` formulations agree on random inputs.
- A perfect forecaster gives Brier 0. Constant 0.5 gives 0.25. ECE is 0 when every bin's accuracy equals its mean confidence.
- Wilson(50/100) = [0.4038, 0.5962]. Exact binomial two-sided p(k=8, n=10) = 0.109375.
- Bootstrap and null-distribution results are identical for the same seed.
- Empty input and confidence outside [0, 1] raise `ValueError`.

**`data/labels.py`**
- Pre-close intraday on a trading day gives `t0 = D`. **Exactly 16:00:00** gives the next day ("strictly after"). After close gives the next trading day.
- Friday after close, Saturday, and a holiday (e.g. 2019-07-04) all roll to the next trading day. `00:00:00` is treated as time unknown and gives the next trading day.
- A UTC-timestamped input converts correctly across both DST transitions.
- Dead-band boundaries: `r = +0.0010` → up, `r = +0.00099` → flat, `r = −0.0010` → down.
- **Look-ahead invariant:** for every labeled row, `close_time(t0) > published_at`. This runs as a unit test and as a data test on the committed sample.
- **Isolation:** changing every price except `close[t0]` and `close[t0+1]` leaves the label unchanged.
- The final day of the price history (no `t0+1`) gives a missing label, and nothing is filled in silently.

**`signals/`** (`FakeClient` with scripted responses and call counting)
- Valid JSON → `ok`, 1 attempt. Bad then good → `ok`, 2 attempts. Bad twice → `parse_failure` with raw text kept.
- Validation failures: `confidence: 1.5`, `direction: "BUY"`, a missing field, prose instead of JSON, and an empty string.
- A raised API error gives `api_error`, never `parse_failure`.
- CostGuard: pre-flight over the cap aborts with **0 client calls**. The in-flight stop triggers when the budget runs out. Cached items don't count against the cap.
- A cache hit skips the client, and changing the prompt text causes a miss.
- The rendered prompt contains neither the publication date nor the year.

**`report/`**
- The README RESULTS block equals `render(metrics.json)` (drift test).
- Each figure function writes a non-empty PNG (smoke test).

---

## 5. Risks

| Risk | Mitigation |
|------|------------|
| **Training-data contamination.** Headlines from 2009–2020 are inside the model's training data, so it may "know" how big events played out. This is a form of look-ahead in its own right. | No dates in the prompt. Make it limitation #1 in the README. Excluding it requires headlines from after the model's training cutoff, which goes under "what I'd do next". |
| **Small n.** With 105 test items and 30–50% neutral, there may be only ~55–75 directional points, so CIs are about ±12 pp. | CIs on every number, the ECE null distribution, and framing as a method demo. Don't scale up (spec §3). |
| **Confidence clustering.** LLMs tend to use 3–5 distinct values (0.7 / 0.8 / 0.85…), so few bins get populated. | Histogram panel, report the count of distinct values, and bins keyed on mean confidence. That alone is a useful finding. |
| High neutral rate lowers coverage | Pre-set dev-set decision rule (Phase 3). |
| Timestamp / timezone ambiguity | Phase 1 check; count and document the rows near the 16:00 boundary. |
| Early-close days (13:00 ET) | Rare. The close is assumed to be 16:00, documented as a known approximation. |
| LLM nondeterminism | The committed cache is the source of truth, and `--offline` reproduction is checked in Phase 5. |
| Benzinga headline copyright | Commit only the 150-row slice, with attribution. If that is judged too much, commit IDs plus the build script instead (reproducing then needs a Kaggle account). |
| Scope creep | Spec §3. New ideas go into the README's "what I'd do next", not into code. |

---

## 6. Definition of done (spec §9 → checks)

- [ ] `pip install -r requirements.txt` then `make run` regenerates every figure and number in the README, verified by the fresh-clone `--offline` diff
- [ ] `pytest` green and `ruff check .` clean, locally and in CI
- [ ] The README follows spec §8 order, and the limitations section is present
- [ ] Every headline metric appears next to its baseline and n (accuracy vs random vs always-up, Brier vs 0.25, ECE vs null)
- [ ] The coverage funnel is reported, with parse failures counted rather than dropped
- [ ] The no-look-ahead rule is documented in the `labels.py` docstring and the README, and enforced by the invariant test
- [ ] The cost cap is enforced in code, and actual spend for a full run is recorded in `metrics.json` (expected < $0.10)
- [ ] API keys come from the environment only, and `.env` is gitignored
- [ ] No real LLM calls in tests or CI

## 7. README "what I'd do next" candidates (write these up, don't build them)
- Headlines from after the model's training cutoff, to rule out contamination
- Walk-forward evaluation across market regimes, with more tickers
- Token-logprob confidence compared with verbalized confidence
- Recalibration (Platt / isotonic fitted on dev, applied to test)
- Brier decomposition (reliability / resolution / uncertainty)
- Full article text instead of headline only
