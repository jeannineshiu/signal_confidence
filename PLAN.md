# PLAN — LLM Signal Confidence & Calibration

Drafted 2026-09-11. The project spec sets the requirements (the *what*). It is kept locally and is not in this repo, and "spec §N" below refers to its sections. This file covers the *how* and *in what order*.
If the two conflict, the spec wins. Update this plan instead of letting it drift.

---

## 0. Decisions (confirmed 2026-09-11)

| # | Decision | Recommendation | Alternative / note |
|---|----------|----------------|--------------------|
| D1 | LLM provider + model | **OpenAI `gpt-4o-mini`** through the plain `openai` SDK (no LangChain), `temperature=0`, `seed=42`. **Locked by D2:** its training cutoff (2023-10-01) predates every headline, and the signal stage refuses any model not in `LLM_TRAINING_CUTOFFS` with a cutoff before `SAMPLE_START`. | This follows `aws-ai-agent` (Pydantic schema, same API key) and keeps dependencies minimal. About 150 calls, so roughly $0.02 per full run. |
| D2 | Headline dataset | **Revised 2026-09-11: only real data, and it must postdate the training cutoff.** Kaggle `frankossai/apple-stock-aapl-historical-financial-news-data` (mostly Yahoo Finance, UTC timestamps, listed as CC0), **AAPL headlines from 2023-11-01 to 2024-11-22**, all after gpt-4o-mini's cutoff, so the model can't "remember" what happened next. | The first build used Benzinga 2009–2020 (NVDA + JNJ), which was real but entirely inside the training window, so it was replaced. The full search is in `notes/dataset_survey.md`. |
| D3 | Label anchor rule | **Timestamp-aware:** the entry close must come strictly *after* publication (see §1) | Taken literally, spec §4 ("close D → close D+1") lets after-close headlines use a close that happened before the news was out. That stays inside spec §2's no-look-ahead principle but flatters the signal. The stricter rule is in the spirit of §2. |
| D4 | Git | `git init` **inside** `signal_confidence/` | `$HOME` is itself a git repo (branch `jeannine`), so without this, commits would land in the home repo. System git is 2.23, so use `git init` then `git checkout -b main` (`init -b` is not available). |

---

## 1. Pre-registered analysis decisions

Write these into code (`sigconf/config.py`) and into the README **before the test split is ever scored**.
Changing any of them after seeing test results burns the test set. That's the whole point of the split.

| Topic | Decision |
|-------|----------|
| Universe | **AAPL only** (spec allows 1–2 tickers). **150 headlines**, seeded sample (`SEED=42`) from 2023-11-01 to the data end (2024-11). |
| Eligibility | In order: published on or after `SAMPLE_START` → names the company (`COMPANY_PATTERN`) → **exact publication time known** (date-only rows stored as midnight UTC/ET are excluded) → **not a live blog** (`LIVE_BLOG_PATTERN`: the title is rewritten after its timestamp, which leaks look-ahead) → first copy of a syndicated title. All rules live in `sigconf/config.py`. No list-headline filter: a regex also removed Apple-specific headlines, so low-information headlines are left for the LLM to call neutral. |
| Independence | **At most one headline per anchor day `t0`.** Headlines sharing `t0` share a label, so keeping several would inflate the effective n. The key is `t0`, not the calendar date: a Saturday headline and a Monday pre-market one both anchor on Monday. |
| Split | **Chronological.** `dev` = earliest 30% (~45) is used *only* to iterate on the prompt. `test` = latest 70% (~105) is run **once** with the frozen prompt. Every README number comes from `test`. |
| Anchor day `t0` | The first trading day whose 16:00 ET close is **strictly after** `published_at`. A pre-16:00 headline on trading day D gives `t0 = D`. After close, a weekend/holiday, **or a missing time-of-day** gives the next trading day. Missing time means exactly midnight in UTC *or* ET. Midnight UTC is the previous evening in ET, so taking it at face value would anchor a day early. |
| Label | `r = close[t0+1] / close[t0] − 1` (the trading day after `t0`). `up = 1` if `r ≥ +0.10%`, `down = 0` if `r ≤ −0.10%`, otherwise **flat**. |
| Dead-band | `|r| < 0.0010` is flat. Flat items are excluded from directional metrics but counted in coverage. The threshold is fixed a priori, never tuned. |
| Prices | `yfinance` daily bars, `auto_adjust=True` passed explicitly (split- and dividend-adjusted), cached to CSV and committed. |
| Confidence meaning | The prompt defines confidence as *"the probability that the price moves in the stated direction over the next trading day"*, with a meaningful range of 0.5–1.0. The schema keeps [0, 1] per spec. **Values below 0.5 are not clamped.** They go into their own `[0.0, 0.5)` bin and are reported as "incoherent confidence". |
| Neutral | Excluded from accuracy, Brier and ECE. Counted in coverage. |
| Bins | `[0.0,0.5) [0.5,0.6) [0.6,0.7) [0.7,0.8) [0.8,0.9) [0.9,1.0]`: left-closed, with the last bin closed on both ends. Edges are explicit literals, not `linspace`. Empty bins are left out of ECE and the plot. |
| Brier | `mean((c − hit)²)` over directional predictions. This equals `mean((P(up) − y)²)` with `P(up) = c` if bullish, `1 − c` if bearish, and a test checks the identity. Baseline: constant 0.5 gives **0.25**. Also report the Brier skill score `1 − B/0.25`. |
| ECE | `Σ_b (n_b/N)·|acc_b − conf_b|`, where `conf_b` is the bin's **mean** confidence (not the bin centre). |
| Inference | Wilson 95% CI on every accuracy. Exact two-sided binomial p-value against 0.5. Bootstrap 95% CI for Brier and ECE (10 000 resamples of the *real* scored rows, seeded). **ECE null distribution:** simulate outcomes `~ Bernoulli(c)` for the *same* confidences, so a perfectly calibrated model at this n gives the reference ECE. With ~60 points, even a perfect model shows ECE > 0, and this is how to say whether the observed ECE is more than noise. **This is a simulation-based significance test, not data. The README says so explicitly (confirmed 2026-09-11).** |
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
│   │   ├── headlines.py         # raw load, eligibility funnel, one-per-anchor-day sample, split
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
- **Prompt content:** ticker and headline only. **No date and no price information.** The headlines postdate the model's cutoff, but leaving dates out is kept as defence in depth, and a test enforces it.
- **Model guard:** `generate.py` refuses to run if `LLM_MODEL` is not in `LLM_TRAINING_CUTOFFS` with a cutoff before `SAMPLE_START`.
- **Cost guard** (env: `LLM_MAX_CALLS`, default 400; `LLM_MAX_COST_USD`, default 1.00):
  - *Pre-flight*, before the batch: `pending × 2 attempts × worst-case cost per call` must fit under the cap, or the run aborts before any call. Worst case uses a conservative input-token estimate (`len(chars)/3` + overhead) and `max_tokens=150` for output. Cached items cost 0.
  - *In-flight*: tally the real `usage` tokens. Before each call, check that the remaining budget covers one worst-case call, and stop if it doesn't.
  - Prices live in one constant with a `# verified YYYY-MM-DD` comment. Re-check `gpt-4o-mini` pricing at implementation time.
- **Reproducibility:** LLMs aren't deterministic even at temperature 0, so the **committed `signals.jsonl` is the source of truth**. Its key includes `prompt_sha`, so editing the prompt causes cache misses and old results can't be reused silently. `make run` calls the API only for missing entries. `--offline` makes a cache miss an error, and CI and the reproduction check use it.
- **README can't drift:** `report/summary.py` renders the numbers block between `<!-- RESULTS:START -->` and `<!-- RESULTS:END -->`, and a test asserts the committed README block equals `render(metrics.json)`. Interpretive sentences such as "over-confident by N pp" are written by hand, and only after checking the underlying rows.

---

## 3. Phases

Every phase ends the same way: `ruff` clean, `pytest` green, one commit. Don't start the next phase with red tests.

### Phase 0: Scaffold (~0.5 day): ✅ done 2026-09-11
- `git init` + `git checkout -b main` (D4). Set up `.gitignore` (`.env`, `data/raw/`, `__pycache__/`, `.venv/`, `.pytest_cache/`) and `.env.example`.
- Create a conda env `signal-confidence` (Python 3.11), matching how `aws-ai-agent` is set up.
- Pin `requirements.txt`: `openai pydantic pandas numpy yfinance matplotlib python-dotenv pytest ruff`. **Don't add scipy** (Wilson and the exact binomial are a few lines each, and tested) **or LangChain**.
- Set up `pyproject.toml` (ruff rules `E F I B UP`, line length 100; pytest `pythonpath=["."]`), the `Makefile`, and the CI workflow.
- `tests/conftest.py`: an autouse fixture deletes `OPENAI_API_KEY`, so an accidental real call fails loudly instead of spending money.
- **Exit:** `make lint test` passes with one smoke test, and CI goes green on the first push.

### Phase 1: Data and labels (~1 day): ✅ done 2026-09-11 (rebuilt on AAPL post-cutoff data)
- Raw file `apple_news_data.csv(.zip)` in `data/raw/` (gitignored). `make sample` rebuilds the committed slice and the price cache.
- Timestamps checked against Apple's five 16:30 ET earnings releases. Handled date-only midnight-UTC rows and live-blog title rewriting. Findings are in `notes/phase1_data_audit.md`.
- Sample: 150 headlines, dev 45 (2023-11 → 2024-03) and test 105 (2024-03 → 2024-11). 10-row manual audit, plus a 150/150 cross-check against unadjusted closes.
- **Base rates differ sharply between splits:** 33% of dev directional labels are up, versus **61% of test (n = 97)**. Always-up is therefore a strong baseline on test.

### Phase 2: Evaluation harness (~1.5 days), built before any LLM call
The harness is the deliverable. It is built and validated **before any LLM call**, so the metrics can't be shaped around the results. **No synthetic data (confirmed 2026-09-11):** the harness is validated on real forecasts.
- `calibration.py`, `brier.py`, `baselines.py`, `uncertainty.py`, `scoring.py`, per §1 and §2.
- Real-data validation using **FiveThirtyEight `checking-our-work-data`** (CC BY 4.0: real published pre-game win probabilities and results). Commit `nba_games.csv` and `nfl_games.csv` under `tests/fixtures/538/` with attribution.
  - (a) **Hand-computed fixture from 5 real NFL rows.** The expected Brier and ECE are worked out by hand in the test file.
  - (b) **Independent-implementation cross-check:** on all ~8.9k real NBA games, our Brier and reliability table must equal scikit-learn's `brier_score_loss` / `calibration_curve` (test-only dependency, `rtol=1e-12`).
  - (c) Map 538's rows into our "direction + confidence" shape (favourite = direction, its probability = confidence ≥ 0.5). Check the result is identical to scoring the raw two-sided probabilities, which is the Brier identity in §1 on real data.
- Math fixtures that are arithmetic, not data (Wilson 50/100, binomial p(8, 10), bin edges) stay as they are.
- `report/figures.py`: the same figure spec as before. Smoke-tested by rendering the 538 NFL data.
- `report/figures.py`:
  - `img/calibration_curve.png`: points at (bin-mean confidence, empirical accuracy) with Wilson error bars, the `y = x` reference, `n` next to each point, and a confidence histogram panel underneath. The histogram shows how few distinct values the LLM actually uses.
  - `img/baselines.png`: LLM vs random vs always-up accuracy with 95% CIs, `n` on each bar, and the random-guess 95% band shaded.
  - Use the matplotlib `Agg` backend and fixed rcParams.
- `report/summary.py`: `metrics.json` plus the README block renderer.
- **Exit:** the harness reproduces the hand-computed and scikit-learn values on real 538 data, and renders both figures from it.

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
- Hand-computed fixture on **5 real 538 NFL games** (expected values written out by hand in the test).
- Cross-check against scikit-learn on all real 538 NBA games (see Phase 2).
- Bin edges: 0.5 → `[.5,.6)`; 0.6 → `[.6,.7)`; 0.9 → `[.9,1.0]`; 1.0 → `[.9,1.0]`; 0.4999 → `[0,.5)`.
- Empty bins are left out and bin weights sum to 1.
- Brier identity: the `(c, hit)` and `(P(up), y)` formulations agree on random inputs.
- A perfect forecaster gives Brier 0. Constant 0.5 gives 0.25. ECE is 0 when every bin's accuracy equals its mean confidence.
- Wilson(50/100) = [0.4038, 0.5962]. Exact binomial two-sided p(k=8, n=10) = 0.109375.
- Bootstrap and null-distribution results are identical for the same seed.
- Empty input and confidence outside [0, 1] raise `ValueError`.

**`data/labels.py`**
- Pre-close intraday on a trading day gives `t0 = D`. **Exactly 16:00:00** gives the next day ("strictly after"). After close gives the next trading day.
- Friday after close, Saturday, and a holiday (e.g. 2019-07-04) all roll to the next trading day. Midnight ET **and midnight UTC** are treated as time unknown and give the next trading day after that date.
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
| ~~Training-data contamination~~ | **Addressed 2026-09-11:** every headline postdates gpt-4o-mini's 2023-10-01 cutoff, the model is locked, and a data test enforces it. What remains is that the model has general priors about Apple. That is ordinary background knowledge, not outcome memory. |
| **One ticker, one year, regime shift.** AAPL only, 2023-11 → 2024-11. Dev is bearish (33% up) and test bullish (61% up). | State both in the README. Always-up (61%) is the baseline to beat. Multi-ticker and walk-forward go under "what I'd do next". |
| Headline text changed after its timestamp | Live blogs are excluded, and timestamps were verified against earnings releases. Ordinary articles can still get small title edits, which stays a documented residual risk. |
| **Small n.** With 105 test items and 30–50% neutral, there may be only ~55–75 directional points, so CIs are about ±12 pp. | CIs on every number, the ECE null distribution, and framing as a method demo. Don't scale up (spec §3). |
| **Confidence clustering.** LLMs tend to use 3–5 distinct values (0.7 / 0.8 / 0.85…), so few bins get populated. | Histogram panel, report the count of distinct values, and bins keyed on mean confidence. That alone is a useful finding. |
| High neutral rate lowers coverage | Pre-set dev-set decision rule (Phase 3). |
| ~~Timestamp ambiguity~~ | **Resolved in Phase 1:** UTC timestamps verified against real events, and date-only rows excluded (`notes/phase1_data_audit.md`). |
| Early-close days (13:00 ET) | Rare. The close is assumed to be 16:00, documented as a known approximation. |
| LLM nondeterminism | The committed cache is the source of truth, and `--offline` reproduction is checked in Phase 5. |
| Headline copyright (publishers own the titles) | Commit only the 150-row slice, with attribution. If that is judged too much, commit IDs plus the build script instead (reproducing then needs a Kaggle account). |
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
- More tickers and walk-forward evaluation across market regimes, still restricted to post-cutoff data
- Token-logprob confidence compared with verbalized confidence
- Recalibration (Platt / isotonic fitted on dev, applied to test)
- Brier decomposition (reliability / resolution / uncertainty)
- Full article text instead of headline only
