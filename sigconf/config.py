"""Project-wide configuration.

The analysis constants in this module are *pre-registered* (PLAN.md §1): they
are fixed before the test split is scored and must never be tuned against test
results. Changing one after the test run invalidates the held-out evaluation.

LLM spend limits come from the environment (spec §7) and are read through
`llm_settings()` at call time rather than at import, so tests can override
them and nothing here loads a `.env` file as a side effect.
"""

import os
from dataclasses import dataclass
from datetime import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_PATH = DATA_DIR / "sample" / "headlines.csv"
CACHE_DIR = DATA_DIR / "cache"
SIGNALS_CACHE_PATH = CACHE_DIR / "signals.jsonl"
RESULTS_DIR = ROOT / "results"
IMG_DIR = ROOT / "img"

# ── Pre-registered analysis decisions (PLAN.md §1) ────────────────────────────
SEED = 42

# Universe: a high-volatility tech name and a low-volatility defensive name,
# both with >700 distinct days of company-specific headlines in the dataset.
# A headline is eligible only if it names the company (pattern below) and is
# not a multi-stock list item ("Stocks That Hit 52-Week Highs On Friday"),
# which says nothing specific about the ticker. Matching is case-insensitive.
TICKER_PATTERNS = {
    "NVDA": r"nvidia|\bnvda\b",
    "JNJ": r"johnson\s*&\s*johnson|\bj\s*&\s*j\b|\bjnj\b",
}
LIST_HEADLINE_PATTERN = (
    r"stocks (?:that|moving|to watch)|biggest movers|top \d|\d+ stocks|mid-day|pre-market"
    r"|after-hours|session|earnings scheduled|mid-afternoon"
)
# Coverage for both tickers is thin before 2011; the dataset ends in June 2020.
SAMPLE_START = "2011-01-01"
PRICE_START = "2010-12-01"
PRICE_END = "2020-07-01"  # exclusive; leaves room for t1 after the last headline
RAW_HEADLINES_FILE = "analyst_ratings_processed.csv"

N_HEADLINES = 150  # split evenly across TICKER_PATTERNS
# Chronological split: the earliest DEV_FRACTION of headlines is used only to
# iterate on the prompt; every reported number comes from the remainder.
DEV_FRACTION = 0.30

# A next-day close-to-close move with |r| < DEAD_BAND is labelled "flat" and
# excluded from directional metrics (but still counted in coverage).
DEAD_BAND = 0.0010

# The anchor close must be strictly after publication; see sigconf/data/labels.py.
MARKET_TZ = "America/New_York"
MARKET_CLOSE = time(16, 0)

# Confidence buckets, left-closed with the last bucket closed on both ends.
# [0.0, 0.5) holds "incoherent" confidences: a directional call the model itself
# rates as more likely wrong than right. They are reported, never clamped.
BIN_EDGES = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

N_BOOTSTRAP = 10_000

# ── Runtime LLM settings (environment) ────────────────────────────────────────
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_MAX_CALLS = 400
DEFAULT_MAX_COST_USD = 1.00


@dataclass(frozen=True)
class LLMSettings:
    model: str
    max_calls: int
    max_cost_usd: float


def llm_settings() -> LLMSettings:
    """Read LLM settings from the environment, validating the spend caps."""
    model = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL).strip()
    max_calls = int(os.environ.get("LLM_MAX_CALLS", DEFAULT_MAX_CALLS))
    max_cost_usd = float(os.environ.get("LLM_MAX_COST_USD", DEFAULT_MAX_COST_USD))

    if not model:
        raise ValueError("LLM_MODEL must not be empty")
    if max_calls < 0:
        raise ValueError(f"LLM_MAX_CALLS must be >= 0, got {max_calls}")
    if not max_cost_usd >= 0:  # also rejects NaN
        raise ValueError(f"LLM_MAX_COST_USD must be >= 0, got {max_cost_usd}")
    return LLMSettings(model=model, max_calls=max_calls, max_cost_usd=max_cost_usd)
