"""Join signals to labels and put every headline in exactly one state.

States, assigned in this order (PLAN.md §1, coverage funnel):

  api_error      the LLM call failed after the SDK's own retries
  parse_failure  the response failed schema validation twice
  neutral        the model declined to call a direction
  flat_label     the next-day move was inside the dead-band, so no direction
                 exists to be right or wrong about
  scored         a bullish/bearish call against an up/down outcome

Nothing is dropped: an item with no signal at all is an error, not a skip,
so accuracy can never be quoted on a silently shrunken set.
"""

import numpy as np
import pandas as pd

from sigconf.data.labels import DOWN, FLAT, UP

SIGNAL_COLUMNS = ["id", "status", "direction", "confidence"]
STATES = ("api_error", "parse_failure", "neutral", "flat_label", "scored")
STATUSES = ("ok", "parse_failure", "api_error")
DIRECTIONS = ("bullish", "bearish", "neutral")


def score(labeled: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """`labeled` rows (id, label, …) + one signal per id → state and hit per row.

    Signals for ids outside `labeled` (e.g. the other split) are ignored.
    """
    if labeled["id"].duplicated().any() or signals["id"].duplicated().any():
        raise ValueError("duplicate ids")
    if not labeled["label"].isin([UP, DOWN, FLAT]).all():
        raise ValueError("every headline needs an up/down/flat label")
    missing = sorted(set(labeled["id"]) - set(signals["id"]))
    if missing:
        raise ValueError(f"{len(missing)} headline(s) have no signal: {missing[:5]}")

    df = labeled.merge(signals[SIGNAL_COLUMNS], on="id", how="left", validate="one_to_one")
    ok = df["status"] == "ok"
    _check_signal_values(df, ok)

    df["state"] = np.select(
        [
            df["status"] == "api_error",
            df["status"] == "parse_failure",
            df["direction"] == "neutral",
            df["label"] == FLAT,
        ],
        list(STATES[:4]),
        default="scored",
    )
    scored = df["state"] == "scored"
    called_up = df["direction"] == "bullish"
    went_up = df["label"] == UP
    df["hit"] = (called_up == went_up).astype("Int64").where(scored, pd.NA)
    return df


def funnel(scored: pd.DataFrame) -> dict[str, int]:
    """Count of items per state, in funnel order, plus the total."""
    counts = scored["state"].value_counts()
    out = {state: int(counts.get(state, 0)) for state in STATES}
    out["total"] = len(scored)
    return out


def _check_signal_values(df: pd.DataFrame, ok: pd.Series) -> None:
    if not df["status"].isin(STATUSES).all():
        raise ValueError(f"status must be one of {STATUSES}")
    if not df.loc[ok, "direction"].isin(DIRECTIONS).all():
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    conf = df.loc[ok, "confidence"].astype(float)
    if conf.isna().any() or ((conf < 0) | (conf > 1)).any():
        raise ValueError("confidence must lie in [0, 1] for every ok signal")
