"""Signal generation: model guard → budget → call → validate → retry once → cache.

Every item ends with a record whose status is one of:
  ok             a response validated against Signal (on attempt 1 or 2)
  parse_failure  both attempts returned output that failed validation
  api_error      the call itself failed (after the SDK's own retries)

Records are appended to data/cache/signals.jsonl as soon as they exist, keyed
by (id, model, prompt_sha). The committed cache is the source of truth for
every reported number: LLM output is not reproducible call-to-call, even at
temperature 0, so reruns read the cache instead of calling again.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from sigconf.config import LLM_TRAINING_CUTOFFS, SAMPLE_START, SIGNALS_CACHE_PATH, LLMSettings
from sigconf.signals.budget import CostGuard
from sigconf.signals.client import LLMCallError, LLMClient
from sigconf.signals.logprob import direction_distribution, logprob_confidence
from sigconf.signals.prompt import (
    CORRECTION_TEMPLATE,
    PROMPT_SHA,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    render_user,
)
from sigconf.signals.schema import Signal


def check_model_allowed(model: str) -> None:
    """Refuse any model whose training data could include the sampled headlines."""
    if model not in LLM_TRAINING_CUTOFFS:
        raise ValueError(
            f"model {model!r} has no documented training cutoff in LLM_TRAINING_CUTOFFS; "
            "only dated snapshots with a cutoff before SAMPLE_START may be used")
    if pd.Timestamp(LLM_TRAINING_CUTOFFS[model]) >= pd.Timestamp(SAMPLE_START):
        raise ValueError(f"{model!r} training cutoff is not before SAMPLE_START={SAMPLE_START}")


def _short_error(err: ValidationError) -> str:
    parts = []
    for e in err.errors()[:3]:
        where = ".".join(str(x) for x in e["loc"]) or "reply"
        parts.append(f"{where}: {e['msg']}")
    return "; ".join(parts)


def generate_signal(
    client: LLMClient,
    guard: CostGuard,
    item_id: str,
    ticker: str,
    headline: str,
    with_logprobs: bool = False,
) -> dict:
    user = render_user(ticker, headline)
    record = {
        "id": item_id, "model": guard.model, "prompt_version": PROMPT_VERSION,
        "prompt_sha": PROMPT_SHA, "status": None, "direction": None, "confidence": None,
        "reasoning": None, "attempts": 0, "raw_texts": [], "error": None,
        "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
    }
    message = user
    for attempt in (1, 2):
        guard.authorize(SYSTEM_PROMPT, message)  # raises BudgetExceeded → stops the batch
        record["attempts"] = attempt
        try:
            completion = client.complete(SYSTEM_PROMPT, message, logprobs=with_logprobs)
        except LLMCallError as exc:
            guard.record(0, 0)  # a failed call still counts against the call cap
            return {**record, "status": "api_error", "error": str(exc)}

        record["cost_usd"] += guard.record(completion.input_tokens, completion.output_tokens)
        record["input_tokens"] += completion.input_tokens
        record["output_tokens"] += completion.output_tokens
        record["raw_texts"].append(completion.text)
        if completion.model != guard.model:
            raise RuntimeError(
                f"requested {guard.model!r} but {completion.model!r} answered; aborting")

        try:
            signal = Signal.model_validate_json(completion.text)
        except ValidationError as exc:
            record["error"] = _short_error(exc)
            message = user + "\n\n" + CORRECTION_TEMPLATE.format(error=record["error"])
            continue
        ok = {**record, "status": "ok", "error": None, **signal.model_dump()}
        return {**ok, **_logprob_fields(completion, signal.direction)} if with_logprobs else ok

    return {**record, "status": "parse_failure"}


def _logprob_fields(completion, direction: str) -> dict:
    """Token-probability confidence for the reply that validated (PLAN.md §8)."""
    try:
        if not completion.token_logprobs:
            raise ValueError("no logprobs returned")
        dist = direction_distribution(completion.token_logprobs)
        if dist["chosen"] != direction:
            raise ValueError(f"token says {dist['chosen']!r}, parsed reply says {direction!r}")
    except ValueError as exc:
        return {"logprob_confidence": None, "logprob_error": str(exc)}
    conf, opposite_listed = logprob_confidence(dist)
    return {
        "logprob_confidence": conf,
        "logprob_error": None,
        "opposite_in_top_k": opposite_listed,
        "direction_p": dist["p"],
        "direction_captured_mass": dist["captured_mass"],
        "direction_top_logprobs": dist["top"],
    }


class SignalCache:
    """Append-only JSONL; the last record per (id, model, prompt_sha) wins."""

    def __init__(self, path: Path = SIGNALS_CACHE_PATH):
        self.path = path
        self._records: dict[tuple[str, str, str], dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._records[(rec["id"], rec["model"], rec["prompt_sha"])] = rec

    def get(self, item_id: str, model: str, prompt_sha: str = PROMPT_SHA) -> dict | None:
        return self._records.get((item_id, model, prompt_sha))

    def append(self, record: dict) -> None:
        record = {**record, "generated_at": datetime.now(UTC).isoformat(timespec="seconds")}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        self._records[(record["id"], record["model"], record["prompt_sha"])] = record


def run_signals(
    items: pd.DataFrame,
    settings: LLMSettings,
    client_factory: Callable[[str], LLMClient],
    cache: SignalCache,
    offline: bool = False,
    retry_api_errors: bool = False,
    log: Callable[[str], None] = print,
    with_logprobs: bool = False,
) -> pd.DataFrame:
    """One signal record per item: from the cache where possible, else the LLM.

    `items` needs id, ticker and headline. The whole batch is checked against
    the spend caps before the first call; a cap hit mid-batch stops the run,
    and everything generated so far is already safely in the cache.
    """
    check_model_allowed(settings.model)

    def needs_call(item_id: str) -> bool:
        rec = cache.get(item_id, settings.model)
        return rec is None or (retry_api_errors and rec["status"] == "api_error")

    pending = items[items["id"].map(needs_call)]
    if len(pending) and offline:
        raise RuntimeError(f"{len(pending)} signal(s) missing from the cache and offline=True")

    if len(pending):
        guard = CostGuard(settings.model, settings.max_calls, settings.max_cost_usd)
        users = [render_user(t, h) for t, h in zip(pending["ticker"], pending["headline"],
                                                    strict=True)]
        worst = guard.preflight(SYSTEM_PROMPT, users)
        log(f"calling {settings.model} for {len(pending)} item(s); worst case ${worst:.4f}")
        client = client_factory(settings.model)
        for row in pending.itertuples():
            cache.append(generate_signal(client, guard, row.id, row.ticker, row.headline,
                                         with_logprobs=with_logprobs))
        log(f"done: {guard.calls} call(s), ${guard.spent_usd:.4f} "
            f"({guard.input_tokens} in / {guard.output_tokens} out tokens)")

    return pd.DataFrame([cache.get(i, settings.model) for i in items["id"]])
