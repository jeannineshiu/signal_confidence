"""The prompt, versioned. Any edit changes PROMPT_SHA and so misses the cache.

Iterate on the dev split only; bump PROMPT_VERSION and log the change here.

Changelog
  v1  initial prompt.
"""

import hashlib
import json

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """\
You are an equity analyst. You will see one news headline about a listed company.
Predict how the company's stock will move over the next trading day, judging only from
the headline.

Return:
- direction: "bullish" if you expect the stock to rise, "bearish" if you expect it to
  fall, "neutral" if the headline gives you no directional view.
- confidence: your probability, from 0.5 to 1.0, that the stock moves in the direction
  you stated over the next trading day. 0.5 means a coin flip; 1.0 means certainty.
  Be calibrated: across many headlines where you say 0.7, the stock should move your
  way about 70% of the time. For "neutral", use 0.5.
- reasoning: one short sentence.

Reply with JSON only."""

USER_TEMPLATE = "Ticker: {ticker}\nHeadline: {headline}"

# Everything that shapes the model's output. Part of the cache key.
GENERATION_PARAMS = {"temperature": 0.0, "seed": 42, "max_completion_tokens": 150}

CORRECTION_TEMPLATE = (
    "Your previous reply was not valid ({error}). "
    "Reply again with JSON only, matching the required fields exactly."
)


def render_user(ticker: str, headline: str) -> str:
    """Ticker and headline only: no date, no prices (see PLAN.md §2)."""
    return USER_TEMPLATE.format(ticker=ticker, headline=headline)


def _sha() -> str:
    material = json.dumps(
        {
            "version": PROMPT_VERSION,
            "system": SYSTEM_PROMPT,
            "user": USER_TEMPLATE,
            "correction": CORRECTION_TEMPLATE,
            "params": GENERATION_PARAMS,
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()[:16]


PROMPT_SHA = _sha()
