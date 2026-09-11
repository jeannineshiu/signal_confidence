"""Token-probability ("logprob") confidence, read from the same reply as the
verbalised confidence. Pre-registered definition: PLAN.md §8.

In the reply `{"direction":"bullish", …}` the direction value is decoded as
one token whose alternatives are the enum prefixes the strict JSON schema
allows ("bull", "bear", "neutral", "ne", …). The API returns the top-20
alternatives at that position with their log-probabilities. From them:

    P(label) = total probability of alternatives that can only begin `label`

    logprob confidence = P(chosen) / (P(bullish) + P(bearish))   for a directional call

i.e. the model's own probability of the direction it chose, conditional on
choosing a direction — the same kind of quantity as the verbalised
confidence (0.5 … 1.0). If the opposite direction is not among the top 20,
its probability is below the smallest one listed; the ratio then treats it as
0 and `opposite_in_top_k` records that the value is an upper bound.
"""

import math
import re

LABELS = ("bullish", "bearish", "neutral")
OPPOSITE = {"bullish": "bearish", "bearish": "bullish"}
_VALUE_START = re.compile(r'"direction"\s*:\s*"')


def _label_of(fragment: str) -> str | None:
    """The single label a token fragment can begin, or None if ambiguous/none."""
    frag = fragment.lower().rstrip('"')
    if not frag:
        return None
    matches = [lab for lab in LABELS if lab.startswith(frag) or frag.startswith(lab)]
    return matches[0] if len(matches) == 1 else None


def direction_distribution(token_logprobs: list[dict]) -> dict:
    """Probabilities of each label at the direction token, from the top-k list."""
    text = "".join(t["token"] for t in token_logprobs)
    m = _VALUE_START.search(text)
    if m is None:
        raise ValueError("no direction value in the reply")
    value_at = m.end()

    offset = 0
    for tok in token_logprobs:
        end = offset + len(tok["token"])
        if offset <= value_at < end:
            break
        offset = end
    else:
        raise ValueError("direction value is not covered by any token")

    prefix = text[offset:value_at]  # e.g. '' when the token is 'bull', '"' for '"bull'
    chosen = _label_of(tok["token"][len(prefix):])
    if chosen is None:
        raise ValueError(f"cannot map the chosen token {tok['token']!r} to a label")

    mass = dict.fromkeys(LABELS, 0.0)
    for alt, logprob in tok["top"]:
        if alt.startswith(prefix) and (label := _label_of(alt[len(prefix):])):
            mass[label] += math.exp(logprob)
    return {
        "chosen": chosen,
        "p": mass,
        "captured_mass": sum(mass.values()),
        "smallest_listed": math.exp(min(lp for _, lp in tok["top"])),
        "top": tok["top"],
    }


def logprob_confidence(dist: dict) -> tuple[float | None, bool]:
    """(confidence in the chosen direction, whether the opposite was listed)."""
    chosen = dist["chosen"]
    if chosen == "neutral":
        return None, True
    p = dist["p"]
    other = p[OPPOSITE[chosen]]
    return p[chosen] / (p[chosen] + other), other > 0
