"""Hard spend limits enforced in code (spec §2), not left to the prompt.

Two checks against LLM_MAX_CALLS and LLM_MAX_COST_USD:

* pre-flight, before the batch: the worst case (every pending item needs its
  retry, every call uses its full output allowance) must fit, or nothing runs;
* in-flight, before every call: the remaining budget must cover one more
  worst-case call, measured against the *actual* tokens billed so far. The
  next call is assumed to cost the larger of the estimate and the priciest
  call actually billed, so an estimate that proves too low stops the batch
  instead of overshooting. Only the very first call relies on the estimate
  alone; output is hard-capped by max_completion_tokens, so it cannot run away.
"""

from dataclasses import dataclass, field

from sigconf.signals.prompt import GENERATION_PARAMS

# USD per 1M tokens (input, output). Verified 2026-09-11 against
# https://developers.openai.com/api/docs/pricing
PRICES_PER_MTOK = {"gpt-4o-mini-2024-07-18": (0.15, 0.60)}

ATTEMPTS_PER_ITEM = 2  # first try + one retry on malformed output
# The JSON schema and chat formatting add tokens beyond the visible text.
INPUT_OVERHEAD_TOKENS = 250


class BudgetExceeded(RuntimeError):
    pass


def estimate_input_tokens(*texts: str) -> int:
    """Deliberately high: ~3 characters per token (English averages ~4)."""
    return sum(len(t) for t in texts) // 3 + INPUT_OVERHEAD_TOKENS


@dataclass
class CostGuard:
    model: str
    max_calls: int
    max_cost_usd: float
    calls: int = 0
    spent_usd: float = 0.0
    input_tokens: int = field(default=0)
    output_tokens: int = field(default=0)
    priciest_call_usd: float = field(default=0.0)

    def __post_init__(self):
        if self.model not in PRICES_PER_MTOK:
            raise ValueError(f"no price on record for {self.model!r}; add it to PRICES_PER_MTOK")

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        p_in, p_out = PRICES_PER_MTOK[self.model]
        return (input_tokens * p_in + output_tokens * p_out) / 1e6

    def worst_case_call(self, system: str, user: str) -> float:
        # A retry re-sends the prompt plus a short correction note: allow 200 chars.
        return self.cost(estimate_input_tokens(system, user) + 70,
                         GENERATION_PARAMS["max_completion_tokens"])

    def preflight(self, system: str, users: list[str]) -> float:
        """Raise before any call if the worst case for the batch would exceed a cap."""
        calls = ATTEMPTS_PER_ITEM * len(users)
        cost = ATTEMPTS_PER_ITEM * sum(self.worst_case_call(system, u) for u in users)
        if self.calls + calls > self.max_calls:
            raise BudgetExceeded(
                f"worst case {calls} calls for {len(users)} items exceeds "
                f"LLM_MAX_CALLS={self.max_calls}")
        if self.spent_usd + cost > self.max_cost_usd:
            raise BudgetExceeded(
                f"worst case ${cost:.4f} for {len(users)} items exceeds "
                f"LLM_MAX_COST_USD={self.max_cost_usd}")
        return cost

    def authorize(self, system: str, user: str) -> None:
        """Called immediately before every request."""
        if self.calls + 1 > self.max_calls:
            raise BudgetExceeded(f"LLM_MAX_CALLS={self.max_calls} reached")
        next_call = max(self.worst_case_call(system, user), self.priciest_call_usd)
        if self.spent_usd + next_call > self.max_cost_usd:
            raise BudgetExceeded(
                f"LLM_MAX_COST_USD={self.max_cost_usd} would be exceeded "
                f"(spent ${self.spent_usd:.4f})")

    def record(self, input_tokens: int, output_tokens: int) -> float:
        cost = self.cost(input_tokens, output_tokens)
        self.calls += 1
        self.spent_usd += cost
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.priciest_call_usd = max(self.priciest_call_usd, cost)
        return cost
