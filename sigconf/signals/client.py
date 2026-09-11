"""LLM client interface and the OpenAI implementation.

generate.py depends only on the LLMClient protocol, so tests drive the whole
signal path with a scripted fake and CI never makes a network call.
"""

from dataclasses import dataclass
from typing import Protocol

from sigconf.signals.prompt import GENERATION_PARAMS
from sigconf.signals.schema import RESPONSE_SCHEMA

TOP_LOGPROBS = 20  # the API maximum


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    model: str  # the model the provider says actually answered
    # Per output token, when requested: {"token", "logprob", "top": [[token, logprob], …]}
    token_logprobs: list[dict] | None = None


class LLMCallError(RuntimeError):
    """The call itself failed (network, auth, rate limit, server) after retries."""


class LLMClient(Protocol):
    def complete(self, system: str, user: str, logprobs: bool = False) -> Completion: ...


class OpenAIClient:
    """Chat Completions with a strict JSON-schema response format."""

    def __init__(self, model: str, timeout: float = 30.0, max_retries: int = 2):
        import openai  # lazy: only real runs need the SDK configured

        self._openai = openai
        self._client = openai.OpenAI(timeout=timeout, max_retries=max_retries)
        self.model = model

    def complete(self, system: str, user: str, logprobs: bool = False) -> Completion:
        # Asking for logprobs only adds them to the response; it does not change
        # how the reply is decoded, so the prompt and generation params are as frozen.
        extra = {"logprobs": True, "top_logprobs": TOP_LOGPROBS} if logprobs else {}
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "signal", "strict": True, "schema": RESPONSE_SCHEMA},
                },
                **GENERATION_PARAMS,
                **extra,
            )
        except self._openai.OpenAIError as exc:
            raise LLMCallError(f"{type(exc).__name__}: {exc}") from exc
        choice = resp.choices[0]
        tokens = None
        if logprobs and choice.logprobs and choice.logprobs.content:
            tokens = [
                {"token": t.token, "logprob": t.logprob,
                 "top": [[a.token, a.logprob] for a in t.top_logprobs]}
                for t in choice.logprobs.content
            ]
        return Completion(
            text=choice.message.content or "",
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            model=resp.model,
            token_logprobs=tokens,
        )
