"""The structured output every LLM call must return (spec §5).

Validated in our own code with Pydantic, never trusted from the provider, so
the retry-once path in generate.py is ours and testable with a fake client.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Signal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=300)


# JSON schema sent as the response format. Kept deliberately plain (no numeric
# bounds): the provider's strict mode does not guarantee every keyword, and the
# bounds are enforced locally by Signal anyway.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["direction", "confidence", "reasoning"],
    "additionalProperties": False,
}
