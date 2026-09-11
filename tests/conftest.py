import pytest


@pytest.fixture(autouse=True)
def _no_real_llm_calls(monkeypatch):
    """Strip API credentials so no test can spend money by accident.

    Signal-generation tests inject a fake client; if any code path ever
    reaches the real OpenAI client, it fails loudly for lack of a key.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
