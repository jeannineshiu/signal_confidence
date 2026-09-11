from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES_538 = Path(__file__).parent / "fixtures" / "538"


def load_538(name: str) -> pd.DataFrame:
    """Real FiveThirtyEight game forecasts; tied games (outcome 0.5) removed."""
    df = pd.read_csv(FIXTURES_538 / f"{name}_games.csv")
    return df[df["prob1_outcome"] != 0.5].reset_index(drop=True)


def favourite_form(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """538 rows in this project's shape: call = the favourite, confidence = its
    win probability (≥ 0.5), hit = whether the favourite won."""
    team1_favoured = df["prob1"] >= df["prob2"]
    conf = np.where(team1_favoured, df["prob1"], df["prob2"])
    hit = np.where(team1_favoured, df["prob1_outcome"], df["prob2_outcome"])
    return conf.astype(float), hit.astype(float)


@pytest.fixture(scope="session")
def nba_538() -> pd.DataFrame:
    return load_538("nba")


@pytest.fixture(scope="session")
def nfl_538() -> pd.DataFrame:
    return load_538("nfl")


@pytest.fixture(autouse=True)
def _no_real_llm_calls(monkeypatch):
    """Strip API credentials so no test can spend money by accident.

    Signal-generation tests inject a fake client; if any code path ever
    reaches the real OpenAI client, it fails loudly for lack of a key.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
