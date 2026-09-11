"""End-to-end entry point: python -m sigconf.pipeline [--split dev|test] [--offline].

Stages (filled in phase by phase, see PLAN.md §3):
  1. load sampled headlines + cached prices → next-day labels     (Phase 1)
  2. evaluation harness over a signals file                        (Phase 2)
  3. LLM signal generation behind the cost guard and cache         (Phase 3)
  4. figures, metrics.json and the README results block            (Phase 2/4)
"""

import argparse
import sys

import pandas as pd

from sigconf import config
from sigconf.data import headlines, labels, prices


def load_labeled(offline: bool = False) -> pd.DataFrame:
    """Stage 1: the committed headline sample with next-day labels attached."""
    sample = headlines.load_sample()
    closes = prices.load_closes(
        config.TICKER, config.PRICE_START, config.PRICE_END, offline=offline
    )
    labeled = labels.label_headlines(sample, closes)
    return labeled.sort_values(["published_at", "id"]).reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sigconf.pipeline", description=__doc__)
    parser.add_argument(
        "--split",
        choices=("dev", "test"),
        default="test",
        help="dev is for prompt iteration only; every reported number comes from test",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="treat a signal-cache miss as an error instead of calling the LLM",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labeled = load_labeled(offline=args.offline)
    split = labeled[labeled["split"] == args.split]
    print(f"[1/4] labels ({args.split}): {len(split)} headlines")
    print(split["label"].value_counts().to_string())

    print("sigconf.pipeline: stages 2-4 not implemented yet (see PLAN.md §3)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
