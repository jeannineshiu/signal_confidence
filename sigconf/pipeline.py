"""End-to-end entry point: python -m sigconf.pipeline [--split dev|test] [--offline].

Stages (filled in phase by phase, see PLAN.md §3):
  1. load sampled headlines + cached prices → next-day labels     (Phase 1)
  2. evaluation harness over a signals file                        (Phase 2)
  3. LLM signal generation behind the cost guard and cache         (Phase 3)
  4. figures, metrics.json and the README results block            (Phase 2/4)
"""

import argparse
import sys


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
    parse_args(argv)
    print("sigconf.pipeline: no stages implemented yet (see PLAN.md §3)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
