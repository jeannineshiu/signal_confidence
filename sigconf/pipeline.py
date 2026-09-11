"""End-to-end entry point: python -m sigconf.pipeline [--split dev|test] [--offline].

  1. committed headline sample + cached prices → next-day labels
  2. LLM signals: from data/cache/signals.jsonl, calling the model only for
     misses, behind the model-cutoff guard and the spend caps
  3. scoring (coverage funnel) and every metric next to its baseline
  4. metrics JSON, figures, and — for the test split — the README results block

Test-split outputs go to results/metrics.json and img/; dev-split outputs go
to results/dev/ so prompt-iteration numbers can never be mistaken for results.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from sigconf import config
from sigconf.data import headlines, labels, prices
from sigconf.eval.calibration import reliability_table
from sigconf.eval.evaluate import evaluate
from sigconf.eval.scoring import score
from sigconf.report import figures, summary
from sigconf.signals.client import OpenAIClient
from sigconf.signals.generate import SignalCache, run_signals
from sigconf.signals.prompt import PROMPT_SHA, PROMPT_VERSION

README_PATH = config.ROOT / "README.md"


def load_labeled(offline: bool = False) -> pd.DataFrame:
    """Stage 1: the committed headline sample with next-day labels attached."""
    sample = headlines.load_sample()
    closes = prices.load_closes(
        config.TICKER, config.PRICE_START, config.PRICE_END, offline=offline
    )
    labeled = labels.label_headlines(sample, closes)
    return labeled.sort_values(["published_at", "id"]).reset_index(drop=True)


def output_paths(split: str) -> dict[str, Path]:
    if split == "test":
        return {"metrics": config.RESULTS_DIR / "metrics.json",
                "calibration": config.IMG_DIR / "calibration_curve.png",
                "baselines": config.IMG_DIR / "baselines.png"}
    base = config.RESULTS_DIR / split
    return {"metrics": base / "metrics.json",
            "calibration": base / "calibration_curve.png",
            "baselines": base / "baselines.png"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sigconf.pipeline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--split", choices=("dev", "test"), default="test",
        help="dev is for prompt iteration only; every reported number comes from test",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="treat a signal-cache miss as an error instead of calling the LLM",
    )
    parser.add_argument(
        "--retry-api-errors", action="store_true",
        help="re-call the LLM for cached items whose call failed (api_error)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, env_file: Path | None = config.ROOT / ".env") -> int:
    args = parse_args(argv)
    if env_file is not None and env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file)  # only here — importing sigconf never reads .env

    labeled = load_labeled(offline=args.offline)
    split = labeled[labeled["split"] == args.split].reset_index(drop=True)
    print(f"[1/4] labels ({args.split}): {len(split)} headlines, "
          f"{split['label'].value_counts().to_dict()}")

    settings = config.llm_settings()
    signals = run_signals(split, settings, OpenAIClient, SignalCache(),
                          offline=args.offline, retry_api_errors=args.retry_api_errors)
    print(f"[2/4] signals ({settings.model}, prompt {PROMPT_VERSION}): "
          f"{signals['status'].value_counts().to_dict()}")

    scored = score(split, signals)
    metrics = evaluate(scored)
    metrics["run"] = {
        "split": args.split,
        "model": settings.model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha": PROMPT_SHA,
        "llm_calls": int(signals["attempts"].sum()),
        "llm_cost_usd": float(signals["cost_usd"].sum()),
    }
    print(f"[3/4] scored: {metrics['funnel']}")

    paths = output_paths(args.split)
    summary.write_metrics(metrics, paths["metrics"])
    rows = scored[scored["state"] == "scored"]
    conf = rows["confidence"].to_numpy(dtype=float)
    table = reliability_table(conf, rows["hit"].to_numpy(dtype=float))
    label = "held-out test" if args.split == "test" else "dev split (prompt iteration only)"
    figures.plot_calibration_curve(table, conf, paths["calibration"],
                                   f"LLM confidence vs outcome · AAPL, {label}")
    figures.plot_baselines(metrics, paths["baselines"], f"Directional accuracy · AAPL, {label}")

    block = summary.render_results_block(metrics)
    if args.split == "test" and README_PATH.exists():
        README_PATH.write_text(summary.replace_results_block(README_PATH.read_text(), block))
    print(f"[4/4] wrote {', '.join(str(p.relative_to(config.ROOT)) for p in paths.values())}")
    print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
