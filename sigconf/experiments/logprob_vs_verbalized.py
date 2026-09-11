"""Secondary analysis: token-probability vs verbalised confidence (PLAN.md §8).

    python -m sigconf.experiments.logprob_vs_verbalized [--split dev|test] [--offline]

Re-runs the frozen prompt v1 with logprobs on the same headlines and reads two
confidences from each reply: the number the model wrote, and the probability
the model gave the direction token it chose. Both describe the same call, so
the comparison is paired and accuracy is identical by construction.

This analysis was added after the primary result was known. It does not
revise the primary result (sigconf/pipeline.py, results/metrics.json); its
definitions were fixed and tagged (`logprob-preregistered`) before its own
test run, and it has its own cache and outputs.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from sigconf import config
from sigconf.eval.calibration import reliability_table
from sigconf.eval.compare import compare_confidences
from sigconf.eval.scoring import SIGNAL_COLUMNS, funnel, score
from sigconf.pipeline import README_PATH, load_labeled
from sigconf.report import figures, summary
from sigconf.signals.client import OpenAIClient
from sigconf.signals.generate import SignalCache, run_signals
from sigconf.signals.prompt import PROMPT_SHA, PROMPT_VERSION

VERBAL_COLOUR = figures.SERIES      # categorical slot 1
LOGPROB_COLOUR = "#eb6834"          # categorical slot 2 (validated adjacent to slot 1)


def output_paths(split: str) -> dict[str, Path]:
    base = config.RESULTS_DIR / "logprob" / split
    figure = (config.IMG_DIR / "logprob_vs_verbalized.png" if split == "test"
              else base / "logprob_vs_verbalized.png")
    return {"metrics": base / "metrics.json", "figure": figure}


def _agreement(items: pd.DataFrame, rerun: pd.DataFrame, model: str) -> dict:
    """How the logprob re-run compares with the primary run on the same headlines."""
    primary = SignalCache(config.SIGNALS_CACHE_PATH)
    pairs = [(primary.get(i, model), r) for i, r in zip(items["id"], rerun.to_dict("records"),
                                                          strict=True)]
    both_ok = [(p, r) for p, r in pairs if p and p["status"] == "ok" and r["status"] == "ok"]
    same_dir = sum(p["direction"] == r["direction"] for p, r in both_ok)
    same_conf = sum(p["confidence"] == r["confidence"] for p, r in both_ok)
    return {"n": len(both_ok), "direction_same": f"{same_dir}/{len(both_ok)}",
            "verbalized_confidence_same": f"{same_conf}/{len(both_ok)}"}


def run(split_name: str, offline: bool = False, client_factory=OpenAIClient,
        log=print) -> tuple[dict, pd.DataFrame]:
    labeled = load_labeled(offline=offline)
    items = labeled[labeled["split"] == split_name].reset_index(drop=True)
    settings = config.llm_settings()
    signals = run_signals(items, settings, client_factory, SignalCache(config.LOGPROB_CACHE_PATH),
                          offline=offline, log=log, with_logprobs=True)

    extra = ["id", "logprob_confidence", "logprob_error", "opposite_in_top_k"]
    scored = score(items, signals[SIGNAL_COLUMNS]).merge(signals[extra], on="id")
    directional = scored[scored["state"] == "scored"]
    both = directional[directional["logprob_confidence"].notna()]
    log(f"[logprob] {split_name}: {len(directional)} scored calls, "
        f"{len(both)} with a token probability")

    metrics = {
        "split": split_name,
        "funnel": funnel(scored),
        "logprob_missing": int(len(directional) - len(both)),
        "opposite_not_in_top_k": int((~both["opposite_in_top_k"].astype(bool)).sum()),
        "comparison": compare_confidences(
            both["confidence"], both["logprob_confidence"], both["hit"].astype(float),
            names=("verbalized", "logprob")),
        "rerun_vs_primary_run": _agreement(items, signals, settings.model),
        "run": {"model": settings.model, "prompt_version": PROMPT_VERSION,
                "prompt_sha": PROMPT_SHA, "logprobs": True,
                "llm_calls": int(signals["attempts"].sum()),
                "llm_cost_usd": float(signals["cost_usd"].sum())},
    }
    return metrics, both


def main(argv: list[str] | None = None, env_file: Path | None = config.ROOT / ".env") -> int:
    parser = argparse.ArgumentParser(prog="sigconf.experiments.logprob_vs_verbalized",
                                     description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if env_file is not None and env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file)

    metrics, both = run(args.split, offline=args.offline)
    paths = output_paths(args.split)
    summary.write_metrics(metrics, paths["metrics"])

    hit = both["hit"].to_numpy(dtype=float)
    panels = []
    for label, column, colour in (("Verbalised confidence", "confidence", VERBAL_COLOUR),
                                  ("Token probability (logprob)", "logprob_confidence",
                                   LOGPROB_COLOUR)):
        conf = both[column].to_numpy(dtype=float)
        panels.append((label, reliability_table(conf, hit), conf, colour))
    split_label = "held-out test" if args.split == "test" else "dev split"
    figures.plot_confidence_comparison(
        panels, paths["figure"],
        f"Two confidences for the same {len(both)} calls · AAPL {split_label}",
        "Left: the number the model wrote · right: the probability it gave the direction "
        "token it chose · bars: Wilson 95% interval")

    block = summary.render_logprob_block(metrics)
    if args.split == "test" and README_PATH.exists():
        text = README_PATH.read_text()
        if summary.LOGPROB_START in text:
            README_PATH.write_text(summary.replace_results_block(
                text, block, summary.LOGPROB_START, summary.LOGPROB_END))
    print(f"[logprob] wrote {', '.join(str(p.relative_to(config.ROOT)) for p in paths.values())}")
    print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
