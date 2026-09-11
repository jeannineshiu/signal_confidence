"""CORP reliability diagram and Brier decomposition of the primary run (PLAN.md §9).

    python -m sigconf.experiments.corp_decomposition [--split dev|test]

Reads the primary run's committed signal cache (never calls the LLM) and
re-reads the same scored calls without bins: the PAV-recalibrated reliability
curve, and Brier = MCB − DSC + UNC, with MCB judged against a perfectly
calibrated forecaster and DSC against confidence unrelated to correctness.

This analysis was added after the primary and secondary results were known.
It does not revise the primary result (results/metrics.json, which keeps the
pre-registered binned ECE); it has its own outputs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sigconf import config
from sigconf.data.labels import UP
from sigconf.eval.corp import corp_analysis
from sigconf.pipeline import README_PATH, run_split
from sigconf.report import figures, summary


def output_paths(split: str) -> dict[str, Path]:
    base = config.RESULTS_DIR / "corp" / split
    figure = (config.IMG_DIR / "corp_reliability.png" if split == "test"
              else base / "corp_reliability.png")
    return {"metrics": base / "metrics.json", "figure": figure}


def _no_client(_model):
    raise AssertionError("the CORP analysis reads the committed cache only")


def run(split_name: str, log=print) -> tuple[dict, pd.DataFrame]:
    primary, scored = run_split(split_name, offline=True, client_factory=_no_client, log=log)
    rows = scored[scored["state"] == "scored"]
    conf = rows["confidence"].to_numpy(dtype=float)
    hit = rows["hit"].to_numpy(dtype=float)
    # Robustness: the same Brier, decomposed on P(up) against the up/down label.
    p_up = np.where(rows["direction"] == "bullish", conf, 1 - conf)
    went_up = (rows["label"] == UP).to_numpy(dtype=float)

    metrics = {
        "split": split_name,
        "hit_form": corp_analysis(conf, hit),
        "p_up_form": corp_analysis(p_up, went_up),
        "accuracy": primary["accuracy"]["value"],
        "run": primary["run"],
    }
    return metrics, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sigconf.experiments.corp_decomposition",
                                     description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    args = parser.parse_args(argv)

    metrics, rows = run(args.split)
    paths = output_paths(args.split)
    summary.write_metrics(metrics, paths["metrics"])

    m = metrics["hit_form"]
    label = "held-out test" if args.split == "test" else "dev split"
    figures.plot_corp_reliability(
        pd.DataFrame(m["curve"]), rows["confidence"].to_numpy(dtype=float), paths["figure"],
        f"CORP reliability diagram · AAPL {label}",
        f"Brier {m['brier']:.3f} = miscalibration {m['mcb']['value']:.3f} "
        f"− discrimination {m['dsc']['value']:.3f} + uncertainty {m['unc']:.3f}")

    block = summary.render_corp_block(metrics)
    if args.split == "test" and README_PATH.exists():
        text = README_PATH.read_text()
        if summary.CORP_START in text:
            README_PATH.write_text(summary.replace_results_block(
                text, block, summary.CORP_START, summary.CORP_END))
    print(f"[corp] wrote {', '.join(str(p.relative_to(config.ROOT)) for p in paths.values())}")
    print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
