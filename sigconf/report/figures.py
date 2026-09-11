"""Static figures for the README (spec: matplotlib PNGs in img/, no UI).

Styling follows one small palette: the forecaster in blue, baselines and
references in neutral ink, recessive axes, no dashed rules. Every mark's
identity is carried by a text label as well as colour.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
BASELINE = "#b4b2aa"

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "font.family": "sans-serif",
    "font.size": 9.5,
}


def plot_calibration_curve(
    table: pd.DataFrame, confidence: np.ndarray, path: Path, title: str
) -> Path:
    """Reliability diagram (top) over the distribution of stated confidence (bottom).

    `table` is calibration.reliability_table(): x = mean confidence in each
    bin, y = empirical accuracy with its Wilson 95% interval.
    """
    with plt.rc_context(RC):
        fig, (ax, hist) = plt.subplots(
            2, 1, figsize=(6.4, 7.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        _reliability_panel(ax, table, SERIES)
        ax.set_ylabel("Empirical accuracy (share of calls that were right)")
        _title(ax, title, "Points: bins of stated confidence · bars: Wilson 95% interval")
        _distribution_panel(hist, confidence, SERIES)
        hist.set_ylabel("Calls")
        _save(fig, path)
    return path


def plot_confidence_comparison(
    panels: list[tuple[str, pd.DataFrame, np.ndarray, str]], path: Path, title: str,
    subtitle: str,
) -> Path:
    """Small multiples: one column per confidence readout of the same calls.

    Each panel is (label, reliability table, confidences, colour). Same axes in
    every column, so the columns can be compared by eye.
    """
    with plt.rc_context(RC):
        fig, axes = plt.subplots(
            2, len(panels), figsize=(5.2 * len(panels), 7.0), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}, squeeze=False,
        )
        for col, (label, table, confidence, colour) in enumerate(panels):
            ax, hist = axes[0, col], axes[1, col]
            if col:  # accuracy axes are shared; each count axis keeps its own scale
                ax.sharey(axes[0, 0])
                ax.tick_params(labelleft=False)
            _reliability_panel(ax, table, colour)
            ax.text(0, 1.02, label, transform=ax.transAxes, color=INK, fontsize=10,
                    fontweight="bold", va="bottom")
            _distribution_panel(hist, confidence, colour)
        axes[0, 0].set_xlim(0, 1.03)  # room for marks at exactly 1.0
        axes[0, 0].set_ylabel("Empirical accuracy")
        axes[1, 0].set_ylabel("Calls")
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
        fig.text(0.01, 0.935, subtitle, ha="left", color=INK_2, fontsize=8.5)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        _save(fig, path)
    return path


def _reliability_panel(ax, table: pd.DataFrame, colour: str) -> None:
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, zorder=1)
    # Label the reference in the empty lower-left, rotated to the line's on-screen angle.
    ax.text(0.12, 0.14, "perfectly calibrated", color=MUTED, ha="left", va="bottom",
            rotation=45, rotation_mode="anchor", transform_rotates_text=True, fontsize=8.5)

    x, y = table["mean_confidence"], table["accuracy"]
    yerr = [y - table["ci_low"], table["ci_high"] - y]
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=colour, elinewidth=1.4, capsize=0,
                alpha=0.55, zorder=2)
    ax.plot(x, y, color=colour, linewidth=2, zorder=3)
    ax.scatter(x, y, s=64, color=colour, edgecolor=SURFACE, linewidth=2, zorder=4)
    ys = list(y)
    for i, (xi, yi, n) in enumerate(zip(x, ys, table["n"], strict=True)):
        # Put the label on the side the line to the next point does not use.
        rising = i + 1 < len(ys) and ys[i + 1] > yi
        ax.annotate(f"n={n}", (xi, yi), xytext=(7, -6 if rising else 5),
                    textcoords="offset points", va="top" if rising else "bottom",
                    color=INK_2, fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)


def _distribution_panel(hist, confidence: np.ndarray, colour: str) -> None:
    values, counts = np.unique(np.asarray(confidence, dtype=float), return_counts=True)
    if len(values) <= 20:
        # Verbalised confidence is discrete: one bar per value actually stated,
        # centred on it (a binned histogram would misplace values on bin edges).
        hist.bar(values, counts, width=0.018, color=colour, zorder=2)
        for v, k in zip(values, counts, strict=True):
            hist.annotate(f"{v:g}", (v, k), xytext=(0, 3), textcoords="offset points",
                          ha="center", color=INK_2, fontsize=8)
        hist.set_ylim(0, counts.max() * 1.25)
    else:
        hist.hist(confidence, bins=np.linspace(0, 1, 41), color=colour, edgecolor=SURFACE,
                  linewidth=1.0)
    hist.set_xlabel("Stated confidence")
    hist.grid(axis="x", visible=False)


def _save(fig, path: Path) -> None:
    if not fig.get_suptitle():  # figures with a suptitle lay themselves out
        fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, metadata={"Software": None})
    plt.close(fig)


def _title(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=8.5, va="bottom")


def plot_baselines(
    metrics: dict, path: Path, title: str, forecaster: str = "LLM signal"
) -> Path:
    """Directional accuracy: forecaster vs coin flip vs always-up, same items."""
    acc = metrics["accuracy"]
    up = metrics["baselines"]["always_up_same_items"]
    coin = metrics["baselines"]["coin_flip"]
    bars = [
        (forecaster, acc["value"], acc["ci95"], SERIES),
        ("Coin flip", 0.5, coin["range95_at_n"], BASELINE),
        ("Always up\n(buy-and-hold)", up["value"], up["ci95"], BASELINE),
    ]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        xs = np.arange(len(bars))
        for x, (_label, value, ci, colour) in zip(xs, bars, strict=True):
            ax.bar(x, value, width=0.42, color=colour, zorder=2)
            ax.vlines(x, ci[0], ci[1], color=INK_2, linewidth=1.4, zorder=3)
            ax.text(x + 0.25, value, f"{value:.1%}", va="center", ha="left", color=INK,
                    fontsize=10)
        ax.set_xticks(xs, [b[0] for b in bars])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Directional accuracy")
        _title(ax, title, f"Same {acc['n']} scored items · lines: 95% interval "
               "(coin flip: range of a random guesser at this n)")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, metadata={"Software": None})
        plt.close(fig)
    return path
