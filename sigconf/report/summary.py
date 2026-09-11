"""metrics.json and the auto-generated README results block.

The README's numbers are rendered from metrics.json between two markers, and
a test asserts the committed block matches — so the README cannot drift from
what the code computes. Interpretive sentences are written by hand outside
the markers, after reading the underlying rows.
"""

import json
from pathlib import Path

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def write_metrics(metrics: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return path


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _ci(ci: list[float], fmt=_pct) -> str:
    return f"95% CI {fmt(ci[0])}–{fmt(ci[1])}"


def _p(p: float) -> str:
    """Never print "p = 0.000": a simulated or exact p-value is only ever small."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def render_results_block(m: dict) -> str:
    acc, f = m["accuracy"], m["funnel"]
    up = m["baselines"]["always_up_same_items"]
    coin = m["baselines"]["coin_flip"]
    mc = m["baselines"]["llm_vs_always_up_mcnemar"]
    b, e = m["brier"], m["ece"]
    null = e["perfect_calibration_null"]
    num = "{:.3f}".format

    lines = [
        START,
        "| Metric | LLM signal | Baseline | Sample |",
        "|---|---|---|---|",
        f"| Directional accuracy | **{_pct(acc['value'])}** ({_ci(acc['ci95'])}) "
        f"| always-up {_pct(up['value'])} ({_ci(up['ci95'])}); coin flip 50% "
        f"(random 95% range {_pct(coin['range95_at_n'][0])}–{_pct(coin['range95_at_n'][1])}) "
        f"| n = {acc['n']} scored of {m['n_headlines']} headlines "
        f"({_pct(m['coverage'])} coverage) |",
        f"| Brier score (lower is better) | **{num(b['value'])}** ({_ci(b['ci95'], num)}) "
        f"| {num(b['coin_flip_reference'])} for always saying 50% "
        f"(skill {b['skill_vs_coin_flip']:+.3f}) | n = {acc['n']} |",
        f"| Expected Calibration Error | **{num(e['value'])}** ({_ci(e['ci95'], num)}) "
        f"| {num(null['mean'])} expected from a perfectly calibrated model at this n "
        f"(simulated; {_p(null['p_value'])}) | n = {acc['n']} |",
        "",
        f"LLM vs always-up on the same items (exact McNemar): LLM-only correct "
        f"{mc['a_only']}, always-up-only correct {mc['b_only']}, {_p(mc['p_value'])}. "
        f"Mean stated confidence {_pct(m['confidence']['mean'])} vs accuracy "
        f"{_pct(acc['value'])}.",
        "",
        f"Coverage funnel: {f['total']} headlines → {f['api_error']} API errors, "
        f"{f['parse_failure']} parse failures, {f['neutral']} neutral, "
        f"{f['flat_label']} flat next-day moves → {f['scored']} scored.",
        END,
    ]
    return "\n".join(lines)


def replace_results_block(text: str, block: str) -> str:
    """Swap the text between the markers (inclusive) for `block`."""
    if text.count(START) != 1 or text.count(END) != 1 or text.index(START) > text.index(END):
        raise ValueError("README must contain exactly one RESULTS:START … RESULTS:END pair")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return head + block + tail
