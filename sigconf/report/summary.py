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


def replace_results_block(text: str, block: str, start: str = START, end: str = END) -> str:
    """Swap the text between the markers (inclusive) for `block`."""
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) > text.index(end):
        raise ValueError(f"README must contain exactly one {start} … {end} pair")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return head + block + tail


LOGPROB_START = "<!-- LOGPROB:START -->"
LOGPROB_END = "<!-- LOGPROB:END -->"


def render_logprob_block(m: dict) -> str:
    """README table for the secondary analysis (token probability vs verbalised)."""
    c = m["comparison"]
    num = "{:.3f}".format

    def fine_pct(x: float) -> str:
        # 0.9996 must not print as "100.0%": that would claim certainty the model never stated.
        return f"{x:.2%}" if x >= 0.995 else _pct(x)

    def row(label: str, s: dict) -> str:
        e, a = s["ece"], s["auroc"]
        auc = "n/a" if a["value"] is None else f"{num(a['value'])} ({_ci(a['ci95'], num)})"
        return (f"| {label} | {fine_pct(s['mean_confidence'])} "
                f"| {num(s['brier']['value'])} ({_ci(s['brier']['ci95'], num)}) "
                f"| {num(e['value'])} (perfect-calibration reference "
                f"{num(e['perfect_calibration_null']['mean'])}; "
                f"{_p(e['perfect_calibration_null']['p_value'])}) | {auc} |")

    d = c["logprob_minus_verbalized"]

    def diff(stat: str) -> str:
        v = d[stat]
        if v["value"] is None:
            return "n/a"
        return f"{v['value']:+.3f} ({_ci(v['ci95'], lambda x: f'{x:+.3f}')})"

    agree = m["rerun_vs_primary_run"]
    lines = [
        LOGPROB_START,
        "| Confidence readout | Mean confidence | Brier (lower is better) | ECE | "
        "AUROC (0.5 = chance) |",
        "|---|---|---|---|---|",
        row("Verbalised (the number in the reply)", c["verbalized"]),
        row("Token probability of the chosen direction", c["logprob"]),
        f"| **Difference**, token − verbalised (paired bootstrap) | "
        f"{100 * (c['logprob']['mean_confidence'] - c['verbalized']['mean_confidence']):+.1f} pp "
        f"| {diff('brier')} | {diff('ece')} | {diff('auroc')} |",
        "",
        f"Same {c['n']} directional calls for both readouts (accuracy {_pct(c['accuracy'])} "
        f"for both, since both come from the same reply). "
        f"Token probability ≥ 0.99 on {_pct(c['logprob']['share_at_or_above_0.99'])} of calls; "
        f"opposite direction outside the top-20 alternatives on "
        f"{m['opposite_not_in_top_k']} call(s) (an upper bound there); "
        f"token probability unavailable on {m['logprob_missing']} call(s).",
        "",
        f"Re-run vs the primary run on the same {agree['n']} headlines: same direction on "
        f"{agree['direction_same']}, same verbalised confidence on "
        f"{agree['verbalized_confidence_same']}.",
        LOGPROB_END,
    ]
    return "\n".join(lines)
