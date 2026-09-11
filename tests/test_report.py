import numpy as np
import pytest
from conftest import favourite_form
from test_scoring_evaluate import _538_as_scored

from sigconf.eval.calibration import reliability_table
from sigconf.eval.evaluate import evaluate
from sigconf.report import figures, summary


@pytest.fixture(scope="module")
def nfl(nfl_538):
    conf, hit = favourite_form(nfl_538)
    metrics = evaluate(_538_as_scored(nfl_538), n_boot=500, n_sims=500)
    return conf, hit, metrics


def test_metrics_round_trip(tmp_path, nfl):
    _, _, m = nfl
    path = summary.write_metrics(m, tmp_path / "metrics.json")
    assert summary.read_metrics(path) == m


def test_results_block_states_value_baseline_and_sample_together(nfl):
    _, _, m = nfl
    block = summary.render_results_block(m)
    assert block.startswith(summary.START) and block.endswith(summary.END)
    acc_row = next(line for line in block.splitlines() if line.startswith("| Directional"))
    assert "always-up" in acc_row and "coin flip" in acc_row
    assert f"n = {m['accuracy']['n']}" in acc_row
    assert "simulated" in block  # the ECE null is labelled as a simulation
    assert "Coverage funnel" in block


def test_small_p_values_are_never_printed_as_zero():
    assert summary._p(0.0) == "p < 0.001"
    assert summary._p(0.00099) == "p < 0.001"
    assert summary._p(0.0421) == "p = 0.042"


def test_replace_results_block_only_touches_the_marked_region():
    readme = f"# Title\nintro\n{summary.START}\nold numbers\n{summary.END}\noutro\n"
    new = summary.replace_results_block(readme, f"{summary.START}\nnew\n{summary.END}")
    assert new == f"# Title\nintro\n{summary.START}\nnew\n{summary.END}\noutro\n"


@pytest.mark.parametrize("text", ["no markers", f"{summary.END} {summary.START}",
                                  f"{summary.START}{summary.START}{summary.END}"])
def test_replace_results_block_rejects_malformed_readmes(text):
    with pytest.raises(ValueError):
        summary.replace_results_block(text, "x")


def test_figures_render_from_real_forecasts(tmp_path, nfl):
    conf, hit, m = nfl
    cal = figures.plot_calibration_curve(reliability_table(conf, hit), conf,
                                         tmp_path / "cal.png", "538 NFL favourites")
    base = figures.plot_baselines(m, tmp_path / "base.png", "538 NFL favourites")
    for path in (cal, base):
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 10_000


def test_figures_accept_a_single_bin(tmp_path):
    conf = np.full(5, 0.7)
    hit = np.array([1, 1, 0, 1, 0])
    path = figures.plot_calibration_curve(reliability_table(conf, hit), conf,
                                          tmp_path / "one.png", "one bin")
    assert path.stat().st_size > 0
