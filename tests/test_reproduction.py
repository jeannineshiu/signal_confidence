"""Every committed number must be recomputable from the committed caches alone.

Runs the pipeline offline (a cache miss is an error and no client may be
built) and compares with the committed metrics JSON. Counts, labels and keys
must match exactly; floats to 1e-12 relative, because summation order in
vectorised numpy can differ across CPU architectures in the last digit (CI
runs on Linux/x86, development on macOS/ARM). Printed figures are unaffected.
"""

import json
import math

import pytest

from sigconf.pipeline import output_paths, run_split


def refuse(_model):
    raise AssertionError("reproduction must not call the LLM")


def assert_same(got, want, path="metrics"):
    if isinstance(want, dict):
        assert isinstance(got, dict) and got.keys() == want.keys(), path
        for key in want:
            assert_same(got[key], want[key], f"{path}.{key}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            assert_same(g, w, f"{path}[{i}]")
    elif isinstance(want, float):
        assert isinstance(got, float | int) and math.isclose(got, want, rel_tol=1e-12,
                                                             abs_tol=1e-15), path
    else:
        assert got == want and type(got) is type(want), path


@pytest.mark.parametrize("split", ["dev", "test"])
def test_committed_metrics_reproduce_offline(split):
    committed_path = output_paths(split)["metrics"]
    if not committed_path.exists():
        pytest.skip(f"no committed {split} metrics yet")
    metrics, _ = run_split(split, offline=True, client_factory=refuse, log=lambda _: None)
    assert_same(json.loads(json.dumps(metrics)), json.loads(committed_path.read_text()))


@pytest.mark.parametrize("split", ["dev", "test"])
def test_committed_logprob_metrics_reproduce_offline(split):
    from sigconf.experiments import logprob_vs_verbalized as lp

    committed_path = lp.output_paths(split)["metrics"]
    if not committed_path.exists():
        pytest.skip(f"no committed logprob {split} metrics yet")
    metrics, _ = lp.run(split, offline=True, client_factory=refuse, log=lambda _: None)
    assert_same(json.loads(json.dumps(metrics)), json.loads(committed_path.read_text()))


def test_readme_logprob_block_matches_committed_metrics():
    from sigconf.experiments import logprob_vs_verbalized as lp
    from sigconf.pipeline import README_PATH
    from sigconf.report import summary

    block = summary.render_logprob_block(summary.read_metrics(lp.output_paths("test")["metrics"]))
    assert block in README_PATH.read_text()


def test_readme_results_block_matches_committed_metrics():
    """The README's numbers table is exactly what the committed metrics render to."""
    from sigconf.pipeline import README_PATH
    from sigconf.report import summary

    text = README_PATH.read_text()
    block = summary.render_results_block(summary.read_metrics(output_paths("test")["metrics"]))
    assert block in text
    assert summary.replace_results_block(text, block) == text


def test_the_comparator_is_strict_where_it_must_be():
    with pytest.raises(AssertionError):
        assert_same({"n": 31}, {"n": 30})
    with pytest.raises(AssertionError):
        assert_same({"a": 0.5}, {"a": 0.5000001})
    with pytest.raises(AssertionError):
        assert_same({"a": 1}, {"a": 1, "b": 2})
    assert_same({"a": 0.1 + 0.2}, {"a": 0.3})  # last-digit difference tolerated
