"""Every committed number must be recomputable from the committed caches alone.

Runs the pipeline offline (a cache miss is an error and no client may be
built) and compares with the committed metrics JSON, value for value.
"""

import json

import pytest

from sigconf.pipeline import output_paths, run_split


def refuse(_model):
    raise AssertionError("reproduction must not call the LLM")


@pytest.mark.parametrize("split", ["dev", "test"])
def test_committed_metrics_reproduce_offline(split):
    committed_path = output_paths(split)["metrics"]
    if not committed_path.exists():
        pytest.skip(f"no committed {split} metrics yet")
    metrics, _ = run_split(split, offline=True, client_factory=refuse, log=lambda _: None)
    assert json.loads(json.dumps(metrics)) == json.loads(committed_path.read_text())
