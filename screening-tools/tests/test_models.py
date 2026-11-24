# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import openvino as ov
import pytest
from sklearn.neural_network import MLPClassifier

from screening_tools.hit_scoring import HitScorer, build_ov_model
from screening_tools.model_store import ModelError


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_openvino_graph_matches_sklearn():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 13)).astype(np.float32)
    y = np.column_stack([x[:, 0] > 0, x[:, 1] > 0.5]).astype(int)
    clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=50, random_state=0).fit(x, y)
    compiled = ov.Core().compile_model(build_ov_model(clf.coefs_, clf.intercepts_), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    np.testing.assert_allclose(compiled(x)[0], clf.predict_proba(x), atol=1e-5)


def test_scorer_is_batch_size_independent(store):
    x = np.random.default_rng(2).normal(size=(300, HitScorer(store, "CPU", 1).n_inputs)).astype(np.float32)
    a1, n1 = HitScorer(store, "CPU", 1).score(x)
    a2, n2 = HitScorer(store, "CPU", 128).score(x)
    np.testing.assert_allclose(a1, a2, atol=1e-5)
    np.testing.assert_allclose(n1, n2, atol=1e-5)
    assert a1.shape == (300,)


def test_tampered_model_is_refused(store, tmp_path):
    import shutil

    from screening_tools.model_store import ModelStore

    copy = ModelStore(tmp_path / "m")
    shutil.copytree(store.directory, copy.directory)
    with copy.path("anomaly_scaler.json").open("a", encoding="utf-8") as fh:
        fh.write(" ")
    try:
        copy.load_anomaly()
    except ModelError as exc:
        assert "anomaly_scaler.json" in str(exc)
    else:
        raise AssertionError("a modified model file was loaded")
