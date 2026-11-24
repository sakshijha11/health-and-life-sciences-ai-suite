# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from screening_tools.config import load_config  # noqa: E402
from screening_tools.model_store import ModelStore  # noqa: E402

SMALL_MODEL = {
    "model": {
        "hidden_layers": [64, 32],
        "epochs": 4,
        "train_plates": 60,
        "val_plates": 10,
        "test_plates": 10,
        "anomaly_trees": 50,
    },
}


@pytest.fixture(scope="session")
def cfg():
    return load_config(overrides={**SMALL_MODEL, "run": {"plates": 30, "run_id": "test"}, "hit_scoring": {"device": "CPU"}})


@pytest.fixture(scope="session")
def store(cfg, tmp_path_factory):
    from screening_tools.setup_models import run_setup

    s = ModelStore(tmp_path_factory.mktemp("models"))
    run_setup(cfg, s, log=lambda *_: None)
    return s
