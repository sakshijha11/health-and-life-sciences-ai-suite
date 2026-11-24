# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from screening_tools.config import ConfigError, load_config, without_faults


def test_defaults_load():
    cfg = load_config()
    assert cfg["plate"]["rows"] * cfg["plate"]["cols"] == 96
    assert cfg["hit_scoring"]["device"] in ("CPU", "GPU")


@pytest.mark.parametrize(
    "override, message",
    [
        ({"qc": {"zprime_fial": 0.3}}, "unknown configuration key"),
        ({"qc": {"zprime_fail": "low"}}, "must be float"),
        ({"run": {"plates": 1.5}}, "must be int"),
        ({"hit_scoring": {"device": "NPU"}}, "CPU' or 'GPU"),
        ({"run": {"run_id": "../escape"}}, "run_id"),
        ({"plate": {"negative_control_column": 12}}, "different columns"),
    ],
)
def test_bad_config_is_rejected(override, message):
    with pytest.raises(ConfigError, match=message):
        load_config(overrides=override)


def test_int_is_accepted_for_float():
    assert load_config(overrides={"qc": {"zprime_fail": 0}})["qc"]["zprime_fail"] == 0.0


def test_without_faults():
    cfg = without_faults(load_config())
    assert not any(f["enabled"] for f in cfg["faults"].values())
    assert load_config()["faults"]["bad_plate"]["enabled"]
