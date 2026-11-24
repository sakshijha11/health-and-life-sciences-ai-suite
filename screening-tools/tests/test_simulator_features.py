# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from screening_tools.features import assemble, compute
from screening_tools.records import WELL_NEGATIVE, WELL_POSITIVE, PlateEnd, PlateStart, Sample, from_json_dict, to_json_dict
from screening_tools.simulator import FAULT_BAD_PLATE, STREAM_RUN, STREAM_TRAINING, Simulator


def _records(cfg, stream, n=3):
    return [to_json_dict(r) for p in Simulator(cfg, stream=stream, plates=n) for r in p.records()]


def test_same_seed_same_stream_is_identical(cfg):
    assert _records(cfg, STREAM_RUN) == _records(cfg, STREAM_RUN)


def test_training_and_run_streams_differ(cfg):
    assert _records(cfg, STREAM_RUN) != _records(cfg, STREAM_TRAINING)


def test_plate_layout_and_record_order(cfg):
    plate = Simulator(cfg, stream=STREAM_RUN, plates=1).plate(0)
    recs = list(plate.records())
    assert isinstance(recs[0], PlateStart) and isinstance(recs[-1], PlateEnd)
    samples = [r for r in recs if isinstance(r, Sample)]
    assert len(samples) == 96 == recs[-1].n_samples
    assert {s.well_type for s in samples if s.well.endswith("01")} == {WELL_NEGATIVE}
    assert {s.well_type for s in samples if s.well.endswith("12")} == {WELL_POSITIVE}
    assert "is_active" not in to_json_dict(samples[0])  # no ground truth in the stream


def test_json_round_trip(cfg):
    for r in Simulator(cfg, stream=STREAM_RUN, plates=1).plate(0).records():
        assert from_json_dict(to_json_dict(r)) == r


def test_controls_land_on_expected_percent_of_control(cfg):
    plate = Simulator(cfg, stream=STREAM_RUN, plates=1).plate(0)
    pf = compute(assemble(plate.context, plate.samples, plate.telemetry, cfg["simulator"]["descriptors"]), cfg)
    assert pf.stats["controls_valid"]
    assert pf.stats["zprime"] > cfg["qc"]["zprime_warn"]
    actives = np.array([t["is_active"] for t in plate.truth["samples"]])
    poc = pf.column("percent_of_control")
    assert abs(np.median(poc[~actives])) < 10
    if actives.any():
        assert poc[actives].mean() > 25


def test_bad_plate_is_labelled_in_truth(cfg):
    idx = cfg["faults"]["bad_plate"]["plates"][0]
    assert FAULT_BAD_PLATE in Simulator(cfg, stream=STREAM_RUN).plate(idx).truth["faults"]


def test_features_do_not_depend_on_other_plates(cfg):
    sim = Simulator(cfg, stream=STREAM_RUN, plates=4)
    n = cfg["simulator"]["descriptors"]
    together = [compute(assemble(p.context, p.samples, p.telemetry, n), cfg) for p in sim]
    alone = Simulator(cfg, stream=STREAM_RUN, plates=4).plate(2)
    single = compute(assemble(alone.context, alone.samples, alone.telemetry, n), cfg)
    np.testing.assert_array_equal(together[2].hit_inputs(), single.hit_inputs())
    np.testing.assert_array_equal(together[2].anomaly_inputs(), single.anomaly_inputs())
