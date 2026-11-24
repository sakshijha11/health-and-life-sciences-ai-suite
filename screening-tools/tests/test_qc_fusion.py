# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import numpy as np
import pytest

from screening_tools.config import without_faults
from screening_tools.features import assemble, compute
from screening_tools.fusion import CANDIDATE, CANDIDATE_QC_FLAGGED, FLAG_NOTE, NON_HIT, fuse, rank_run
from screening_tools.qc import FAIL, PASS, REASON_CODES, AutoQC
from screening_tools.simulator import STREAM_RUN, Simulator


def _plates(cfg, n=None, stream=STREAM_RUN):
    d = cfg["simulator"]["descriptors"]
    for p in Simulator(cfg, stream=stream, plates=n):
        yield compute(assemble(p.context, p.samples, p.telemetry, d), cfg)


def _qc_run(cfg, store, n=None):
    qc = AutoQC(cfg, *store.load_anomaly())
    return [(pf, qc.evaluate(pf)) for pf in _plates(cfg, n)]


def test_every_event_uses_a_documented_code(cfg, store):
    for _, q in _qc_run(cfg, store, 50):
        for e in q.events:
            assert e.reason_code in REASON_CODES
            assert e.severity == REASON_CODES[e.reason_code][0]


def test_bad_plate_fails_plate_qc(cfg, store):
    bad = set(cfg["faults"]["bad_plate"]["plates"])
    for pf, q in _qc_run(cfg, store, max(bad) + 1):
        if pf.plate.context.plate_index in bad:
            assert q.plate_status == FAIL
            assert any(c.startswith("QC_PLATE_") for c in q.plate_codes)


def test_temperature_excursion_raises_telemetry_code(cfg, store):
    f = cfg["faults"]["temperature_excursion"]
    window = range(f["start_plate"], f["start_plate"] + f["plates"])
    for pf, q in _qc_run(cfg, store, window.stop):
        if pf.plate.context.plate_index in window:
            assert "TELEMETRY_TEMP_OUT_OF_BAND" in q.plate_codes


def test_clean_plates_pass(cfg, store):
    clean = without_faults(cfg)
    statuses = [q.plate_status for _, q in _qc_run(clean, store, 20)]
    assert statuses.count(FAIL) == 0
    assert statuses.count(PASS) >= 18


def test_lot_change_is_detected(cfg, store):
    c = copy.deepcopy(cfg)
    at = c["faults"]["reagent_lot_change"]["at_plate"]
    c["faults"]["reagent_lot_change"]["positive_factor"] = 0.7
    codes = [(pf.plate.context.plate_index, set(q.plate_codes) | {e.reason_code for e in q.events}) for pf, q in _qc_run(c, store, at + 8)]
    hits = [i for i, cs in codes if "DRIFT_REAGENT_LOT_SHIFT" in cs]
    assert hits and at <= hits[0] <= at + c["drift"]["lot_window"]


def test_fusion_tiers_and_order(cfg, store):
    pf, q = _qc_run(cfg, store, 1)[0]
    n = len(pf.sample_index)
    rng = np.random.default_rng(0)
    activity = rng.uniform(0, 1, n).astype(np.float32)
    activity[:5] = 0.99  # ties: broken by sample id
    novelty = rng.uniform(0, 1, n).astype(np.float32)
    q.sample_status[0] = FAIL
    result = fuse(pf, q, activity, novelty, 0.5)
    tiers = [r.disposition for r in result.hit_list]
    assert tiers == sorted(tiers, key=lambda t: t != CANDIDATE)  # no flagged sample above an unflagged one
    assert result.records[0].disposition == CANDIDATE_QC_FLAGGED
    assert FLAG_NOTE
    assert all(r.disposition == NON_HIT for r in result.records if r.hit_score < 0.5)
    assert [r.rank_in_plate for r in result.hit_list] == list(range(1, len(result.hit_list) + 1))
    t1 = [r for r in result.hit_list if r.disposition == CANDIDATE]
    for a, b in zip(t1, t1[1:]):
        assert (a.hit_score, b.sample_id) >= (b.hit_score, a.sample_id)
    run = rank_run(result.hit_list[::-1])
    assert [r.sample_id for r in run] == [r.sample_id for r in result.hit_list]


def test_fusion_rejects_mismatched_plates(cfg, store):
    (pf0, q0), (pf1, _) = _qc_run(cfg, store, 2)
    n = len(pf1.sample_index)
    with pytest.raises(ValueError):
        fuse(pf1, q0, np.zeros(n, np.float32), np.zeros(n, np.float32), 0.5)
