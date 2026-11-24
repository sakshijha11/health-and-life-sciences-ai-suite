# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json
import urllib.error
import urllib.request

import pytest

from screening_tools.dashboard.server import DashboardServer
from screening_tools.export import write_summary
from screening_tools.ingestion import ReplaySource, SimulatorSource, write_run_file
from screening_tools.pipeline import Pipeline


def _run(cfg, store, out, source=None, truth=None):
    truth = truth if source is not None else out / "ground_truth.jsonl"
    pipe = Pipeline(cfg, source or SimulatorSource(cfg, truth), store, out, truth)
    pipe.run()
    write_summary(pipe, 1.0, "test")
    return pipe


def test_same_seed_gives_identical_outputs(cfg, store, tmp_path):
    _run(cfg, store, tmp_path / "a")
    small = copy.deepcopy(cfg)
    small["ingestion"]["buffer_size"] = 50  # different timing and backpressure, same results
    small["ingestion"]["stage_queue_plates"] = 1
    _run(small, store, tmp_path / "b")
    for name in ("results.csv", "hits_by_plate.csv", "events.csv", "hits_run.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name


def test_replay_matches_live(cfg, store, tmp_path):
    live = _run(cfg, store, tmp_path / "live")
    rec = tmp_path / "rec.jsonl"
    write_run_file(cfg, rec, tmp_path / "rec.truth.jsonl")
    _run(cfg, store, tmp_path / "replay", ReplaySource(rec), tmp_path / "rec.truth.jsonl")
    assert (tmp_path / "live" / "results.csv").read_bytes() == (tmp_path / "replay" / "results.csv").read_bytes()
    assert live.telemetry.plates_out == cfg["run"]["plates"]


def test_summary_contents(cfg, store, tmp_path):
    pipe = _run(cfg, store, tmp_path / "s")
    s = json.loads((tmp_path / "s" / "summary.json").read_text(encoding="utf-8"))
    assert s["status"] == "completed"
    assert s["metrics"]["samples"] == cfg["run"]["plates"] * 80
    assert s["seed"] == cfg["run"]["seed"]
    assert "openvino" in s["software"] and "openvino_devices" in s["hardware"]
    assert s["evaluation"]["plates_evaluated"] == cfg["run"]["plates"]
    assert s["evaluation"]["faults"]["bad_plate"]["rate"] == 1.0
    assert len(s["hit_list"]) == len(pipe.final_hits)


def test_dashboard_endpoints(cfg, store, tmp_path):
    out = tmp_path / "d"
    pipe = Pipeline(cfg, SimulatorSource(cfg, out / "t.jsonl"), store, out, out / "t.jsonl")
    server = DashboardServer(pipe, "127.0.0.1", 0, 1000)
    server.start()
    try:
        pipe.run()
        base = server.url
        snap = json.load(urllib.request.urlopen(base + "api/snapshot"))
        assert snap["status"] == "completed" and snap["plates_done"] == cfg["run"]["plates"]
        page = urllib.request.urlopen(base)
        assert page.headers["Content-Security-Policy"].startswith("default-src 'self'")
        for bad in ("../server.py", "static/app.js", "%2e%2e/server.py"):
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(base + bad)
            assert err.value.code == 404
        req = urllib.request.Request(base + "api/ui-lag", data=b'{"lag_ms": 12.5}', headers={"Content-Type": "application/json"})
        assert json.load(urllib.request.urlopen(req))["ok"]
        assert pipe.telemetry.ui_lag_summary()["count"] == 1
        req = urllib.request.Request(base + "api/ui-lag", data=b"x" * 5000, headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req)
        assert err.value.code == 400
        req = urllib.request.Request(base + "api/snapshot", headers={"Host": "evil.example"})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req)
        assert err.value.code == 403
    finally:
        server.stop()
