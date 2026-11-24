# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run summary export.

``summary.json`` alone is enough to compare two machines: configuration and
seed, hardware and software versions, every metric, QC event counts by
reason code and the final hit list.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

from screening_tools import __version__
from screening_tools.pipeline import RESULT_FIELDS, Pipeline, _row


def hardware_info() -> dict:
    import openvino as ov

    core = ov.Core()
    devices = {}
    for d in core.available_devices:
        try:
            devices[d] = str(core.get_property(d, "FULL_DEVICE_NAME"))
        except Exception:  # noqa: BLE001
            devices[d] = d
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "openvino_devices": devices,
    }


def software_info() -> dict:
    import openvino as ov
    import sklearn

    return {
        "app": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "openvino": ov.get_version(),
    }


def write_summary(pipe: Pipeline, wall_seconds: float, mode: str) -> Path:
    tel = pipe.telemetry
    snap = pipe.snapshot()
    hits = pipe.final_hits or []
    with (pipe.out_dir / "hits_run.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, RESULT_FIELDS)
        w.writeheader()
        w.writerows(_row(r) for r in hits)
    steady = tel.steady_throughput()
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": pipe.cfg["run"]["run_id"],
        "status": snap["status"],
        "mode": mode,
        "seed": pipe.cfg["run"]["seed"],
        "config": pipe.cfg,
        "hardware": hardware_info(),
        "software": software_info(),
        "models": {
            "created_utc": pipe.manifest.get("created_utc"),
            "hit_model": {k: pipe.manifest["hit_model"][k] for k in ("hidden_layers", "parameters", "test_metrics")},
            "anomaly_model": pipe.manifest["anomaly_model"]["test_metrics"],
        },
        "devices": snap["devices"],
        "metrics": {
            "plates": tel.plates_out,
            "samples": tel.samples_out,
            "wall_seconds": round(wall_seconds, 3),
            "throughput_samples_per_s": round(steady, 1) if steady else None,
            "throughput_samples_per_hour": round(steady * 3600) if steady else None,
            "warmup_plates_excluded": tel.warmup,
            "latency": tel.latency_summary(),
            "backpressure_events": pipe.buffer.backpressure_events,
            "buffer_peak_depth": pipe.buffer.max_depth,
            "buffer_size": pipe.buffer.maxsize,
            "utilisation": tel.utilisation_summary(),
            "ui_refresh_lag": tel.ui_lag_summary(),
        },
        "dispositions": snap["dispositions"],
        "qc_event_counts": snap["event_counts"],
        "evaluation": snap["evaluation"],
        "hit_list": [r.to_dict() for r in hits],
        "files": ["summary.json", "hits_run.csv", "hits_by_plate.csv", "results.csv", "events.csv"],
    }
    path = pipe.out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path
