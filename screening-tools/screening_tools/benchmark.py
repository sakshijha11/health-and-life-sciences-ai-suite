# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hit-scoring benchmark: CPU against iGPU (ITEP-97836).

The same OpenVINO model runs on each device over a sweep of batch sizes. For
each combination it reports the P50/P95 latency of one inference request and
the throughput in samples per second. It then scores the same plates on every
device and checks that the per-plate top-N ranking is the same.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from screening_tools.export import hardware_info, software_info
from screening_tools.features import assemble, compute
from screening_tools.hit_scoring import DeviceUnavailable, HitScorer
from screening_tools.model_store import ModelStore
from screening_tools.simulator import STREAM_RUN, Simulator
from screening_tools.telemetry import percentiles

BENCH_STREAM = STREAM_RUN + 10


def _plates(cfg: dict, store: ModelStore, n_plates: int) -> list[np.ndarray]:
    scaler = store.load_hit_scaler()
    n_desc = cfg["simulator"]["descriptors"]
    return [
        scaler.transform(compute(assemble(p.context, p.samples, p.telemetry, n_desc), cfg).hit_inputs())
        for p in Simulator(cfg, stream=BENCH_STREAM, plates=n_plates)
    ]


def run_benchmark(cfg: dict, store: ModelStore, devices: list[str], batch_sizes: list[int], seconds: float, n_plates: int, top_n: int, log=print) -> dict:
    store.check(cfg)
    plates = _plates(cfg, store, n_plates)
    pool = np.vstack(plates)
    rows, scores, skipped = [], {}, {}
    for device in devices:
        for bs in batch_sizes:
            try:
                scorer = HitScorer(store, device, bs)
            except DeviceUnavailable as exc:
                skipped[device] = str(exc)
                break
            reps = int(np.ceil(bs / pool.shape[0]))
            batch = np.tile(pool, (reps, 1))[:bs]
            for _ in range(5):
                scorer.infer_batch(batch)
            lat = []
            t_end = time.perf_counter() + seconds
            while time.perf_counter() < t_end or len(lat) < 20:
                t0 = time.perf_counter()
                scorer.infer_batch(batch)
                lat.append(time.perf_counter() - t0)
            p = percentiles(lat)
            sps = bs * len(lat) / sum(lat)
            row = {"device": device, "device_name": scorer.device_name, "precision": scorer.precision, "batch_size": bs, **p, "samples_per_s": round(sps, 1)}
            rows.append(row)
            log(f"  {device:4s} batch {bs:5d}: P50 {p['p50_ms']:8.3f} ms  P95 {p['p95_ms']:8.3f} ms  {sps:12,.0f} samples/s")
        if device not in skipped:
            scorer = HitScorer(store, device, cfg["hit_scoring"]["batch_size"])
            scores[device] = [scorer.score(x)[0] for x in plates]

    agreement = None
    if len(scores) >= 2:
        names = list(scores)
        ref = names[0]
        agreement = {}
        for other in names[1:]:
            overlaps, max_diff = [], 0.0
            for a, b in zip(scores[ref], scores[other]):
                n = min(top_n, a.size)
                ta = set(np.lexsort((np.arange(a.size), -a))[:n])
                tb = set(np.lexsort((np.arange(b.size), -b))[:n])
                overlaps.append(len(ta & tb) / n if n else 1.0)
                max_diff = max(max_diff, float(np.max(np.abs(a - b))))
            agreement[f"{ref}_vs_{other}"] = {
                "top_n": top_n,
                "plates": len(overlaps),
                "mean_top_n_overlap": round(float(np.mean(overlaps)), 4),
                "min_top_n_overlap": round(float(np.min(overlaps)), 4),
                "max_abs_score_difference": round(max_diff, 6),
            }
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": hardware_info(),
        "software": software_info(),
        "model": {k: store.manifest()["hit_model"][k] for k in ("hidden_layers", "parameters")},
        "seconds_per_point": seconds,
        "results": rows,
        "skipped_devices": skipped,
        "ranking_agreement": agreement,
    }


def write_benchmark(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"benchmark-{stamp}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path
