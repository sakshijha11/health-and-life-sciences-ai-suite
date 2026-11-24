# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observability telemetry (part of ITEP-97838).

Collected inside the pipeline, independent of the dashboard: per-stage and
end-to-end latency (P50/P95), throughput, queue depth, backpressure, and
sampled CPU and iGPU utilisation.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from screening_tools.utilization import CpuReader, make_gpu_reader

STAGES = ("features", "auto_qc", "hit_scoring", "fusion", "end_to_end")


def percentiles(values) -> dict:
    if not values:
        return {"p50_ms": None, "p95_ms": None, "count": 0}
    arr = np.asarray(values, dtype=np.float64) * 1000.0
    return {"p50_ms": round(float(np.percentile(arr, 50)), 3), "p95_ms": round(float(np.percentile(arr, 95)), 3), "count": int(arr.size)}


class Telemetry:
    def __init__(self, cfg: dict):
        self.warmup = cfg["telemetry"]["warmup_plates"]
        self.interval = cfg["telemetry"]["sample_interval_s"]
        self.lock = threading.Lock()
        self.latency: dict[str, list[float]] = {s: [] for s in STAGES}
        self._seen: dict[str, int] = {}
        self.devices: dict[str, str] = {"features": "CPU", "auto_qc": "CPU", "fusion": "CPU"}
        self.samples_out = 0
        self.plates_out = 0
        self.t_start: float | None = None
        self.t_steady: float | None = None
        self.steady_samples = 0
        self.t_last: float | None = None
        self.out_times: deque[tuple[float, int]] = deque()
        self.series: deque[dict] = deque(maxlen=1200)
        self.ui_lag: deque[float] = deque(maxlen=1000)
        self.queue_probes: dict = {}
        self.cpu = CpuReader()
        self.gpu = make_gpu_reader()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- recording -----------------------------------------------------
    def start(self, queue_probes: dict) -> None:
        self.queue_probes = queue_probes
        self.t_start = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_loop, name="telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.gpu.close()

    def record(self, stage: str, seconds: float) -> None:
        """Record one plate's latency for a stage; the first warm-up plates are skipped."""
        with self.lock:
            self._seen[stage] = self._seen.get(stage, 0) + 1
            if self._seen[stage] > self.warmup:
                self.latency[stage].append(seconds)

    def plate_done(self, n_samples: int, e2e_seconds: float) -> None:
        now = time.perf_counter()
        with self.lock:
            self.plates_out += 1
            self.samples_out += n_samples
            self.t_last = now
            if self.plates_out == self.warmup:
                self.t_steady = now
            elif self.plates_out > self.warmup:
                self.steady_samples += n_samples
                self.latency["end_to_end"].append(e2e_seconds)
            self.out_times.append((now, n_samples))
            while self.out_times and now - self.out_times[0][0] > 5.0:
                self.out_times.popleft()

    def _sample_loop(self) -> None:
        self.cpu.prime()
        while not self._stop.wait(self.interval):
            point = {"t": round(time.perf_counter() - (self.t_start or 0.0), 2)}
            point.update(self.cpu.read())
            gpu = self.gpu.read() if self.gpu.available else None
            point["gpu_percent"] = gpu["gpu_percent"] if gpu else None
            point["process_gpu_percent"] = gpu["process_gpu_percent"] if gpu else None
            point["gpu_by_engine"] = gpu["gpu_by_engine"] if gpu else None
            point["queues"] = {name: probe() for name, probe in self.queue_probes.items()}
            point["throughput_sps"] = self.rolling_throughput()
            with self.lock:
                self.series.append(point)

    # -- reading -------------------------------------------------------
    def rolling_throughput(self) -> float:
        with self.lock:
            if len(self.out_times) < 2:
                return 0.0
            span = self.out_times[-1][0] - self.out_times[0][0]
            n = sum(c for _, c in list(self.out_times)[1:])
        return round(n / span, 1) if span > 0 else 0.0

    def steady_throughput(self) -> float | None:
        with self.lock:
            if self.t_steady is None or self.t_last is None or self.t_last <= self.t_steady:
                return None
            return self.steady_samples / (self.t_last - self.t_steady)

    def utilisation_summary(self) -> dict:
        with self.lock:
            pts = list(self.series)

        def agg(key):
            vals = [p[key] for p in pts if p.get(key) is not None]
            return {"avg": round(float(np.mean(vals)), 2), "peak": round(float(np.max(vals)), 2)} if vals else None

        out = {k: agg(k) for k in ("cpu_percent", "process_cpu_percent", "gpu_percent", "process_gpu_percent")}
        out["gpu_note"] = self.gpu.reason or "busiest GPU engine, system-wide (Windows GPU Engine counters)"
        depths = {}
        for name in self.queue_probes:
            vals = [p["queues"][name] for p in pts if "queues" in p]
            depths[name] = {"avg": round(float(np.mean(vals)), 2), "peak": int(max(vals))} if vals else None
        out["queue_depth"] = depths
        return out

    def latency_summary(self) -> dict:
        with self.lock:
            return {s: percentiles(v) for s, v in self.latency.items()}

    def record_ui_lag(self, seconds: float) -> None:
        if 0 <= seconds < 3600:
            with self.lock:
                self.ui_lag.append(seconds)

    def ui_lag_summary(self) -> dict:
        with self.lock:
            return percentiles(list(self.ui_lag))
