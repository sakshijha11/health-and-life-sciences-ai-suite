# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The processing pipeline.

Stages run as worker threads connected by bounded queues::

    source -> [buffer] -> assemble + features -> [qc queue]  -> auto-QC + drift -.
                                              \\-> [hit queue] -> hit scoring ----+-> fusion -> results

The unit of work after ingestion is the plate. Auto-QC and hit scoring run on
the same plate in parallel; fusion joins their results by plate id. Results
do not depend on thread timing: plates are processed independently, drift is
updated in plate order by a single thread, and every ranking has a fixed
tie-break.
"""

from __future__ import annotations

import csv
import queue
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Iterable

from screening_tools.evaluation import Evaluator
from screening_tools.features import PlateFeatures, assemble, compute
from screening_tools.fusion import CANDIDATE, CANDIDATE_QC_FLAGGED, NON_HIT, PlateResult, fuse, rank_run
from screening_tools.hit_scoring import HitScorer
from screening_tools.ingestion import END_OF_STREAM, BoundedBuffer, feed
from screening_tools.model_store import ModelStore
from screening_tools.qc import AutoQC
from screening_tools.records import PlateEnd, PlateStart, Sample, Telemetry
from screening_tools.telemetry import Telemetry as RunTelemetry

RESULT_FIELDS = (
    "sample_id",
    "plate_id",
    "well",
    "qc_status",
    "reason_codes",
    "anomaly_score",
    "hit_score",
    "novelty_score",
    "disposition",
    "rank_in_plate",
    "rank_in_run",
)
EVENT_FIELDS = ("reason_code", "severity", "plate_id", "sample_id", "value", "limit", "message")
_STOP = object()


class PipelineError(RuntimeError):
    pass


def _row(r) -> dict:
    d = r.to_dict()
    d["reason_codes"] = ";".join(d["reason_codes"])
    return d


class Pipeline:
    def __init__(self, cfg: dict, source: Iterable, store: ModelStore, out_dir: Path, truth_path: Path | None = None):
        self.cfg = cfg
        self.source = source
        self.out_dir = out_dir
        self.store = store
        self.manifest = store.check(cfg)
        anomaly_model, anomaly_scaler = store.load_anomaly()
        self.hit_scaler = store.load_hit_scaler()
        hs = cfg["hit_scoring"]
        self.scorer = HitScorer(store, hs["device"], hs["batch_size"])
        self.qc = AutoQC(cfg, anomaly_model, anomaly_scaler)
        self.threshold = hs["hit_threshold"]
        self.n_desc = cfg["simulator"]["descriptors"]

        stage_q = cfg["ingestion"]["stage_queue_plates"]
        self.buffer = BoundedBuffer(cfg["ingestion"]["buffer_size"])
        self.qc_q: queue.Queue = queue.Queue(maxsize=stage_q)
        self.hit_q: queue.Queue = queue.Queue(maxsize=stage_q)
        self.join_q: queue.Queue = queue.Queue(maxsize=2 * stage_q)
        self.stop = threading.Event()
        self.errors: list[str] = []
        self.counters = {"samples_in": 0}

        self.telemetry = RunTelemetry(cfg)
        self.telemetry.devices["hit_scoring"] = f"{self.scorer.device} ({self.scorer.device_name}, {self.scorer.precision})"
        self.evaluator = Evaluator(truth_path, cfg["drift"]["lot_window"])

        # State shared with the dashboard (guarded by self.lock).
        self.lock = threading.Lock()
        self.status = "starting"
        self.run_hits: list = []
        self.dispositions: Counter = Counter()
        self.event_counts: Counter = Counter()
        self.recent_events: deque = deque(maxlen=200)
        self.plate_summaries: deque = deque(maxlen=2000)
        self.drift_points: deque = deque(maxlen=2000)
        self.last_plate: dict | None = None
        self.last_result_wall: float | None = None
        self.final_hits: list | None = None
        self.t_wall_start = time.time()
        self.t_wall_end: float | None = None

    # -- worker bodies -------------------------------------------------
    def _guard(self, fn):
        def run():
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - reported to the user and stops the run
                self.errors.append(f"{threading.current_thread().name}: {type(exc).__name__}: {exc}")
                self.stop.set()

        return run

    def _put(self, q: queue.Queue, item) -> bool:
        while not self.stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _get(self, q):
        while not self.stop.is_set():
            try:
                return q.get(timeout=0.1)
            except queue.Empty:
                continue
        return _STOP

    def _assemble(self) -> None:
        ctx: PlateStart | None = None
        samples: list[Sample] = []
        tele: list[Telemetry] = []
        while True:
            rec = self._get(self.buffer)
            if rec is _STOP:
                return
            if rec is END_OF_STREAM:
                break
            if isinstance(rec, PlateStart):
                ctx, samples, tele = rec, [], []
            elif isinstance(rec, Sample):
                samples.append(rec)
            elif isinstance(rec, Telemetry):
                tele.append(rec)
            elif isinstance(rec, PlateEnd):
                if ctx is None or rec.plate_id != ctx.plate_id or rec.n_samples != len(samples):
                    raise PipelineError(f"incomplete plate {rec.plate_id} in the input stream")
                t0 = time.perf_counter()
                plate = assemble(ctx, samples, tele, self.n_desc)
                plate.enqueued_at = rec.enqueued_at
                pf = compute(plate, self.cfg)
                self.telemetry.record("features", time.perf_counter() - t0)
                if not (self._put(self.qc_q, pf) and self._put(self.hit_q, pf)):
                    return
                ctx = None
        self._put(self.qc_q, END_OF_STREAM)
        self._put(self.hit_q, END_OF_STREAM)

    def _auto_qc(self) -> None:
        while True:
            pf = self._get(self.qc_q)
            if pf is _STOP:
                return
            if pf is END_OF_STREAM:
                self._put(self.join_q, ("qc", END_OF_STREAM, None))
                return
            t0 = time.perf_counter()
            result = self.qc.evaluate(pf)
            self.telemetry.record("auto_qc", time.perf_counter() - t0)
            self._put(self.join_q, ("qc", pf, result))

    def _hit_scoring(self) -> None:
        while True:
            pf = self._get(self.hit_q)
            if pf is _STOP:
                return
            if pf is END_OF_STREAM:
                self._put(self.join_q, ("hit", END_OF_STREAM, None))
                return
            t0 = time.perf_counter()
            activity, novelty = self.scorer.score(self.hit_scaler.transform(pf.hit_inputs()))
            self.telemetry.record("hit_scoring", time.perf_counter() - t0)
            self._put(self.join_q, ("hit", pf, (activity, novelty)))

    def _fusion(self) -> None:
        pending: dict[str, dict] = {}
        ended = set()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with (
            (self.out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f_res,
            (self.out_dir / "hits_by_plate.csv").open("w", newline="", encoding="utf-8") as f_hits,
            (self.out_dir / "events.csv").open("w", newline="", encoding="utf-8") as f_ev,
        ):
            w_res = csv.DictWriter(f_res, RESULT_FIELDS)
            w_hits = csv.DictWriter(f_hits, RESULT_FIELDS)
            w_ev = csv.DictWriter(f_ev, EVENT_FIELDS)
            for w in (w_res, w_hits, w_ev):
                w.writeheader()
            while len(ended) < 2:
                msg = self._get(self.join_q)
                if msg is _STOP:
                    return
                kind, pf, payload = msg
                if pf is END_OF_STREAM:
                    ended.add(kind)
                    continue
                entry = pending.setdefault(pf.plate.context.plate_id, {"pf": pf})
                entry[kind] = payload
                if "qc" in entry and "hit" in entry:
                    del pending[pf.plate.context.plate_id]
                    t0 = time.perf_counter()
                    activity, novelty = entry["hit"]
                    result = fuse(pf, entry["qc"], activity, novelty, self.threshold)
                    self.telemetry.record("fusion", time.perf_counter() - t0)
                    w_res.writerows(_row(r) for r in result.records)
                    w_hits.writerows(_row(r) for r in result.hit_list)
                    w_ev.writerows(e.to_dict() for e in entry["qc"].events)
                    self._publish(pf, entry["qc"], result)
                    self.telemetry.plate_done(len(result.records), time.perf_counter() - pf.plate.enqueued_at)
            if pending:
                raise PipelineError(f"{len(pending)} plates never completed both QC and hit scoring")

    def _publish(self, pf: PlateFeatures, qc, result: PlateResult) -> None:
        self.evaluator.add(result, qc.events)
        counts = Counter(r.disposition for r in result.records)
        st = pf.stats
        heat = {
            "rows": pf.plate.context.rows,
            "cols": pf.plate.context.cols,
            "wells": [
                {
                    "well": r.well,
                    "hit_score": r.hit_score,
                    "disposition": r.disposition,
                    "qc_status": r.qc_status,
                }
                for r in result.records
            ],
        }

        def num(v):
            return None if v is None or v != v else round(float(v), 4)

        summary = {
            "plate_id": result.plate_id,
            "plate_index": result.plate_index,
            "reagent_lot": pf.plate.context.reagent_lot,
            "status": result.plate_status,
            "codes": qc.plate_codes,
            "zprime": num(st["zprime"]),
            "signal_to_background": num(st["signal_to_background"]),
            "candidates": counts[CANDIDATE],
            "qc_flagged": counts[CANDIDATE_QC_FLAGGED],
        }
        with self.lock:
            self.run_hits.extend(result.hit_list)
            self.dispositions.update(counts)
            self.event_counts.update(e.reason_code for e in qc.events)
            self.recent_events.extend(e.to_dict() for e in qc.events)
            self.plate_summaries.append(summary)
            self.drift_points.append(qc.drift)
            self.last_plate = {**summary, "heatmap": heat, "hit_list": [r.to_dict() for r in result.hit_list[:20]]}
            self.last_result_wall = time.time()

    # -- control -------------------------------------------------------
    def run(self) -> None:
        workers = [
            threading.Thread(target=self._guard(lambda: feed(self.source, self.buffer, self.cfg["ingestion"]["rate"], self.stop, self.counters)), name="source"),
            threading.Thread(target=self._guard(self._assemble), name="features"),
            threading.Thread(target=self._guard(self._auto_qc), name="auto_qc"),
            threading.Thread(target=self._guard(self._hit_scoring), name="hit_scoring"),
            threading.Thread(target=self._guard(self._fusion), name="fusion"),
        ]
        self.telemetry.start(
            {
                "buffer": self.buffer.depth,
                "qc_queue": self.qc_q.qsize,
                "hit_queue": self.hit_q.qsize,
                "fusion_queue": self.join_q.qsize,
            }
        )
        with self.lock:
            self.status = "running"
        for w in workers:
            w.daemon = True
            w.start()
        try:
            for w in workers:
                while w.is_alive():
                    w.join(timeout=0.2)
        except KeyboardInterrupt:
            self.stop.set()
            self.errors.append("interrupted by user")
            for w in workers:
                w.join(timeout=5)
        self.telemetry.stop()
        with self.lock:
            self.final_hits = rank_run(self.run_hits)
            self.status = "failed" if self.errors else "completed"
            self.t_wall_end = time.time()
        if self.errors:
            raise PipelineError("; ".join(self.errors))

    def request_stop(self) -> None:
        self.stop.set()

    # -- views ---------------------------------------------------------
    def snapshot(self) -> dict:
        tel = self.telemetry
        with self.lock:
            hits = self.final_hits if self.final_hits is not None else sorted(self.run_hits, key=lambda r: (r.disposition != CANDIDATE, -r.hit_score, r.sample_id))
            snap = {
                "status": self.status,
                "run_id": self.cfg["run"]["run_id"],
                "elapsed_s": round((self.t_wall_end or time.time()) - self.t_wall_start, 1),
                "plates_planned": self.cfg["run"]["plates"],
                "plates_done": tel.plates_out,
                "samples_in": self.counters["samples_in"],
                "samples_out": tel.samples_out,
                "dispositions": {k: self.dispositions.get(k, 0) for k in (CANDIDATE, CANDIDATE_QC_FLAGGED, NON_HIT)},
                "event_counts": dict(sorted(self.event_counts.items())),
                "recent_events": list(self.recent_events)[-50:][::-1],
                "plates": list(self.plate_summaries)[-200:],
                "drift": list(self.drift_points)[-200:],
                "last_plate": self.last_plate,
                "run_hit_list": [r.to_dict() for r in hits[:50]],
                "last_result_wall": self.last_result_wall,
            }
        with tel.lock:
            series = list(tel.series)[-240:]
        snap["devices"] = dict(tel.devices)
        snap["backpressure_events"] = self.buffer.backpressure_events
        snap["throughput_sps"] = tel.rolling_throughput()
        steady = tel.steady_throughput()
        snap["steady_throughput_sps"] = round(steady, 1) if steady else None
        snap["latency"] = tel.latency_summary()
        snap["series"] = series
        snap["gpu_note"] = tel.gpu.reason
        snap["evaluation"] = self.evaluator.metrics() if self.evaluator.enabled else None
        snap["errors"] = list(self.errors)
        return snap
