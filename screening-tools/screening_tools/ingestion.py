# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""High-throughput ingestion (ITEP-97831).

Two interchangeable sources feed the same bounded buffer:

* ``SimulatorSource`` generates plates live from the seeded simulator.
* ``ReplaySource`` reads a run file written earlier by ``write_run_file``.

When the buffer is full the source waits; nothing is dropped. Each wait is
counted as a backpressure event.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Iterable, Iterator

from screening_tools.records import PlateEnd, Sample, from_json_dict, to_json_dict
from screening_tools.simulator import STREAM_RUN, Simulator

END_OF_STREAM = object()


class SimulatorSource:
    """Live simulated instrument. Writes ground truth to ``truth_path`` as it goes."""

    def __init__(self, cfg: dict, truth_path: Path | None = None):
        self.cfg = cfg
        self.truth_path = truth_path

    def __iter__(self) -> Iterator:
        truth_fh = None
        if self.truth_path is not None:
            self.truth_path.parent.mkdir(parents=True, exist_ok=True)
            truth_fh = self.truth_path.open("w", encoding="utf-8")
        try:
            for plate in Simulator(self.cfg, stream=STREAM_RUN):
                if truth_fh is not None:
                    truth_fh.write(json.dumps(plate.truth) + "\n")
                    truth_fh.flush()
                yield from plate.records()
        finally:
            if truth_fh is not None:
                truth_fh.close()


class ReplaySource:
    """Replays a run file (JSON Lines, one record per line)."""

    def __init__(self, run_path: Path):
        self.run_path = Path(run_path)

    def __iter__(self) -> Iterator:
        with self.run_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield from_json_dict(json.loads(line))


def write_run_file(cfg: dict, run_path: Path, truth_path: Path) -> int:
    """Write a simulated run to ``run_path`` and its ground truth to ``truth_path``."""
    run_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with run_path.open("w", encoding="utf-8") as fh:
        for record in SimulatorSource(cfg, truth_path):
            fh.write(json.dumps(to_json_dict(record)) + "\n")
            count += 1
    return count


class BoundedBuffer:
    """A bounded FIFO that blocks the producer when full and counts the waits."""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.backpressure_events = 0
        self.max_depth = 0

    def put(self, item, stop: threading.Event | None = None) -> bool:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.backpressure_events += 1
            while True:
                if stop is not None and stop.is_set():
                    return False
                try:
                    self._q.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
        depth = self._q.qsize()
        if depth > self.max_depth:
            self.max_depth = depth
        return True

    def get(self, timeout: float | None = None):
        return self._q.get(timeout=timeout)

    def depth(self) -> int:
        return self._q.qsize()


class Pacer:
    """Holds the source to a fixed sample rate. ``rate <= 0`` means as fast as possible."""

    def __init__(self, rate: float):
        self.rate = rate
        self.count = 0
        self.t0: float | None = None

    def tick(self) -> None:
        if self.rate <= 0:
            return
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now
        self.count += 1
        target = self.t0 + self.count / self.rate
        if target > now:
            time.sleep(target - now)


def feed(source: Iterable, buffer: BoundedBuffer, rate: float, stop: threading.Event, counters: dict) -> None:
    """Thread body: move records from ``source`` into ``buffer`` at the configured rate."""
    pacer = Pacer(rate)
    try:
        for record in source:
            if stop.is_set():
                break
            if isinstance(record, Sample):
                pacer.tick()
                counters["samples_in"] += 1
            elif isinstance(record, PlateEnd):
                record.enqueued_at = time.perf_counter()
            if not buffer.put(record, stop):
                break
    finally:
        buffer.put(END_OF_STREAM, stop)
