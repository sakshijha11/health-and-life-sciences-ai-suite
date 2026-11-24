# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rule-based drift monitoring across plates (part of ITEP-97834).

Tracks the negative- and positive-control levels of each plate, relative to
the plate's calibration context, against statistical limits learned from the
first in-control plates. No training and no model: each rule that fires is
its own explanation.

* EWMA: exponentially weighted moving average outside its control limits.
* CUSUM: cumulative sum of standardised deviations above a decision limit.
* Reagent lot: assay window before vs after a lot change.

Plates that failed plate QC are not used, so one bad plate does not look like
drift. After a reagent lot change the limits are re-learned.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from screening_tools.records import QCEvent

_SIGMA_FLOOR = 0.005


class _Series:
    def __init__(self, name: str, d: dict):
        self.name = name
        self.lam = d["ewma_lambda"]
        self.L = d["ewma_limit_sigma"]
        self.k = d["cusum_k_sigma"]
        self.h = d["cusum_h_sigma"]
        self.n_base = d["baseline_plates"]
        self.rebaseline()

    def rebaseline(self) -> None:
        self.baseline: list[float] = []
        self.mu = self.sigma = None
        self.ewma = None
        self.s_hi = self.s_lo = 0.0
        self.ewma_alarm = False
        self.cusum_alarm: set[str] = set()

    def limits(self) -> tuple[float, float] | None:
        if self.mu is None:
            return None
        half = self.L * self.sigma * math.sqrt(self.lam / (2.0 - self.lam))
        return self.mu - half, self.mu + half

    def update(self, x: float) -> list[tuple[str, float, float, str]]:
        if self.mu is None:
            self.baseline.append(x)
            if len(self.baseline) >= self.n_base:
                self.mu = float(np.mean(self.baseline))
                self.sigma = max(float(np.std(self.baseline, ddof=1)), _SIGMA_FLOOR)
                self.ewma = self.mu
            return []
        out = []
        self.ewma = self.lam * x + (1.0 - self.lam) * self.ewma
        lo, hi = self.limits()
        outside = not lo <= self.ewma <= hi
        if outside and not self.ewma_alarm:
            limit = hi if self.ewma > hi else lo
            out.append(("DRIFT_CONTROL_EWMA", self.ewma, limit, f"{self.name} control EWMA left its limits"))
        self.ewma_alarm = outside
        z = (x - self.mu) / self.sigma
        self.s_hi = max(0.0, self.s_hi + z - self.k)
        self.s_lo = max(0.0, self.s_lo - z - self.k)
        # One alarm per excursion: a direction re-arms once its sum is back to zero.
        for s, direction in ((self.s_hi, "upward"), (self.s_lo, "downward")):
            if s > self.h and direction not in self.cusum_alarm:
                out.append(("DRIFT_CONTROL_CUSUM", s, self.h, f"{self.name} control {direction} CUSUM"))
                self.cusum_alarm.add(direction)
            elif s == 0.0:
                self.cusum_alarm.discard(direction)
        return out


class DriftMonitor:
    def __init__(self, cfg: dict):
        d = cfg["drift"]
        self.d = d
        self.series = {"negative": _Series("negative", d), "positive": _Series("positive", d)}
        self.window_history: deque[float] = deque(maxlen=d["lot_window"])
        self.last_lot: str | None = None
        self.lot_change: dict | None = None

    def update(self, pf, plate_status: str) -> tuple[list[QCEvent], dict]:
        from screening_tools.qc import make_event  # local import avoids a cycle

        ctx, st = pf.plate.context, pf.stats
        pid = ctx.plate_id
        events: list[QCEvent] = []
        neg_ratio = st["negative_mean"] / ctx.expected_negative
        pos_ratio = st["positive_mean"] / ctx.expected_positive
        window_ratio = (st["positive_mean"] - st["negative_mean"]) / (ctx.expected_positive - ctx.expected_negative)
        used = plate_status != "fail" and bool(np.isfinite(neg_ratio) and np.isfinite(pos_ratio))

        if self.last_lot is not None and ctx.reagent_lot != self.last_lot:
            before = float(np.mean(self.window_history)) if self.window_history else None
            self.lot_change = {"plate_id": pid, "before": before, "after": [], "from": self.last_lot, "to": ctx.reagent_lot}
            for s in self.series.values():
                s.rebaseline()
        self.last_lot = ctx.reagent_lot

        if used:
            for name, value in (("negative", neg_ratio), ("positive", pos_ratio)):
                for code, v, lim, msg in self.series[name].update(float(value)):
                    events.append(make_event(code, pid, None, v, lim, msg))
            self.window_history.append(float(window_ratio))
            lc = self.lot_change
            if lc is not None and lc["before"] is not None:
                lc["after"].append(float(window_ratio))
                if len(lc["after"]) >= self.d["lot_window"]:
                    change = float(np.mean(lc["after"])) / lc["before"] - 1.0
                    if abs(change) > self.d["lot_shift_warn"]:
                        events.append(
                            make_event(
                                "DRIFT_REAGENT_LOT_SHIFT",
                                pid,
                                None,
                                change,
                                self.d["lot_shift_warn"] * np.sign(change),
                                f"assay window change after lot {lc['from']} -> {lc['to']} (plate {lc['plate_id']})",
                            )
                        )
                    self.lot_change = None

        def _pt(s: _Series):
            lim = s.limits()
            return {"ewma": s.ewma, "low": lim[0] if lim else None, "high": lim[1] if lim else None}

        point = {
            "plate_id": pid,
            "plate_index": ctx.plate_index,
            "reagent_lot": ctx.reagent_lot,
            "negative_ratio": float(neg_ratio) if np.isfinite(neg_ratio) else None,
            "positive_ratio": float(pos_ratio) if np.isfinite(pos_ratio) else None,
            "negative": _pt(self.series["negative"]),
            "positive": _pt(self.series["positive"]),
            "used": used,
        }
        return events, point
