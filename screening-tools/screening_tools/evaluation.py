# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Demo-mode evaluation against ground truth.

The pipeline never sees ground truth. This module reads the simulator's
ground-truth file and compares it with what the pipeline produced: hit-list
precision and recall (tier 1 only, and tiers 1 and 2 together), detection
of each injected fault type, and events raised on plates with no fault.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from screening_tools.fusion import CANDIDATE, CANDIDATE_QC_FLAGGED, PlateResult
from screening_tools.qc import FAIL, REASON_CODES
from screening_tools.simulator import FAULT_BAD_PLATE, FAULT_LOT_CHANGE, FAULT_SLOW_DRIFT, FAULT_TEMPERATURE, FAULT_WELL_ARTIFACT

SAMPLE_QC_CODES = {"QC_SAMPLE_QC_METRIC_HIGH", "QC_SAMPLE_SATURATED", "QC_SAMPLE_ANOMALY"}
PLATE_FAIL_CODES = {c for c, (sev, _) in REASON_CODES.items() if c.startswith("QC_PLATE_") and sev == FAIL}
DRIFT_CODES = {"DRIFT_CONTROL_EWMA", "DRIFT_CONTROL_CUSUM"}


class Evaluator:
    """Keeps a compact per-plate summary (hits and flagged samples only) and
    joins it with ground truth on request."""

    def __init__(self, truth_path: Path | None, lot_window: int):
        self.truth_path = Path(truth_path) if truth_path else None
        self.lot_window = lot_window
        self.truth: dict[str, dict] = {}
        self._offset = 0
        self.plates: dict[str, dict] = {}

    @property
    def enabled(self) -> bool:
        return self.truth_path is not None

    def add(self, result: PlateResult, plate_events: list) -> None:
        self.plates[result.plate_id] = {
            "index": result.plate_index,
            "tier1": {r.sample_id for r in result.hit_list if r.disposition == CANDIDATE},
            "tier2": {r.sample_id for r in result.hit_list if r.disposition == CANDIDATE_QC_FLAGGED},
            "sample_flagged": {e.sample_id for e in plate_events if e.sample_id and e.reason_code in SAMPLE_QC_CODES},
            "codes": Counter(e.reason_code for e in plate_events if e.sample_id is None),
        }

    def _load_truth(self) -> None:
        if self.truth_path is None or not self.truth_path.is_file():
            return
        with self.truth_path.open("r", encoding="utf-8") as fh:
            fh.seek(self._offset)
            while True:
                line = fh.readline()
                if not line or not line.endswith("\n"):
                    break
                self._offset = fh.tell()
                t = json.loads(line)
                self.truth[t["plate_id"]] = t

    def metrics(self) -> dict | None:
        if not self.enabled:
            return None
        self._load_truth()
        done = [pid for pid in self.plates if pid in self.truth]
        if not done:
            return {"plates_evaluated": 0}
        tp1 = tp12 = n1 = n12 = actives = 0
        artifacts = artifacts_flagged = artifacts_in_tier1 = 0
        fault_plates: dict[str, list[int]] = defaultdict(list)
        unlabelled_events: Counter = Counter()
        unlabelled_plates = 0
        index_of = {}
        for pid in done:
            p, t = self.plates[pid], self.truth[pid]
            index_of[pid] = p["index"]
            act = {s["sample_id"] for s in t["samples"] if s["is_active"]}
            arts = {s["sample_id"] for s in t["samples"] if s["fault"] == FAULT_WELL_ARTIFACT}
            actives += len(act)
            tp1 += len(p["tier1"] & act)
            tp12 += len((p["tier1"] | p["tier2"]) & act)
            n1 += len(p["tier1"])
            n12 += len(p["tier1"]) + len(p["tier2"])
            artifacts += len(arts)
            artifacts_flagged += len(arts & p["sample_flagged"])
            artifacts_in_tier1 += len(arts & p["tier1"])
            for f in t["faults"]:
                fault_plates[f].append(p["index"])

        # A plate counts as fault-free if neither it nor the previous
        # `lot_window` plates had a fault: detectors may legitimately fire a few
        # plates after the event (lot shift, end of a drift).
        faulty_idx = {i for v in fault_plates.values() for i in v}
        for pid in done:
            i = self.plates[pid]["index"]
            if not any(j in faulty_idx for j in range(i - self.lot_window, i + 1)):
                unlabelled_plates += 1
                unlabelled_events.update(self.plates[pid]["codes"])

        by_index = {self.plates[pid]["index"]: self.plates[pid] for pid in done}

        def codes_at(i: int) -> Counter:
            return by_index[i]["codes"] if i in by_index else Counter()

        faults = {}
        for name in (FAULT_BAD_PLATE, FAULT_TEMPERATURE):
            plates = fault_plates.get(name, [])
            if plates:
                want = PLATE_FAIL_CODES if name == FAULT_BAD_PLATE else {"TELEMETRY_TEMP_OUT_OF_BAND"}
                hit = sum(1 for i in plates if want & set(codes_at(i)))
                faults[name] = {"plates": len(plates), "detected": hit, "rate": round(hit / len(plates), 4)}
        for i in fault_plates.get(FAULT_LOT_CHANGE, []):
            found = [j for j in range(i, i + self.lot_window + 1) if "DRIFT_REAGENT_LOT_SHIFT" in codes_at(j)]
            faults[FAULT_LOT_CHANGE] = {"at_plate": i, "detected": bool(found), "detected_at_plate": found[0] if found else None}
        drift_plates = fault_plates.get(FAULT_SLOW_DRIFT, [])
        if drift_plates:
            start = min(drift_plates)
            found = [i for i in sorted(by_index) if i >= start and DRIFT_CODES & set(codes_at(i))]
            faults[FAULT_SLOW_DRIFT] = {
                "start_plate": start,
                "detected": bool(found),
                "detection_delay_plates": (found[0] - start) if found else None,
            }
        if artifacts:
            faults[FAULT_WELL_ARTIFACT] = {
                "wells": artifacts,
                "flagged_by_sample_qc": artifacts_flagged,
                "rate": round(artifacts_flagged / artifacts, 4),
                "left_in_tier1": artifacts_in_tier1,
            }

        def ratio(a, b):
            return round(a / b, 4) if b else None

        return {
            "plates_evaluated": len(done),
            "actives": actives,
            "tier1": {"listed": n1, "true_hits": tp1, "precision": ratio(tp1, n1), "recall": ratio(tp1, actives)},
            "tier1_and_2": {"listed": n12, "true_hits": tp12, "precision": ratio(tp12, n12), "recall": ratio(tp12, actives)},
            "faults": faults,
            "plates_without_fault": unlabelled_plates,
            "events_on_plates_without_fault": dict(sorted(unlabelled_events.items())),
        }
