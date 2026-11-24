# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU auto-QC and sample anomaly detection (ITEP-97834).

Three parts, all on the CPU:

1. Plate QC rules computed from the control wells when a plate completes.
2. Sample checks: QC-metric and saturation rules plus an Isolation Forest
   anomaly score with a configurable threshold.
3. Rule-based drift monitoring across plates (see ``drift.py``).

Every finding is a ``QCEvent`` carrying a reason code from ``REASON_CODES``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from screening_tools.drift import DriftMonitor
from screening_tools.features import ANOMALY_FEATURES, PlateFeatures, Standardizer
from screening_tools.records import QCEvent

PASS, WARN, FAIL = "pass", "warn", "fail"

# reason code -> (severity, description). Keep docs/user-guide/qc-rules.md in sync.
REASON_CODES: dict[str, tuple[str, str]] = {
    "QC_PLATE_CONTROLS_INVALID": (FAIL, "Too few valid control wells, or no assay window between controls"),
    "QC_PLATE_ZPRIME_FAIL": (FAIL, "Z'-factor below the fail limit"),
    "QC_PLATE_ZPRIME_MARGINAL": (WARN, "Z'-factor below the warning limit"),
    "QC_PLATE_NEG_CONTROL_OUT_OF_RANGE": (FAIL, "Negative-control mean outside calibration tolerance"),
    "QC_PLATE_POS_CONTROL_OUT_OF_RANGE": (FAIL, "Positive-control mean outside calibration tolerance"),
    "QC_PLATE_SIGNAL_TO_BACKGROUND_LOW": (FAIL, "Positive/negative control ratio below the minimum"),
    "QC_PLATE_CONTROL_CV_HIGH": (WARN, "Coefficient of variation of a control group above the limit"),
    "TELEMETRY_TEMP_OUT_OF_BAND": (WARN, "Instrument temperature outside the allowed band during the plate"),
    "QC_SAMPLE_QC_METRIC_HIGH": (FAIL, "Per-sample instrument QC metric above the limit"),
    "QC_SAMPLE_SATURATED": (FAIL, "Assay value at the detector maximum"),
    "QC_SAMPLE_ANOMALY": (WARN, "Isolation Forest anomaly score above the threshold"),
    "DRIFT_CONTROL_EWMA": (WARN, "Control level moving average outside its control limits"),
    "DRIFT_CONTROL_CUSUM": (WARN, "Cumulative sum of control deviations above its limit"),
    "DRIFT_REAGENT_LOT_SHIFT": (WARN, "Assay window changed after a reagent lot change"),
}


def make_event(code: str, plate_id: str, sample_id: str | None, value: float | None, limit: float | None, message: str) -> QCEvent:
    severity = REASON_CODES[code][0]
    v = None if value is None or not np.isfinite(value) else round(float(value), 4)
    lim = None if limit is None else round(float(limit), 4)
    return QCEvent(code, severity, plate_id, sample_id, v, lim, message)


@dataclass
class PlateQC:
    plate_id: str
    plate_status: str
    plate_codes: list[str]
    sample_status: list[str]  # per sample well, in PlateFeatures.sample_index order
    sample_codes: list[list[str]]
    anomaly_score: np.ndarray
    events: list[QCEvent] = field(default_factory=list)
    drift: dict = field(default_factory=dict)


def _status(codes: list[str]) -> str:
    severities = {REASON_CODES[c][0] for c in codes}
    return FAIL if FAIL in severities else WARN if WARN in severities else PASS


class AutoQC:
    def __init__(self, cfg: dict, anomaly_model, anomaly_scaler: Standardizer):
        self.cfg = cfg
        self.q = cfg["qc"]
        self.model = anomaly_model
        self.scaler = anomaly_scaler
        self.drift = DriftMonitor(cfg)

    def anomaly_scores(self, pf: PlateFeatures) -> np.ndarray:
        x = pf.anomaly_inputs()
        if x.shape[0] == 0:
            return np.zeros(0)
        # score_samples is higher for normal points; negate to get 0..1, higher = more unusual.
        return -self.model.score_samples(self.scaler.transform(x))

    def plate_rules(self, pf: PlateFeatures) -> list[QCEvent]:
        q, st, ctx = self.q, pf.stats, pf.plate.context
        pid = ctx.plate_id
        ev: list[QCEvent] = []
        if not st["controls_valid"]:
            ev.append(
                make_event(
                    "QC_PLATE_CONTROLS_INVALID",
                    pid,
                    None,
                    float(min(st["valid_negative_controls"], st["valid_positive_controls"])),
                    q["min_controls"],
                    "valid controls (min of negative/positive) or no positive window",
                )
            )
        else:
            z = st["zprime"]
            if z < q["zprime_fail"]:
                ev.append(make_event("QC_PLATE_ZPRIME_FAIL", pid, None, z, q["zprime_fail"], "Z'-factor"))
            elif z < q["zprime_warn"]:
                ev.append(make_event("QC_PLATE_ZPRIME_MARGINAL", pid, None, z, q["zprime_warn"], "Z'-factor"))
            if st["signal_to_background"] < q["signal_to_background_min"]:
                ev.append(
                    make_event("QC_PLATE_SIGNAL_TO_BACKGROUND_LOW", pid, None, st["signal_to_background"], q["signal_to_background_min"], "S/B")
                )
        tol = ctx.control_tolerance
        for code, mean, expected in (
            ("QC_PLATE_NEG_CONTROL_OUT_OF_RANGE", st["negative_mean"], ctx.expected_negative),
            ("QC_PLATE_POS_CONTROL_OUT_OF_RANGE", st["positive_mean"], ctx.expected_positive),
        ):
            if np.isfinite(mean) and abs(mean / expected - 1.0) > tol:
                ev.append(make_event(code, pid, None, mean / expected, 1.0 + np.sign(mean / expected - 1.0) * tol, "control mean / calibration"))
        for name in ("negative_cv", "positive_cv"):
            cv = st[name]
            if np.isfinite(cv) and cv > q["control_cv_warn"]:
                ev.append(make_event("QC_PLATE_CONTROL_CV_HIGH", pid, None, cv, q["control_cv_warn"], name.replace("_", " ")))
        if st["max_temperature_deviation"] > q["temperature_band"]:
            ev.append(
                make_event("TELEMETRY_TEMP_OUT_OF_BAND", pid, None, st["max_temperature_deviation"], q["temperature_band"], "max |temperature - set point| (C)")
            )
        return ev

    def evaluate(self, pf: PlateFeatures) -> PlateQC:
        q = self.q
        pid = pf.plate.context.plate_id
        plate_events = self.plate_rules(pf)
        plate_codes = [e.reason_code for e in plate_events]
        plate_status = _status(plate_codes)

        idx = pf.sample_index
        qc_metric = pf.plate.qc_metric[idx]
        assay = pf.plate.assay[idx]
        scores = self.anomaly_scores(pf)
        sat = self.cfg["simulator"]["saturation_value"]
        z_dev = np.abs(self.scaler.transform(pf.anomaly_inputs())) if idx.size else np.zeros((0, len(ANOMALY_FEATURES)))

        sample_events: list[QCEvent] = []
        sample_codes: list[list[str]] = []
        sample_status: list[str] = []
        for i in range(idx.size):
            sid = pf.plate.sample_id[idx[i]]
            codes: list[str] = []
            if qc_metric[i] > q["qc_metric_fail"]:
                codes.append("QC_SAMPLE_QC_METRIC_HIGH")
                sample_events.append(make_event("QC_SAMPLE_QC_METRIC_HIGH", pid, sid, qc_metric[i], q["qc_metric_fail"], "qc_metric"))
            if assay[i] >= sat:
                codes.append("QC_SAMPLE_SATURATED")
                sample_events.append(make_event("QC_SAMPLE_SATURATED", pid, sid, assay[i], sat, "assay_value"))
            if scores[i] > q["anomaly_threshold"]:
                codes.append("QC_SAMPLE_ANOMALY")
                top = ANOMALY_FEATURES[int(np.argmax(z_dev[i]))]
                sample_events.append(make_event("QC_SAMPLE_ANOMALY", pid, sid, scores[i], q["anomaly_threshold"], f"most deviating feature: {top}"))
            all_codes = codes + plate_codes
            sample_codes.append(all_codes)
            sample_status.append(_status(all_codes))

        drift_events, drift_point = self.drift.update(pf, plate_status)
        return PlateQC(
            plate_id=pid,
            plate_status=plate_status,
            plate_codes=plate_codes,
            sample_status=sample_status,
            sample_codes=sample_codes,
            anomaly_score=scores,
            events=plate_events + sample_events + drift_events,
            drift=drift_point,
        )
