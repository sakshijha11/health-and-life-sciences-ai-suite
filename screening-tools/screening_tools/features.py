# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Feature engineering engine (ITEP-97833).

One library used by both model training (``setup``) and the running
pipeline, so the models always see the features they were trained on.

Normalisation is per plate, using that plate's control wells. It never
depends on the inference batch or on other plates in the run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from screening_tools.records import WELL_NEGATIVE, WELL_POSITIVE, WELL_SAMPLE, PlateStart, Sample, Telemetry

# Bump when the feature list or any formula changes. Models built with a
# different version are refused at start-up.
FEATURE_VERSION = 1

# Per-sample engineered features (columns of PlateFeatures.matrix).
SAMPLE_FEATURES = (
    "percent_of_control",
    "robust_z",
    "qc_metric",
    "row_frac",
    "col_frac",
    "edge",
    "temperature_deviation",
)
# Inputs of the hit-scoring model: assay-derived features plus compound descriptors.
# The QC metric and telemetry are deliberately left to auto-QC, so that a
# high score on an untrustworthy read is caught by decision fusion.
HIT_FEATURES = ("percent_of_control", "robust_z", "row_frac", "col_frac", "edge")
# Inputs of the sample anomaly model.
ANOMALY_FEATURES = ("percent_of_control", "robust_z", "qc_metric", "temperature_deviation")

_EPS = 1e-9


@dataclass
class PlateData:
    """A completed plate as arrays, in well order."""

    context: PlateStart
    sample_id: list[str]
    well: list[str]
    well_type: np.ndarray  # str
    assay: np.ndarray
    qc_metric: np.ndarray
    read_time: np.ndarray
    row: np.ndarray
    col: np.ndarray
    descriptors: np.ndarray  # (n_wells, n_descriptors); zeros for controls
    telemetry_time: np.ndarray
    telemetry_temp: np.ndarray
    enqueued_at: float = 0.0


@dataclass
class PlateFeatures:
    plate: PlateData
    stats: dict  # plate-level features
    sample_index: np.ndarray  # well indices of sample wells
    matrix: np.ndarray  # (n_sample_wells, len(SAMPLE_FEATURES))
    descriptors: np.ndarray  # (n_sample_wells, n_descriptors)

    def column(self, name: str) -> np.ndarray:
        return self.matrix[:, SAMPLE_FEATURES.index(name)]

    def hit_inputs(self) -> np.ndarray:
        cols = [SAMPLE_FEATURES.index(n) for n in HIT_FEATURES]
        return np.hstack([self.matrix[:, cols], self.descriptors]).astype(np.float32)

    def anomaly_inputs(self) -> np.ndarray:
        cols = [SAMPLE_FEATURES.index(n) for n in ANOMALY_FEATURES]
        return self.matrix[:, cols]


def hit_feature_names(n_descriptors: int) -> list[str]:
    return list(HIT_FEATURES) + [f"descriptor_{i}" for i in range(n_descriptors)]


def assemble(context: PlateStart, samples: list[Sample], telemetry: list[Telemetry], n_descriptors: int) -> PlateData:
    """Turn the records of one plate into arrays."""
    n = len(samples)
    row = np.fromiter((ord(s.well[0]) - ord("A") for s in samples), dtype=np.int32, count=n)
    col = np.fromiter((int(s.well[1:]) - 1 for s in samples), dtype=np.int32, count=n)
    desc = np.zeros((n, n_descriptors), dtype=np.float64)
    for i, s in enumerate(samples):
        if s.descriptors:
            desc[i, : len(s.descriptors)] = s.descriptors
    return PlateData(
        context=context,
        sample_id=[s.sample_id for s in samples],
        well=[s.well for s in samples],
        well_type=np.array([s.well_type for s in samples]),
        assay=np.fromiter((s.assay_value for s in samples), dtype=np.float64, count=n),
        qc_metric=np.fromiter((s.qc_metric for s in samples), dtype=np.float64, count=n),
        read_time=np.fromiter((s.read_time for s in samples), dtype=np.float64, count=n),
        row=row,
        col=col,
        descriptors=desc,
        telemetry_time=np.array([t.sim_time for t in telemetry], dtype=np.float64),
        telemetry_temp=np.array([t.instrument_temp for t in telemetry], dtype=np.float64),
    )


def _control_stats(values: np.ndarray) -> tuple[float, float, int]:
    if values.size == 0:
        return float("nan"), float("nan"), 0
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return float(values.mean()), sd, int(values.size)


def compute(plate: PlateData, cfg: dict) -> PlateFeatures:
    """Compute plate-level and per-sample features for one plate (vectorised)."""
    qc = cfg["qc"]
    sat = cfg["simulator"]["saturation_value"]
    valid = (plate.qc_metric <= qc["qc_metric_fail"]) & (plate.assay < sat)
    neg_vals = plate.assay[(plate.well_type == WELL_NEGATIVE) & valid]
    pos_vals = plate.assay[(plate.well_type == WELL_POSITIVE) & valid]
    neg_mean, neg_sd, n_neg = _control_stats(neg_vals)
    pos_mean, pos_sd, n_pos = _control_stats(pos_vals)
    window = pos_mean - neg_mean
    controls_valid = bool(n_neg >= qc["min_controls"] and n_pos >= qc["min_controls"] and np.isfinite(window) and window > _EPS)

    if controls_valid:
        zprime = 1.0 - 3.0 * (pos_sd + neg_sd) / abs(window)
        sb = pos_mean / max(neg_mean, _EPS)
    else:
        zprime, sb = float("nan"), float("nan")
    neg_cv = neg_sd / neg_mean if n_neg and neg_mean > _EPS else float("nan")
    pos_cv = pos_sd / pos_mean if n_pos and pos_mean > _EPS else float("nan")

    if plate.telemetry_time.size:
        temp = np.interp(plate.read_time, plate.telemetry_time, plate.telemetry_temp)
        temp_dev = temp - plate.context.temperature_setpoint
        tele_dev = plate.telemetry_temp - plate.context.temperature_setpoint
        mean_dev, max_dev = float(tele_dev.mean()), float(np.abs(tele_dev).max())
    else:
        temp_dev = np.zeros_like(plate.assay)
        mean_dev = max_dev = 0.0

    idx = np.flatnonzero(plate.well_type == WELL_SAMPLE)
    x = plate.assay[idx]
    poc = 100.0 * (x - neg_mean) / window if controls_valid else np.zeros_like(x)
    med = np.median(x) if x.size else 0.0
    mad = 1.4826 * np.median(np.abs(x - med)) if x.size else 0.0
    if mad <= _EPS:
        mad = float(x.std()) if x.size and x.std() > _EPS else 1.0
    robust_z = (x - med) / mad

    rows, cols = plate.context.rows, plate.context.cols
    r, c = plate.row[idx], plate.col[idx]
    edge = ((r == 0) | (r == rows - 1) | (c == 0) | (c == cols - 1)).astype(np.float64)
    matrix = np.column_stack(
        [
            poc,
            robust_z,
            plate.qc_metric[idx],
            r / max(rows - 1, 1),
            c / max(cols - 1, 1),
            edge,
            temp_dev[idx],
        ]
    )
    stats = {
        "negative_mean": neg_mean,
        "negative_sd": neg_sd,
        "positive_mean": pos_mean,
        "positive_sd": pos_sd,
        "valid_negative_controls": n_neg,
        "valid_positive_controls": n_pos,
        "controls_valid": controls_valid,
        "zprime": zprime,
        "signal_to_background": sb,
        "negative_cv": neg_cv,
        "positive_cv": pos_cv,
        "mean_temperature_deviation": mean_dev,
        "max_temperature_deviation": max_dev,
    }
    return PlateFeatures(plate=plate, stats=stats, sample_index=idx, matrix=matrix, descriptors=plate.descriptors[idx])


class Standardizer:
    """Mean/std scaling fitted on training data and stored as JSON (never pickle)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.where(np.asarray(std, dtype=np.float32) < 1e-6, 1.0, np.asarray(std, dtype=np.float32)).astype(np.float32)

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        return cls(x.mean(axis=0), x.std(axis=0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"mean": self.mean.tolist(), "std": self.std.tolist()}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Standardizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(np.array(data["mean"]), np.array(data["std"]))
