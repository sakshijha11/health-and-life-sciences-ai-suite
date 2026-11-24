# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data records shared by every stage.

The ingestion stream is a sequence of four record types per plate:
``PlateStart`` (plate context), ``Sample`` (one per well), ``Telemetry``
(instrument readings) and ``PlateEnd``. Ground truth is never part of this
stream; the simulator writes it to a separate file for evaluation only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

WELL_SAMPLE = "sample"
WELL_NEGATIVE = "negative_control"
WELL_POSITIVE = "positive_control"
WELL_TYPES = (WELL_SAMPLE, WELL_NEGATIVE, WELL_POSITIVE)


@dataclass(slots=True)
class PlateStart:
    """Plate context: layout, reagent lot and calibration expectations."""

    run_id: str
    plate_id: str
    plate_index: int
    rows: int
    cols: int
    reagent_lot: str
    expected_negative: float
    expected_positive: float
    control_tolerance: float
    temperature_setpoint: float
    start_time: float  # simulated seconds since run start


@dataclass(slots=True)
class Sample:
    """One well as reported by the instrument."""

    run_id: str
    plate_id: str
    well: str  # e.g. "C05"
    sample_id: str  # e.g. "P0007-C05"
    well_type: str
    assay_value: float
    qc_metric: float
    read_time: float  # simulated seconds since run start
    descriptors: tuple[float, ...] = ()  # compound-library descriptors (empty for controls)


@dataclass(slots=True)
class Telemetry:
    """An instrument reading (Phase 1: temperature)."""

    plate_id: str
    sim_time: float
    instrument_temp: float


@dataclass(slots=True)
class PlateEnd:
    plate_id: str
    n_samples: int
    enqueued_at: float = 0.0  # wall-clock perf_counter when the record entered the buffer


@dataclass(slots=True)
class QCEvent:
    """A reason-coded QC, anomaly or drift event."""

    reason_code: str
    severity: str  # "warn" or "fail"
    plate_id: str
    sample_id: str | None
    value: float | None
    limit: float | None
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ResultRecord:
    """Final per-sample decision produced by decision fusion."""

    sample_id: str
    plate_id: str
    well: str
    qc_status: str  # "pass", "warn" or "fail"
    reason_codes: list[str] = field(default_factory=list)
    anomaly_score: float = 0.0
    hit_score: float = 0.0
    novelty_score: float = 0.0
    disposition: str = "non_hit"
    rank_in_plate: int | None = None
    rank_in_run: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def well_name(row: int, col: int) -> str:
    """0-based row/col to a well name such as ``A01``. Supports up to 26 rows."""
    return f"{chr(ord('A') + row)}{col + 1:02d}"


def plate_id_for(index: int) -> str:
    return f"P{index + 1:04d}"


def to_json_dict(record) -> dict:
    """Serialize a stream record with its type tag."""
    tag = {PlateStart: "plate_start", Sample: "sample", Telemetry: "telemetry", PlateEnd: "plate_end"}[type(record)]
    data = asdict(record)
    if tag == "sample":
        data["descriptors"] = list(record.descriptors)
    if tag == "plate_end":
        data.pop("enqueued_at", None)
    return {"type": tag, **data}


def from_json_dict(data: dict):
    kind = data.pop("type")
    if kind == "plate_start":
        return PlateStart(**data)
    if kind == "sample":
        data["descriptors"] = tuple(data.get("descriptors", ()))
        return Sample(**data)
    if kind == "telemetry":
        return Telemetry(**data)
    if kind == "plate_end":
        return PlateEnd(**data)
    raise ValueError(f"unknown record type: {kind!r}")
