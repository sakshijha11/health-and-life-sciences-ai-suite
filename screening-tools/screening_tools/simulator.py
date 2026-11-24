# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seeded high-throughput-screening instrument simulator.

Produces runs of plates with sample wells, negative and positive control
wells, instrument telemetry and switchable fault scenarios. Everything is
drawn from one explicitly seeded ``numpy.random.Generator``; the same seed
and configuration always produce the same run.

Ground truth (which samples are truly active or novel, which faults were
injected) is returned separately in ``SimPlate.truth`` and is written to its
own file. It is never part of the record stream the pipeline consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from screening_tools.records import (
    WELL_NEGATIVE,
    WELL_POSITIVE,
    WELL_SAMPLE,
    PlateEnd,
    PlateStart,
    Sample,
    Telemetry,
    plate_id_for,
    well_name,
)

# The compound library (positions of the active compound families) is a fixed
# property of the simulated world, shared by model training and every run.
LIBRARY_SEED = 20270
# Independent random streams derived from the run seed.
STREAM_TRAINING = 1
STREAM_RUN = 2

FAULT_SLOW_DRIFT = "slow_drift"
FAULT_LOT_CHANGE = "reagent_lot_change"
FAULT_TEMPERATURE = "temperature_excursion"
FAULT_BAD_PLATE = "bad_plate"
FAULT_WELL_ARTIFACT = "well_artifacts"

NOVEL_ACTIVE_SHARE = 0.25


@dataclass
class SimPlate:
    context: PlateStart
    samples: list[Sample]
    telemetry: list[Telemetry]
    truth: dict  # ground truth for this plate; evaluation only

    def records(self) -> Iterator:
        """The plate as an ingestion stream: start, samples and telemetry in read order, end."""
        yield self.context
        tele = iter(self.telemetry)
        nxt = next(tele, None)
        for sample in self.samples:
            while nxt is not None and nxt.sim_time <= sample.read_time:
                yield nxt
                nxt = next(tele, None)
            yield sample
        while nxt is not None:
            yield nxt
            nxt = next(tele, None)
        yield PlateEnd(plate_id=self.context.plate_id, n_samples=len(self.samples))


def chemotype_centres(n_known: int, n_novel: int, n_descriptors: int) -> np.ndarray:
    rng = np.random.default_rng(LIBRARY_SEED)
    return rng.normal(0.0, 2.2, size=(n_known + n_novel, n_descriptors))


class Simulator:
    """Generates plates for one run (or for model training)."""

    def __init__(self, cfg: dict, stream: int = STREAM_RUN, seed: int | None = None, plates: int | None = None):
        self.cfg = cfg
        self.sim = cfg["simulator"]
        self.faults = cfg["faults"]
        self.n_plates = cfg["run"]["plates"] if plates is None else plates
        self.run_id = cfg["run"]["run_id"]
        self.seed = cfg["run"]["seed"] if seed is None else seed
        self.stream = stream

        p = cfg["plate"]
        self.rows, self.cols = p["rows"], p["cols"]
        r, c = np.divmod(np.arange(self.rows * self.cols), self.cols)
        self.row, self.col = r, c
        self.well_type = np.full(r.shape, WELL_SAMPLE, dtype=object)
        self.well_type[c == p["negative_control_column"] - 1] = WELL_NEGATIVE
        self.well_type[c == p["positive_control_column"] - 1] = WELL_POSITIVE
        self.is_sample = self.well_type == WELL_SAMPLE
        self.edge = (r == 0) | (r == self.rows - 1) | (c == 0) | (c == self.cols - 1)
        self.wells = [well_name(int(a), int(b)) for a, b in zip(r, c)]
        self.centres = chemotype_centres(self.sim["known_chemotypes"], self.sim["novel_chemotypes"], self.sim["descriptors"])

    # -- fault helpers -------------------------------------------------
    def _on(self, name: str) -> dict | None:
        f = self.faults[name]
        return f if f["enabled"] else None

    def _plate_conditions(self, idx: int) -> dict:
        cond = {"gain": 1.0, "window": 1.0, "pos_control": 1.0, "temp_offset": 0.0, "noise": 1.0, "lot": "LOT-A", "faults": []}
        if f := self._on(FAULT_SLOW_DRIFT):
            if f["start_plate"] <= idx < f["start_plate"] + f["plates"]:
                cond["gain"] += f["gain_per_plate"] * (idx - f["start_plate"] + 1)
                cond["faults"].append(FAULT_SLOW_DRIFT)
        if f := self._on(FAULT_LOT_CHANGE):
            if idx >= f["at_plate"]:
                cond["window"] = f["positive_factor"]
                cond["lot"] = "LOT-B"
                if idx == f["at_plate"]:
                    cond["faults"].append(FAULT_LOT_CHANGE)
        if f := self._on(FAULT_TEMPERATURE):
            if f["start_plate"] <= idx < f["start_plate"] + f["plates"]:
                cond["temp_offset"] = f["delta_c"]
                cond["noise"] = f["noise_factor"]
                cond["faults"].append(FAULT_TEMPERATURE)
        if f := self._on(FAULT_BAD_PLATE):
            if idx in f["plates"]:
                cond["pos_control"] = f["positive_factor"]
                cond["faults"].append(FAULT_BAD_PLATE)
        return cond

    # -- generation ----------------------------------------------------
    def plate(self, idx: int) -> SimPlate:
        # Each plate has its own random stream: plate k is the same whether it
        # is generated alone or as part of a run.
        s, rng = self.sim, np.random.default_rng([self.seed, self.stream, idx])
        n = self.rows * self.cols
        cond = self._plate_conditions(idx)
        plate_id = plate_id_for(idx)
        start_time = idx * s["plate_read_seconds"]

        # Biology: which compounds are active and how strongly.
        active = self.is_sample & (rng.random(n) < s["active_rate"])
        fraction = rng.normal(0.0, 0.03, n)
        fraction[active] = rng.uniform(0.3, 1.0, int(active.sum()))
        fraction[self.well_type == WELL_NEGATIVE] = 0.0
        fraction[self.well_type == WELL_POSITIVE] = 1.0

        # Compound descriptors: actives come from chemotype families.
        n_known = s["known_chemotypes"]
        n_families = n_known + s["novel_chemotypes"]
        descriptors = rng.normal(0.0, 1.6, size=(n, s["descriptors"]))
        novel = np.zeros(n, dtype=bool)
        idx_active = np.flatnonzero(active)
        if idx_active.size:
            is_novel = rng.random(idx_active.size) < NOVEL_ACTIVE_SHARE if s["novel_chemotypes"] else np.zeros(idx_active.size, bool)
            family = np.where(
                is_novel,
                rng.integers(n_known, max(n_families, n_known + 1), idx_active.size),
                rng.integers(0, n_known, idx_active.size),
            )
            descriptors[idx_active] = self.centres[family] + rng.normal(0.0, 0.6, size=(idx_active.size, s["descriptors"]))
            novel[idx_active] = is_novel

        # Measurement: assay window, controls, noise, edge effect and gain.
        neg = s["negative_level"]
        pos = neg + (s["positive_level"] - neg) * cond["window"]
        level = neg + (pos - neg) * fraction
        level[self.well_type == WELL_POSITIVE] = neg + (pos - neg) * cond["pos_control"]
        noise_sd = (s["negative_noise"] + (s["positive_noise"] - s["negative_noise"]) * np.clip(fraction, 0, 1)) * cond["noise"]
        signal = level + rng.normal(0.0, 1.0, n) * noise_sd
        signal = signal * np.where(self.edge, 1.0 + s["edge_effect"], 1.0) * cond["gain"]

        m = s["qc_metric_mean"]
        qc_metric = rng.beta(2.0, 2.0 / m - 2.0, n)

        artifact = np.zeros(n, dtype=bool)
        if f := self._on(FAULT_WELL_ARTIFACT):
            artifact = self.is_sample & (rng.random(n) < f["rate"])
            k = int(artifact.sum())
            if k:
                signal[artifact] += rng.uniform(0.6, 1.2, k) * (s["positive_level"] - neg) * cond["gain"]
                visible = rng.random(k) < f["visible_fraction"]
                qc_metric[np.flatnonzero(artifact)[visible]] = rng.uniform(0.3, 0.8, int(visible.sum()))
        signal = np.clip(signal, 0.0, s["saturation_value"])

        read_step = s["plate_read_seconds"] / n
        read_time = start_time + np.arange(n) * read_step

        p = self.cfg["plate"]

        def edge_gain(column: int) -> float:
            return 1.0 + s["edge_effect"] if column in (1, self.cols) else 1.0

        context = PlateStart(
            run_id=self.run_id,
            plate_id=plate_id,
            plate_index=idx,
            rows=self.rows,
            cols=self.cols,
            reagent_lot=cond["lot"],
            expected_negative=s["negative_level"] * edge_gain(p["negative_control_column"]),
            expected_positive=s["positive_level"] * edge_gain(p["positive_control_column"]),
            control_tolerance=self.cfg["qc"]["control_tolerance"],
            temperature_setpoint=s["temperature_setpoint"],
            start_time=float(start_time),
        )
        samples = []
        for i in range(n):
            desc = tuple(round(float(v), 5) for v in descriptors[i]) if self.is_sample[i] else ()
            samples.append(
                Sample(
                    run_id=self.run_id,
                    plate_id=plate_id,
                    well=self.wells[i],
                    sample_id=f"{plate_id}-{self.wells[i]}",
                    well_type=str(self.well_type[i]),
                    assay_value=round(float(signal[i]), 4),
                    qc_metric=round(float(qc_metric[i]), 5),
                    read_time=round(float(read_time[i]), 3),
                    descriptors=desc,
                )
            )
        k = s["telemetry_per_plate"]
        t_times = start_time + (np.arange(k) + 0.5) * s["plate_read_seconds"] / max(k, 1)
        temps = s["temperature_setpoint"] + cond["temp_offset"] + rng.normal(0.0, s["temperature_noise"], k)
        telemetry = [Telemetry(plate_id, round(float(t), 3), round(float(v), 4)) for t, v in zip(t_times, temps)]

        truth = {
            "plate_id": plate_id,
            "plate_index": idx,
            "faults": cond["faults"],
            "samples": [
                {
                    "sample_id": samples[i].sample_id,
                    "is_active": bool(active[i]),
                    "is_novel": bool(novel[i]),
                    "fault": FAULT_WELL_ARTIFACT if artifact[i] else None,
                }
                for i in range(n)
                if self.is_sample[i]
            ],
        }
        return SimPlate(context, samples, telemetry, truth)

    def __iter__(self) -> Iterator[SimPlate]:
        for idx in range(self.n_plates):
            yield self.plate(idx)
