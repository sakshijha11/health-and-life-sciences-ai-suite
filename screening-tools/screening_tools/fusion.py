# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decision fusion engine (ITEP-97837).

Joins the auto-QC result and the hit score of every sample by sample id and
gives each sample one disposition:

==========================  ==============================================
``candidate``               hit, QC clean (pass or warn)
``candidate_qc_flagged``    hit, but QC failed: "high hit score but QC
                            failure, possible false positive"
``non_hit``                 below the hit threshold
==========================  ==============================================

Ranking uses two tiers (agreed with the requester: QC-flagged hits are kept,
ranked lower): every ``candidate`` first, ordered by activity probability,
then every ``candidate_qc_flagged``. Ties break on ``sample_id`` so the order
never depends on timing. Novelty is reported but does not affect the order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from screening_tools.features import PlateFeatures
from screening_tools.qc import FAIL, PlateQC
from screening_tools.records import ResultRecord

CANDIDATE = "candidate"
CANDIDATE_QC_FLAGGED = "candidate_qc_flagged"
NON_HIT = "non_hit"
FLAG_NOTE = "high hit score but QC failure, possible false positive"
_TIER = {CANDIDATE: 0, CANDIDATE_QC_FLAGGED: 1}


@dataclass
class PlateResult:
    plate_id: str
    plate_index: int
    plate_status: str
    records: list[ResultRecord]  # all sample wells
    hit_list: list[ResultRecord]  # ranked candidates (tier 1 then tier 2)


def rank_key(r: ResultRecord) -> tuple:
    return (_TIER[r.disposition], -r.hit_score, r.sample_id)


def fuse(pf: PlateFeatures, qc: PlateQC, activity: np.ndarray, novelty: np.ndarray, threshold: float) -> PlateResult:
    if qc.plate_id != pf.plate.context.plate_id:
        raise ValueError(f"fusion join mismatch: {qc.plate_id} != {pf.plate.context.plate_id}")
    records = []
    for i, well_idx in enumerate(pf.sample_index):
        status = qc.sample_status[i]
        hit = bool(activity[i] >= threshold)
        disposition = NON_HIT if not hit else CANDIDATE_QC_FLAGGED if status == FAIL else CANDIDATE
        records.append(
            ResultRecord(
                sample_id=pf.plate.sample_id[well_idx],
                plate_id=qc.plate_id,
                well=pf.plate.well[well_idx],
                qc_status=status,
                reason_codes=list(qc.sample_codes[i]),
                anomaly_score=round(float(qc.anomaly_score[i]), 4),
                hit_score=round(float(activity[i]), 4),
                novelty_score=round(float(novelty[i]), 4),
                disposition=disposition,
            )
        )
    hits = sorted((r for r in records if r.disposition != NON_HIT), key=rank_key)
    for rank, r in enumerate(hits, start=1):
        r.rank_in_plate = rank
    return PlateResult(qc.plate_id, pf.plate.context.plate_index, qc.plate_status, records, hits)


def rank_run(hits: list[ResultRecord]) -> list[ResultRecord]:
    """Final per-run hit list. Assigns ``rank_in_run``."""
    ranked = sorted(hits, key=rank_key)
    for rank, r in enumerate(ranked, start=1):
        r.rank_in_run = rank
    return ranked
