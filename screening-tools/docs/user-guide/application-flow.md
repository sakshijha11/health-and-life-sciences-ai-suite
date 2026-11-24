<!--hide_directive
```{eval-rst}
.. meta::
   :description: End-to-end flow of the Screening Tools sample application: data ingestion, feature engineering, auto-QC, drift monitoring, iGPU hit scoring, decision fusion and outputs.
   :keywords: application flow, pipeline, plate reader, ingestion, auto-QC, hit scoring, decision fusion, OpenVINO
```
hide_directive-->

# Application Flow

```text
source ──► bounded buffer ──► feature engine ──┬──► auto-QC + drift (CPU) ──┐
                                               └──► hit scoring (iGPU)   ───┴──► decision fusion ──► outputs + dashboard
```

Each stage runs in its own thread. Stages are connected by bounded queues, so a
slow stage makes the source wait (back-pressure) instead of using more memory.
Auto-QC and hit scoring work on the same plate at the same time.

## 1. Data source

The pipeline reads a stream of four record types per plate:

| Record | Content |
| --- | --- |
| `plate_start` | Plate id and index, layout, reagent lot, expected control levels, control tolerance, temperature set point |
| `sample` | One well: sample id, well, well type (sample, negative or positive control), **one assay value**, a read-quality metric, read time, compound descriptors |
| `telemetry` | Instrument temperature readings during the plate |
| `plate_end` | Number of wells, closes the plate |

Phase 1 has two sources:

- **Simulator** (default). Generates plates from the `[simulator]` settings.
  Column 1 holds negative controls and column 12 positive controls, so a
  96-well plate has 80 sample wells. About 3% of samples are active. Active
  compounds belong to known or novel chemical families, which the compound
  descriptors reflect.
- **Replay**. `simulate` writes a run to a JSON Lines file; `run --replay`
  plays it back. A replay gives the same results as the live run with the same
  settings.

The simulator writes the true labels (active, novel, which faults apply) to a
separate `ground_truth.jsonl`. The pipeline never reads this file. It is used
only for the demo-mode evaluation.

### Fault scenarios

All are on by default and switched off with `--clean`. Plate indices start at 0.

| Scenario | Default | Effect |
| --- | --- | --- |
| Slow drift | plates 10–27 | Raw signal gain grows by 0.6% per plate, then the instrument is recalibrated |
| Reagent lot change | plate 30 | Lot `LOT-A` → `LOT-B`; positive control drops to 85% |
| Temperature excursion | plates 40–43 | +3 °C, noisier reads |
| Bad plate | plates 22 and 47 | Positive controls collapse to 45% |
| Well artifacts | 0.6% of sample wells | Dispense or bubble artifact; 80% of them also raise the read-quality metric |

## 2. Feature engine (CPU)

When a plate is complete, the feature engine computes plate-level statistics
from the control wells and per-sample features. Normalisation uses the plate's
own controls only, so results never depend on batch size or on other plates.
The same code is used for model training and at run time. See
[QC Rules](./qc-rules.md#features) for every feature.

## 3. Auto-QC and drift (CPU)

- Plate rules on the controls (Z'-factor, control range, control CV,
  signal-to-background, temperature).
- Sample rules (read-quality metric, saturation) and an Isolation Forest
  anomaly score.
- Drift rules across plates (EWMA, CUSUM, reagent lot shift).

Every finding is an event with a reason code, a severity (`warn` or `fail`),
the measured value and the limit. A sample's QC status is the worst severity
of its own codes and its plate's codes. See [QC Rules](./qc-rules.md).

## 4. Hit scoring (iGPU)

A multilayer perceptron with two outputs, **activity** and **novelty**, scores
every sample. It is trained with scikit-learn during setup and rebuilt as an
OpenVINO™ model from its weights, so the run time needs no PyTorch. Inputs are
the assay features (percent of control, robust Z, well position) and the
compound descriptors. The read-quality metric and the temperature are left out
on purpose: they belong to auto-QC, and fusion catches a high score on an
untrustworthy read.

The device is set explicitly (`GPU` by default). If it is not available, the
run stops with an error; it does not fall back to the CPU without telling you.

## 5. Decision fusion

Fusion joins the QC result and the hit score of every sample by sample id.

| Disposition | Rule |
| --- | --- |
| `candidate` | Activity ≥ `hit_threshold` and QC status `pass` or `warn` |
| `candidate_qc_flagged` | Activity ≥ `hit_threshold` and QC status `fail`. Note: *high hit score but QC failure, possible false positive* |
| `non_hit` | Activity < `hit_threshold` |

Ranking has two tiers. Tier 1 is every `candidate`, then tier 2 is every
`candidate_qc_flagged`. Within a tier, samples are ordered by activity score
(highest first), then by sample id, so the order never depends on timing.
Novelty is shown as a column and does not change the order.

Ranking is done per plate (`rank_in_plate`) and once more over the whole run
(`rank_in_run`).

## 6. Outputs

`results.csv`, `hits_by_plate.csv` and `hits_run.csv` have these columns:

| Column | Content |
| --- | --- |
| `sample_id`, `plate_id`, `well` | Identity, for example `P0007-C05` |
| `qc_status` | `pass`, `warn` or `fail` |
| `reason_codes` | Sample and plate reason codes, separated by `;` |
| `anomaly_score` | Isolation Forest score, 0–1, higher is more unusual |
| `hit_score` | Activity probability, 0–1 |
| `novelty_score` | Probability that an active belongs to a novel chemical family, 0–1 |
| `disposition` | `candidate`, `candidate_qc_flagged` or `non_hit` |
| `rank_in_plate`, `rank_in_run` | Rank in the plate and run hit lists (empty for non-hits) |

`events.csv` has `reason_code`, `severity`, `plate_id`, `sample_id` (empty for
plate and drift events), `value`, `limit` and `message`.

`summary.json` records settings, devices, hardware and software versions,
throughput, per-stage latency, utilisation, back-pressure, disposition and
event counts, and the demo-mode evaluation.

## Demo-mode evaluation

For simulated runs, and for replays given `--truth`, the summary and the
dashboard compare the results with the ground truth:

- Tier 1 and tier 1 + 2 precision and recall against the true actives.
- Detection of every fault scenario: bad plates caught by a plate fail code,
  temperature excursion plates flagged, reagent lot shift raised within the
  comparison window, slow-drift detection delay, and share of artifacts
  flagged or left in tier 1.
- Events on plates without any fault (false alarms). A plate counts as
  fault-free only if it and the `lot_window` plates before it had no fault.

## Telemetry

- Throughput and latency exclude the first `warmup_plates` plates.
- Latency is per plate and per stage, reported as P50 and P95. End-to-end
  latency includes time spent waiting in the buffer.
- CPU utilisation comes from `psutil`. iGPU utilisation is the busiest GPU
  engine from the Windows *GPU Engine* performance counters, system-wide. It is
  not available on other operating systems.
- UI refresh lag is how old the newest result is when the dashboard has drawn
  it.
