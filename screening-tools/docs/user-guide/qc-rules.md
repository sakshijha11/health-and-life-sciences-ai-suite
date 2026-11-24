<!--hide_directive
```{eval-rst}
.. meta::
   :description: Features, QC rules, drift rules, reason codes and decision fusion rules of the Screening Tools sample application.
   :keywords: assay QC, Z'-factor, signal to background, coefficient of variation, EWMA, CUSUM, reagent lot, Isolation Forest, reason codes
```
hide_directive-->

# QC Rules and Reason Codes

This page lists every feature, rule, threshold and reason code that the
application uses. All thresholds are in `configs/default.toml` and can be
changed without code changes (see [Runtime Configuration](./runtime-configuration.md)).

> [!NOTE]
> These rules and default thresholds are a **proposal for review** by assay
> experts. They follow common screening practice, but they are tuned on
> simulated data only. See [Open questions](#open-questions-for-review).

## Features

The feature engine works on one completed plate at a time.

### Control statistics (per plate)

A control well is **valid** if its read-quality metric is at most
`qc.qc_metric_fail` and its value is below the detector maximum
(`simulator.saturation_value`). Only valid control wells are used.

| Statistic | Definition |
| --- | --- |
| μ<sub>n</sub>, σ<sub>n</sub> | Mean and standard deviation of the valid negative controls |
| μ<sub>p</sub>, σ<sub>p</sub> | Mean and standard deviation of the valid positive controls |
| Assay window | μ<sub>p</sub> − μ<sub>n</sub> |
| Z'-factor | 1 − 3 (σ<sub>p</sub> + σ<sub>n</sub>) / \|μ<sub>p</sub> − μ<sub>n</sub>\| |
| Signal-to-background (S/B) | μ<sub>p</sub> / μ<sub>n</sub> |
| Control CV | σ / μ, for each control group |
| Control ratio | μ / expected level from the plate's calibration context, for each control group |
| Temperature deviation | Largest \|temperature − set point\| over the plate's telemetry |

### Sample features (per sample well)

| Feature | Definition | Used by |
| --- | --- | --- |
| `percent_of_control` | 100 × (x − μ<sub>n</sub>) / (μ<sub>p</sub> − μ<sub>n</sub>) | hit model, anomaly model |
| `robust_z` | (x − median) / (1.4826 × MAD), over the plate's sample wells | hit model, anomaly model |
| `qc_metric` | Read-quality metric from the instrument, 0 (good) to 1 (bad) | sample rule, anomaly model |
| `row_frac`, `col_frac` | Well row and column scaled to 0–1 | hit model |
| `edge` | 1 for outer-row or outer-column wells, else 0 | hit model |
| `temperature_deviation` | Instrument temperature at the well's read time minus the set point | anomaly model |
| `descriptor_0` … `descriptor_7` | Compound descriptors from the library | hit model |

The read-quality metric and the temperature are deliberately **not** inputs of
the hit model. A high hit score on an untrustworthy read must be caught by QC
and fusion, not hidden inside the score.

## Plate rules

A plate's status is `fail` if any of its codes is a fail, `warn` if any is a
warn, otherwise `pass`. Plate codes apply to every sample on the plate.

| Reason code | Severity | Fires when | Default |
| --- | --- | --- | --- |
| `QC_PLATE_CONTROLS_INVALID` | fail | Fewer than `min_controls` valid wells in either control group, or assay window ≤ 0 | `min_controls = 4` |
| `QC_PLATE_ZPRIME_FAIL` | fail | Z' < `zprime_fail` | `0.3` |
| `QC_PLATE_ZPRIME_MARGINAL` | warn | `zprime_fail` ≤ Z' < `zprime_warn` | `0.5` |
| `QC_PLATE_SIGNAL_TO_BACKGROUND_LOW` | fail | S/B < `signal_to_background_min` | `2.0` |
| `QC_PLATE_NEG_CONTROL_OUT_OF_RANGE` | fail | \|negative control ratio − 1\| > `control_tolerance` | `0.25` (±25%) |
| `QC_PLATE_POS_CONTROL_OUT_OF_RANGE` | fail | \|positive control ratio − 1\| > `control_tolerance` | `0.25` (±25%) |
| `QC_PLATE_CONTROL_CV_HIGH` | warn | CV of either control group > `control_cv_warn` | `0.20` |
| `TELEMETRY_TEMP_OUT_OF_BAND` | warn | Temperature deviation > `temperature_band` | `1.0` °C |

Z' and S/B are not checked when the controls are invalid; the plate already
fails with `QC_PLATE_CONTROLS_INVALID`.

## Sample rules

A sample's QC status is the worst severity of its own codes **and** its plate's
codes.

| Reason code | Severity | Fires when | Default |
| --- | --- | --- | --- |
| `QC_SAMPLE_QC_METRIC_HIGH` | fail | `qc_metric` > `qc_metric_fail` | `0.25` |
| `QC_SAMPLE_SATURATED` | fail | Assay value ≥ detector maximum | `4000` |
| `QC_SAMPLE_ANOMALY` | warn | Anomaly score > `anomaly_threshold` | `0.62` |

### Anomaly score

An Isolation Forest (200 trees) scores every sample on standardised
`percent_of_control`, `robust_z`, `qc_metric` and `temperature_deviation`. The
score runs from 0 to 1; higher is more unusual. The event message names the
feature that deviates most from the training data.

The forest is trained during setup on fault-free plates. Real actives are rare,
so on their own they would look unusual and be flagged. They are therefore
repeated in the training set until they make up about one third of it: a high
activity with clean measurements is then normal, and only unusual measurement
conditions score high.

Measured on the 100 test plates during setup (Windows, PTL):

| Measurement | Result |
| --- | --- |
| Clean samples above the threshold | 0.26% |
| Clean **active** samples above the threshold | 3.6% |
| Well artifacts above the threshold | 78% |

## Drift rules

Drift rules watch two series across plates: the **negative** and **positive
control ratio** (control mean / expected level). They need no training; each
rule is its own explanation. Drift events are reported as plate events but do
**not** change sample QC status or the hit lists. They tell the operator that
the instrument or reagents need attention.

- **Baseline.** The mean μ and standard deviation σ of each series are learned
  from the first `baseline_plates` usable plates (σ is at least 0.005). A
  plate is usable if it did not fail plate QC, so one bad plate does not look
  like drift.
- **After a reagent lot change** the baseline is learned again from the next
  `baseline_plates` usable plates.

| Reason code | Severity | Fires when | Default |
| --- | --- | --- | --- |
| `DRIFT_CONTROL_EWMA` | warn | EWMA<sub>t</sub> = λ x<sub>t</sub> + (1 − λ) EWMA<sub>t−1</sub> leaves μ ± L σ √(λ / (2 − λ)). One event when it leaves; re-arms when it returns inside | λ = `ewma_lambda` = 0.3, L = `ewma_limit_sigma` = 3.5 |
| `DRIFT_CONTROL_CUSUM` | warn | With z = (x − μ) / σ: S⁺ = max(0, S⁺ + z − k), S⁻ = max(0, S⁻ − z − k). Fires when S⁺ or S⁻ > h. One event per excursion and direction; re-arms when the sum returns to 0 | k = `cusum_k_sigma` = 0.5, h = `cusum_h_sigma` = 6.0 |
| `DRIFT_REAGENT_LOT_SHIFT` | warn | After a lot change, the mean assay-window ratio of the first `lot_window` usable plates differs from the last `lot_window` usable plates before it by more than `lot_shift_warn` | `lot_window = 4`, `lot_shift_warn = 0.08` (8%) |

The assay-window ratio is (μ<sub>p</sub> − μ<sub>n</sub>) / (expected positive
− expected negative).

## Decision fusion

| Disposition | Rule |
| --- | --- |
| `candidate` (tier 1) | Hit score ≥ `hit_scoring.hit_threshold` (0.5) and QC status `pass` or `warn` |
| `candidate_qc_flagged` (tier 2) | Hit score ≥ threshold and QC status `fail`. Note: *high hit score but QC failure, possible false positive* |
| `non_hit` | Hit score < threshold |

Tier 1 is ranked above tier 2. Within a tier: hit score, highest first, then
sample id. Novelty is shown but does not change the order. Hit lists are
produced per plate and per run.

## Measured on simulated data

Windows, Intel® Core™ Ultra 7 356H, 600 plates (48,000 samples). Plate indices
start at 0.

| Check | All faults on | No faults (`--clean`) |
| --- | --- | --- |
| Bad plates caught by a plate fail code | 2 of 2 | – |
| Temperature excursion plates flagged | 4 of 4 | – |
| Reagent lot shift (change at plate 30) | raised at plate 33 | – |
| Slow drift (starts at plate 10) | detected after 6 plates | – |
| Well artifacts flagged by sample QC | 253 of 284 (89%) | – |
| Tier 1 precision / recall | 0.972 / 0.983 | 1.000 / 0.998 |
| Tier 1 + 2 recall | 0.997 | 0.999 |
| Events on fault-free plates | CUSUM 5, EWMA 8, CV high 4, Z' marginal 9 (563 plates) | CUSUM 6, EWMA 1, CV high 4 (600 plates) |

Hit scoring on the CPU and on the iGPU gave the same dispositions.

## Open questions for review

1. **Anomaly severity.** `QC_SAMPLE_ANOMALY` is a warning, so a hit flagged
   only by the anomaly model stays in tier 1. In the 600-plate faulted run, 42
   of 284 artifact wells stayed in tier 1 this way. Making it a fail would move
   them to tier 2, at the cost of also moving the 3.6% of clean actives that
   score above the threshold.
2. **Thresholds.** Z' 0.5 / 0.3, S/B 2.0, control tolerance ±25%, control CV
   20% and the temperature band ±1 °C are common starting values. Please
   confirm or give assay-specific values.
3. **Drift sensitivity.** The EWMA and CUSUM limits were set to keep false
   alarms low on simulated data (7 drift events in a 600-plate run without
   faults). Real instrument data may need different limits.
4. **Drift consequences.** Drift events do not change the hit lists. Should a
   drift alarm also mark the affected plates, for example as `warn`?
