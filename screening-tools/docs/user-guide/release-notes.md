<!--hide_directive
```{eval-rst}
.. meta::
   :description: Release notes of the Screening Tools sample application.
   :keywords: release notes, screening tools, version history
```
hide_directive-->

# Release Notes

## Version 0.1.0 (Phase 1)

**Release date:** to be announced

First release of the Screening Tools reference application.

### New features

- Plate simulator with one value per well, controls, compound descriptors,
  instrument temperature telemetry and five labelled fault scenarios: slow
  drift, reagent lot change, temperature excursion, bad plate and well
  artifacts.
- Recording and replay of runs (JSON Lines).
- Bounded ingestion buffer with back-pressure, and a threaded pipeline with
  auto-QC and hit scoring running in parallel.
- Feature engine with per-plate normalisation to the plate's own controls,
  shared between model training and run time.
- CPU auto-QC: plate rules (Z'-factor, signal-to-background, control range,
  control CV, temperature), sample rules (QC metric, saturation) and an
  Isolation Forest anomaly score. Every finding has a reason code.
- Rule-based drift monitoring: EWMA, CUSUM and reagent lot shift.
- iGPU hit scoring with OpenVINO™: activity and novelty scores, explicit
  device selection without a silent fallback.
- Decision fusion with two-tier hit lists per plate and per run. QC-failed
  hits are kept and flagged as possible false positives.
- Local web dashboard with live KPIs, pipeline stages, hit list, events,
  control trend, plate heat map, throughput and CPU/iGPU utilisation.
- CPU vs iGPU hit-scoring benchmark with a ranking agreement check.
- Demo-mode evaluation against the simulator's ground truth.
- Windows setup and run scripts.

### Differences from the proof of concept

- The hit model is trained with scikit-learn and rebuilt as an OpenVINO™
  model from its weights. The application does not need PyTorch or a model
  download.
- Novelty comes from the compound descriptors, through a second output of the
  hit model.
- Auto-QC on the CPU, not hit scoring, limits the end-to-end throughput.

### Known limitations

- Simulated data only. The rules and thresholds are tuned on simulated data
  and are awaiting review by assay experts (see
  [QC Rules](./qc-rules.md#open-questions-for-review)).
- Validated on Windows 11 with an Intel® Core™ Ultra processor. The pipeline
  also runs on Linux, where iGPU utilisation is not measured.
- The dashboard is local only (`127.0.0.1`) and has no login.
- The NPU is not used.
