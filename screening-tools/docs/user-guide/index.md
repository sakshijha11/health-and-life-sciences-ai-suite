<!--hide_directive
```{eval-rst}
.. meta::
   :description: Overview of the Screening Tools sample application, which uses OpenVINO™ on Intel hardware for real-time quality control, drift monitoring and hit scoring of high-throughput screening assays.
   :keywords: high-throughput screening, assay QC, Z'-factor, drift detection, hit scoring, OpenVINO, Intel iGPU, Intel Core Ultra
```
hide_directive-->

# Screening Tools Sample Application

<!--hide_directive
::::{container} component_header_row
<div class="component_card_widget">
  <a class="icon_github" href="https://github.com/open-edge-platform/health-and-life-sciences-ai-suite/tree/main/screening-tools">
     GitHub
  </a>
  <a class="icon_document" href="https://github.com/open-edge-platform/health-and-life-sciences-ai-suite/blob/main/screening-tools/README.md">
     Readme
  </a>
</div>
hide_directive-->

> [!NOTE]
> This application is for **reference and evaluation purposes only**. It uses
> **simulated** assay data and is **not intended for direct use in laboratory,
> clinical or diagnostic decisions**. It is not validated for such a purpose.

<!--hide_directive :::: hide_directive-->

High-throughput screening tests thousands of compounds on 96- or 384-well
plates. A plate reader produces one value per well. Problems such as a weak
reagent lot, a slow instrument drift, a temperature excursion or a bad dispense
are often found only after the run, when the data is analysed.

This application checks the data while the run is going, on one Intel® Core™
Ultra PC:

- **Auto-QC on the CPU.** Plate rules on the control wells, sample rules, and an
  Isolation Forest anomaly score. Every finding has a reason code.
- **Drift monitoring on the CPU.** EWMA and CUSUM rules on the control levels
  across plates, plus a check after each reagent lot change.
- **Hit scoring on the Intel® iGPU.** A small neural network, run with the
  OpenVINO™ toolkit, gives every sample an activity and a novelty score.
- **Decision fusion.** Hits with a QC failure are kept but ranked below clean
  hits, with the note *"high hit score but QC failure, possible false
  positive"*.
- **Dashboard.** A local web page shows the hit list, events, control trend,
  plate heat map, throughput and CPU/iGPU utilisation while the run is going.

Phase 1 uses a built-in simulator with labelled fault scenarios, so every
detection can be checked against ground truth. A recorded run can be replayed
instead of the live simulator.

## Next Steps

- See [Get Started](./get-started.md) to install and run the application.
- See [Application Flow](./application-flow.md) for what each stage does and
  which files a run writes.
- See [QC Rules and Reason Codes](./qc-rules.md) for every feature, rule,
  threshold and reason code.
- See [Runtime Configuration](./runtime-configuration.md) for all settings and
  command-line options.
- See [Benchmarking](./benchmarking.md) for measured CPU and iGPU results.
- See [Troubleshooting](./troubleshooting.md) for common issues.
- See [Release Notes](./release-notes.md) for version history.
- See [Uninstall](./uninstall.md) to remove the application.

<!--hide_directive
:::{toctree}
:hidden:

Get Started <get-started.md>
Application Flow <./application-flow.md>
QC Rules and Reason Codes <./qc-rules.md>
Runtime Configuration <./runtime-configuration.md>
Benchmarking <./benchmarking.md>
Troubleshooting <./troubleshooting.md>
Release Notes <./release-notes.md>
Uninstall <./uninstall.md>

:::
hide_directive-->
