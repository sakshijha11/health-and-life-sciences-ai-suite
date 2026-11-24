# Screening Tools Sample Application

> [!NOTE]
> This application is for **reference and evaluation purposes only**. It uses
> **simulated** assay data. It is **not intended for direct use in laboratory,
> clinical or diagnostic decisions** and is not validated for such a purpose.

Screening Tools shows how one Intel® Core™ Ultra PC can watch a high-throughput
screening assay while it runs. Plate readers produce one value per well. The
application checks the quality of every plate and every sample on the CPU,
scores every sample for activity on the Intel® iGPU with the OpenVINO™ toolkit,
and joins both results into a ranked hit list per plate and per run. Every QC
decision carries a reason code that says which rule fired and by how much.

| Feature | Description |
| --- | --- |
| **Data source** | Built-in plate simulator with labelled fault scenarios, or replay of a recorded run |
| **Auto-QC (CPU)** | Plate rules (Z'-factor, control range, control CV, signal-to-background, temperature), sample rules (QC metric, saturation) and an Isolation Forest anomaly score |
| **Drift (CPU)** | Rule-based EWMA and CUSUM on control levels, plus a reagent-lot shift check |
| **Hit scoring (iGPU)** | Multilayer perceptron (activity + novelty) run with OpenVINO™ on `GPU` or `CPU`; no silent fallback |
| **Decision fusion** | Hits with a QC failure are kept but ranked lower and flagged *"high hit score but QC failure, possible false positive"* |
| **Outputs** | Hit list per plate and per run, per-sample results, reason-coded events, run summary (CSV and JSON) |
| **Dashboard** | Local web page: KPIs, pipeline stages, hit list, events, control trend, plate heat map, throughput, CPU/iGPU utilisation |
| **Runtime** | Native Windows Python application (PowerShell + venv). Dependencies: OpenVINO™, NumPy, scikit-learn, psutil |

## Topology

```text
simulator / replay file
        |  plate_start, sample (one value per well), telemetry, plate_end
        v
  bounded buffer (back-pressure)
        |
        v
  feature engine (CPU) ── per-plate normalisation to the plate's own controls
        |
        +──────────────────────────┐
        v                          v
  auto-QC + drift (CPU)       hit scoring (iGPU, OpenVINO)
  reason-coded events         activity + novelty per sample
        |                          |
        └──────────► decision fusion ◄┘
                          |
                          v
     hits_by_plate.csv, hits_run.csv, results.csv, events.csv, summary.json
                          |
                          v
                 dashboard (http://127.0.0.1:8765/)
```

Every stage runs in its own thread and is connected by bounded queues, so a slow
stage slows the source instead of using unbounded memory.

## Quick start (Windows, PowerShell)

```powershell
git clone --filter=blob:none --sparse --branch main https://github.com/open-edge-platform/health-and-life-sciences-ai-suite.git
cd health-and-life-sciences-ai-suite
git sparse-checkout set screening-tools
cd screening-tools

.\setup.ps1                 # venv, dependencies, model build (about 1-2 minutes)
.\run.ps1                   # run with the dashboard
```

Behind a proxy, use `.\setup.ps1 -Proxy http://<host>:<port>`. For other
options, see [Get Started](./docs/user-guide/get-started.md).

```powershell
.\run.ps1 -Headless -Plates 600   # no dashboard; prints a summary
.\run.ps1 -Device CPU             # hit scoring on the CPU
.\run.ps1 -Clean                  # switch off all fault scenarios
.\run.ps1 -Benchmark              # compare hit scoring on CPU and iGPU
```

## Measured results

Intel® Core™ Ultra 7 356H, Intel® Graphics (iGPU), Windows 11, OpenVINO™
2026.4.1, Python 3.12.10. The models are trained on each machine during setup,
so numbers on another machine can differ slightly.

| Measurement | Result |
| --- | --- |
| Setup (dependencies already installed) | 77 s |
| 600-plate run with all faults, hit scoring on iGPU | 6,309 samples/s |
| 600-plate run with all faults, hit scoring on CPU | 6,597 samples/s |
| CPU / iGPU utilisation during the iGPU run | 7.5% average / 20.6% average, 31.3% peak |
| Tier 1 hit list precision / recall (faults on) | 0.972 / 0.983 |
| Tier 1 + tier 2 recall (faults on) | 0.997 |
| Fault detection | bad plates 2/2, temperature excursion 4/4, reagent lot shift detected, slow drift detected 6 plates after onset, 89% of well artifacts flagged |
| Hit scoring alone, batch 1024 | CPU 592k samples/s, iGPU 4.84M samples/s |

The full pipeline is limited by the auto-QC stage on the CPU, not by hit
scoring. See [Benchmarking](./docs/user-guide/benchmarking.md).

## Documentation

- [Overview](./docs/user-guide/index.md)
- [Get Started](./docs/user-guide/get-started.md)
- [Application Flow](./docs/user-guide/application-flow.md)
- [QC Rules and Reason Codes](./docs/user-guide/qc-rules.md)
- [Runtime Configuration](./docs/user-guide/runtime-configuration.md)
- [Benchmarking](./docs/user-guide/benchmarking.md)
- [Troubleshooting](./docs/user-guide/troubleshooting.md)
- [Release Notes](./docs/user-guide/release-notes.md)
- [Uninstall](./docs/user-guide/uninstall.md)

## Folder layout

```text
screening-tools/
├── configs/default.toml     every threshold, the seed, device and batch size
├── models/                  built by setup (not committed)
├── runs/                    one folder per run (not committed)
├── screening_tools/         application package
│   └── dashboard/           local web dashboard
├── tests/                   pytest suite
├── setup.ps1, run.ps1       Windows entry points
└── requirements*.txt
```

## Tests

```powershell
.\setup.ps1 -Dev -SkipModels
.\venv\Scripts\python.exe -m pytest -q
```
