<!--hide_directive
```{eval-rst}
.. meta::
   :description: Step-by-step instructions to install and run the Screening Tools sample application on Windows with an Intel® Core™ Ultra processor.
   :keywords: get started, setup, Windows, Intel Core Ultra, OpenVINO, screening assay, dashboard
```
hide_directive-->

# Get Started

This guide takes you from a fresh clone to a running dashboard on a Windows PC
with an Intel® Core™ Ultra processor.

> [!IMPORTANT]
> Run every command in **Windows PowerShell 5.1** or **PowerShell 7**, not in
> Command Prompt (CMD). A regular (non-Administrator) window is enough.

## Prerequisites

- Windows 11 with an Intel® Core™ Ultra processor and its integrated GPU.
- An up-to-date Intel graphics driver. The iGPU must be visible to OpenVINO™
  (setup lists the devices it finds).
- Python 3.11 or newer, 64-bit. By default setup uses the Python launcher with
  Python 3.12 (`py -3.12`).
- [Git for Windows](https://git-scm.com/download/win).
- Internet access for `pip`, directly or through a proxy.

If running a script fails with "running scripts is disabled on this system",
allow scripts for the current window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Clone the repository

The `--sparse` option downloads only this application:

```powershell
git clone --filter=blob:none --sparse --branch main https://github.com/open-edge-platform/health-and-life-sciences-ai-suite.git
cd health-and-life-sciences-ai-suite
git sparse-checkout set screening-tools
cd screening-tools
```

## Set up

```powershell
.\setup.ps1
```

Setup does the following:

1. Creates a virtual environment in `venv\`.
2. Installs the packages in `requirements.txt`: OpenVINO™, NumPy, scikit-learn
   and psutil.
3. Lists the OpenVINO™ devices it finds, for example:

   ```text
   CPU - Intel(R) Core(TM) Ultra 7 356H
   GPU - Intel(R) Graphics (iGPU)
   ```

4. Builds the models (about one to two minutes). It simulates training,
   validation and test plates, trains the hit and anomaly models, converts the
   hit model to OpenVINO™, checks that OpenVINO™ gives the same scores as
   scikit-learn, and writes `models\manifest.json` with test metrics and a
   SHA-256 hash of every model file. No model is downloaded.

Setup options:

| Option | Use |
| --- | --- |
| `-Proxy http://<host>:<port>` | Use a proxy for `pip` |
| `-Python <path>` | Use this Python executable instead of the `py` launcher |
| `-PythonVersion 3.11` | Python version for the `py` launcher (default `3.12`) |
| `-Venv <folder>` | Virtual environment folder (default `venv`) |
| `-Dev` | Also install the test tools (`requirements-dev.txt`) |
| `-SkipInstall` | Do not run `pip` |
| `-SkipModels` | Do not build the models |

Offline, download the wheels on another machine and install from that folder:

```powershell
.\venv\Scripts\python.exe -m pip install --no-index --find-links <wheels_folder> -r requirements.txt
.\setup.ps1 -SkipInstall
```

## Run with the dashboard

```powershell
.\run.ps1
```

The browser opens `http://127.0.0.1:8765/`. The default run simulates 60 plates
with all fault scenarios switched on. When the run ends, the console prints a
summary and the dashboard stays open so you can review it. Press **Ctrl+C** in
the console to exit.

The dashboard shows:

- **KPIs**: plates, samples processed, throughput, end-to-end latency P95,
  candidates, QC-flagged candidates, QC events and elapsed time.
- **Pipeline**: queue depth and P50/P95 latency of every stage, and the device
  each stage runs on.
- **Hit list**: the current run ranking, with QC status, reason codes, hit,
  novelty and anomaly scores.
- **Events**: the newest reason-coded QC, anomaly and drift events.
- **Control trend**: control levels per plate with EWMA limits and reagent-lot
  changes.
- **Last plate**: a heat map of the last plate with candidates, QC-flagged
  candidates and failed samples.
- **Throughput** and **utilisation**: samples per second, CPU and iGPU busy
  percentage.
- **Plates**: plate status and Z'-factor per plate.
- **Demo-mode evaluation**: results compared with the simulator's ground truth.

The **Stop** button ends the run early. The results written so far are kept.
**Download summary** becomes available when the run ends and returns
`summary.json`.

## Run without the dashboard

```powershell
.\run.ps1 -Headless -Plates 600
```

Example output (shortened):

```text
Run run-20261124T101908Z completed: 600 plates, 48,000 samples in 7.6 s
  devices: hit scoring on GPU (Intel(R) Graphics (iGPU), f16); features, auto-QC and fusion on CPU
  steady-state throughput: 6,309 samples/s
  latency per plate auto_qc      P50    14.39 ms ...
  iGPU utilisation avg 20.63% peak 31.29%
  candidates: ..., QC-flagged candidates: ...
  demo-mode ground truth: tier 1 precision 0.972 recall 0.983; tier 1+2 recall 0.997
  outputs: ...\runs\run-20261124T101908Z
```

## Other run options

```powershell
.\run.ps1 -Device CPU              # hit scoring on the CPU instead of the iGPU
.\run.ps1 -Clean                   # switch off every fault scenario
.\run.ps1 -Config my.toml          # your settings, merged over configs\default.toml
.\run.ps1 -Benchmark               # CPU vs iGPU hit scoring, see Benchmarking
```

Any command-line arguments can be passed after `--`:

```powershell
.\run.ps1 -- simulate --plates 100 --run-id demo
.\run.ps1 -- run --replay runs\recordings\demo.jsonl --truth runs\recordings\demo.truth.jsonl
```

See [Runtime Configuration](./runtime-configuration.md) for every option.

## Run outputs

Each run writes one folder, `runs\<run-id>\`:

| File | Content |
| --- | --- |
| `hits_run.csv` | Final run hit list, ranked (`rank_in_run`) |
| `hits_by_plate.csv` | Hit list of every plate, ranked within the plate (`rank_in_plate`) |
| `results.csv` | Every sample well with QC status, reason codes, scores and disposition |
| `events.csv` | Every reason-coded QC, anomaly and drift event |
| `summary.json` | Settings, devices, hardware and software versions, throughput, latency, utilisation, event counts and, for simulated runs, the ground-truth evaluation |
| `ground_truth.jsonl` | Simulator labels (simulated runs only). The pipeline never reads it; it is used only for the demo-mode evaluation |

A run never overwrites an existing folder. Use a new `--run-id`, or leave it
empty to get `run-<UTC time>`.

## Run the tests

```powershell
.\setup.ps1 -Dev -SkipModels
.\venv\Scripts\python.exe -m pytest -q
```

The tests build small models of their own and do not change `models\`.
