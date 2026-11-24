<!--hide_directive
```{eval-rst}
.. meta::
   :description: Configuration file keys, command-line options and PowerShell script parameters of the Screening Tools sample application.
   :keywords: runtime configuration, TOML, command line, PowerShell, OpenVINO device, batch size, QC thresholds
```
hide_directive-->

# Runtime Configuration

Settings come from three places, later ones win:

1. `configs/default.toml`: every setting with its default.
2. Your own TOML file, passed with `--config my.toml` (or `.\run.ps1 -Config my.toml`).
   It only needs the keys you want to change.
3. Command-line options such as `--plates` or `--device`.

Unknown keys and wrong types are refused with a clear error, so a typo cannot
be silently ignored. A whole number is accepted where a decimal is expected,
but not the other way round.

Example `my.toml`:

```toml
[run]
plates = 200

[plate]
rows = 16      # 384-well plate
cols = 24
positive_control_column = 24

[qc]
zprime_warn = 0.6
anomaly_threshold = 0.65

[hit_scoring]
device = "CPU"
```

> [!NOTE]
> Changing `[simulator]`, `[plate]` or `[model]` settings changes the data the
> models were trained on. Run `python -m screening_tools setup --config my.toml`
> again after such changes. The application refuses models built with a
> different number of descriptors.

## Configuration keys

### `[run]`

| Key | Default | Meaning |
| --- | --- | --- |
| `seed` | `2027` | One seed for the simulator and model training. Same seed and settings give the same results |
| `plates` | `60` | Plates per run (at least 1) |
| `run_id` | `""` | Run folder name. Empty gives `run-<UTC time>`. Letters, digits, `.`, `_`, `-`, up to 64 characters |
| `output_dir` | `"runs"` | Output folder, relative to the application folder unless absolute |

### `[plate]`

| Key | Default | Meaning |
| --- | --- | --- |
| `rows`, `cols` | `8`, `12` | Plate format (96 wells). Use `16`, `24` for 384 wells |
| `negative_control_column` | `1` | Column with negative controls (1-based) |
| `positive_control_column` | `12` | Column with positive controls (1-based, different from the negative column) |

### `[simulator]`

| Key | Default | Meaning |
| --- | --- | --- |
| `negative_level`, `positive_level` | `100.0`, `400.0` | Mean raw signal of the controls |
| `negative_noise`, `positive_noise` | `12.0`, `18.0` | Standard deviation of negative and positive wells |
| `active_rate` | `0.03` | Share of sample wells that are truly active |
| `edge_effect` | `0.06` | Relative signal increase on outer wells |
| `temperature_setpoint`, `temperature_noise` | `37.0`, `0.15` | Instrument temperature, °C |
| `qc_metric_mean` | `0.06` | Mean read-quality metric (0 good, 1 bad) |
| `saturation_value` | `4000.0` | Detector maximum |
| `plate_read_seconds` | `90.0` | Simulated time to read one plate |
| `telemetry_per_plate` | `6` | Temperature readings per plate |
| `descriptors` | `8` | Compound descriptors per sample |
| `known_chemotypes`, `novel_chemotypes` | `4`, `2` | Active compound families seen in training, and families reported as novel |

### `[faults.*]`

Each fault has `enabled = true`. `--clean` switches all of them off. Plate
indices start at 0.

| Table | Keys (default) |
| --- | --- |
| `[faults.slow_drift]` | `start_plate` (10), `plates` (18), `gain_per_plate` (0.006) |
| `[faults.reagent_lot_change]` | `at_plate` (30), `positive_factor` (0.85) |
| `[faults.temperature_excursion]` | `start_plate` (40), `plates` (4), `delta_c` (3.0), `noise_factor` (1.8) |
| `[faults.bad_plate]` | `plates` ([22, 47]), `positive_factor` (0.45) |
| `[faults.well_artifacts]` | `rate` (0.006), `visible_fraction` (0.8) |

### `[ingestion]`

| Key | Default | Meaning |
| --- | --- | --- |
| `buffer_size` | `2048` | Records held between the source and the feature engine |
| `stage_queue_plates` | `4` | Plates held between later stages |
| `rate` | `0.0` | Source speed in wells per second, controls included; `0` is as fast as possible. Throughput is reported in sample wells, so a 96-well plate with 16 controls shows 80/96 of this rate |

Results do not depend on the buffer sizes or the rate.

### `[qc]` and `[drift]`

See [QC Rules and Reason Codes](./qc-rules.md) for what each key does.

| Key | Default |
| --- | --- |
| `qc.zprime_warn`, `qc.zprime_fail` | `0.5`, `0.3` |
| `qc.control_tolerance` | `0.25` |
| `qc.control_cv_warn` | `0.20` |
| `qc.signal_to_background_min` | `2.0` |
| `qc.min_controls` | `4` |
| `qc.qc_metric_fail` | `0.25` |
| `qc.anomaly_threshold` | `0.62` |
| `qc.temperature_band` | `1.0` |
| `drift.ewma_lambda`, `drift.ewma_limit_sigma` | `0.3`, `3.5` |
| `drift.cusum_k_sigma`, `drift.cusum_h_sigma` | `0.5`, `6.0` |
| `drift.baseline_plates` | `10` |
| `drift.lot_window`, `drift.lot_shift_warn` | `4`, `0.08` |

### `[hit_scoring]`

| Key | Default | Meaning |
| --- | --- | --- |
| `device` | `"GPU"` | `"GPU"` (iGPU) or `"CPU"`. No silent fallback |
| `batch_size` | `256` | Samples per inference request |
| `hit_threshold` | `0.5` | Activity score at or above this is a hit |

### `[model]` (setup only)

| Key | Default | Meaning |
| --- | --- | --- |
| `hidden_layers` | `[512, 512, 256]` | Hidden layer sizes of the hit model |
| `epochs`, `learning_rate`, `batch_size` | `30`, `0.001`, `512` | Hit model training |
| `train_plates`, `val_plates`, `test_plates` | `600`, `100`, `100` | Simulated plates for training, early stopping and test metrics |
| `anomaly_trees` | `200` | Isolation Forest trees |

### `[dashboard]` and `[telemetry]`

| Key | Default | Meaning |
| --- | --- | --- |
| `dashboard.port` | `8765` | Local port (1024–65535). The dashboard listens on `127.0.0.1` only |
| `dashboard.refresh_ms` | `1000` | Browser refresh interval |
| `telemetry.sample_interval_s` | `0.5` | CPU and iGPU sampling period |
| `telemetry.warmup_plates` | `3` | Plates left out of throughput and latency |

## Command line

Run from the application folder with the virtual environment's Python:

```powershell
.\venv\Scripts\python.exe -m screening_tools <command> [options]
```

Every command accepts `--config <file>` and `--seed <n>`.

| Command | Options |
| --- | --- |
| `setup` | Builds and checks the models. Run once, and again after changing `[simulator]`, `[plate]` or `[model]` |
| `simulate` | `--plates`, `--run-id`, `--clean`, `--out <file>` (default `runs/recordings/<run-id>.jsonl`). Also writes `<name>.truth.jsonl` |
| `run` | `--plates`, `--run-id`, `--clean`, `--device CPU\|GPU`, `--batch-size`, `--rate`, `--replay <file>`, `--truth <file>`, `--headless`, `--port`, `--no-browser` |
| `benchmark` | `--devices` (`CPU,GPU`), `--batch-sizes` (`1,16,64,256,1024`), `--seconds` (3.0 per point), `--bench-plates` (50), `--top-n` (10) |

Exit codes: `0` success, `1` run or device error, `2` configuration or usage
error.

## PowerShell scripts

`setup.ps1`: `-Proxy`, `-Python`, `-PythonVersion`, `-Venv`, `-Dev`,
`-SkipInstall`, `-SkipModels`. See [Get Started](./get-started.md#set-up).

`run.ps1`:

| Parameter | Same as |
| --- | --- |
| `-Headless` | `run --headless` |
| `-Benchmark` | `benchmark` |
| `-Clean` | `run --clean` |
| `-Device CPU\|GPU` | `run --device` |
| `-Plates <n>` | `run --plates` |
| `-Config <file>` | `--config` |
| `-Venv <folder>` | Virtual environment folder (default `venv`) |
| `-- <arguments>` | Passes the arguments to `python -m screening_tools` unchanged |
