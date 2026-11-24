<!--hide_directive
```{eval-rst}
.. meta::
   :description: Solutions to common setup and run problems of the Screening Tools sample application.
   :keywords: troubleshooting, proxy, pip, OpenVINO GPU, model manifest, port in use, PowerShell execution policy
```
hide_directive-->

# Troubleshooting

## Setup

### "running scripts is disabled on this system"

Allow scripts for the current PowerShell window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### "could not create the virtual environment"

Setup uses the Python launcher with Python 3.12 (`py -3.12`). Check which
versions are installed with `py --list`, then pick one (3.11 or newer):

```powershell
.\setup.ps1 -PythonVersion 3.11
.\setup.ps1 -Python C:\Path\To\python.exe
```

### `pip install` fails or times out

Behind a corporate proxy, pass it to setup:

```powershell
.\setup.ps1 -Proxy http://<proxy-host>:<port>
```

Without internet access, see the offline steps in
[Get Started](./get-started.md#set-up).

## Starting a run

### "OpenVINO device 'GPU' is not available"

The application does not switch to the CPU on its own, so a run never reports
CPU results as iGPU results.

1. Check that the iGPU appears in **Task Manager → Performance**.
2. Install the latest Intel graphics driver and restart.
3. Check what OpenVINO™ finds:

   ```powershell
   .\venv\Scripts\python.exe -c "import openvino as ov; print(ov.Core().available_devices)"
   ```

4. To run on the CPU instead, use `.\run.ps1 -Device CPU` or set
   `hit_scoring.device = "CPU"`.

### "no models in …" or "missing model file …"

Build the models:

```powershell
.\venv\Scripts\python.exe -m screening_tools setup
```

### "… does not match the manifest written by setup; refusing to load it"

A model file was changed or replaced after setup. Run setup again. Only use
model files that you built yourself: the anomaly model is a Python pickle and
can run code when it is loaded.

### "models were built with feature version …" or "simulator.descriptors differs …"

The feature definitions or the descriptor count changed since setup. Run setup
again, with the same `--config` you run with.

### "configuration error: unknown configuration key" or "… must be int, got float"

The TOML file has a typo or a wrong type. Key names and types must match
`configs/default.toml`. A whole number such as `400` is accepted where a decimal
is expected, but `400.0` is refused where a whole number is expected.

### "… already exists; choose another --run-id"

Runs never overwrite earlier results. Use a different `--run-id`, or leave it
out to get `run-<UTC time>`.

### "cannot open the dashboard on port 8765"

Another program, or another run, is using the port. Close it, or pick another
port:

```powershell
.\run.ps1 -- run --port 8770
```

## Dashboard

### The browser does not open

Open `http://127.0.0.1:8765/` yourself. The address is printed in the
console.

### The dashboard cannot be reached from another computer

The dashboard listens on `127.0.0.1` only and has no login, so it cannot be
reached from the network. To view it from another computer, use an SSH
tunnel:

```text
ssh -L 8765:127.0.0.1:8765 <user>@<windows-host>
```

Then open `http://127.0.0.1:8765/` on the other computer.

### The console does not return after the run

With the dashboard, the application stays open after the run so you can review
the results. Press **Ctrl+C** to exit. Use `-Headless` to exit right after the
run.

### iGPU utilisation shows "not available"

iGPU utilisation is read from the Windows *GPU Engine* performance counters.
They are not available on other operating systems, and can be missing if the
counters are disabled. Rebuild them from an Administrator PowerShell with
`lodctr /R`, then restart. Hit scoring still runs on the iGPU; only the
measurement is missing.

## Results

### Throughput is lower than in [Benchmarking](./benchmarking.md)

- Close other programs that use the CPU or the GPU.
- Use the **Best performance** power mode and keep the laptop on mains power.
- With `--rate`, the source is paced on purpose. `rate` counts every well
  including controls, while throughput counts sample wells only.
- The first `telemetry.warmup_plates` plates are left out of throughput. Very
  short runs may not have enough plates for a steady-state value.

### Results differ slightly from the documentation

The models are trained on each machine during setup. Different CPUs and
library builds give slightly different weights, so scores, hit lists and event
counts can differ a little. On one machine, the same seed and settings always
give the same results.
