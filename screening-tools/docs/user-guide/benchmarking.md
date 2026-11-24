<!--hide_directive
```{eval-rst}
.. meta::
   :description: Measured throughput, latency, CPU and iGPU utilisation and CPU vs iGPU hit-scoring results of the Screening Tools sample application.
   :keywords: benchmarking, throughput, latency, iGPU, OpenVINO, CPU, batch size, utilisation
```
hide_directive-->

# Benchmarking

There are two measurements:

- **Full pipeline** (`run --headless`): everything from the source to the
  outputs, the way the application is used.
- **Hit scoring only** (`benchmark`): the OpenVINO™ model alone on each device
  and batch size, to compare CPU and iGPU.

## Test system

| Item | Value |
| --- | --- |
| Processor | Intel® Core™ Ultra 7 356H, 16 logical CPUs |
| GPU | Intel® Graphics (iGPU) |
| Memory | 47 GB |
| Operating system | Windows 11 (build 26300) |
| Software | Python 3.12.10, OpenVINO™ 2026.4.1, NumPy 2.5.5, scikit-learn 1.9.1 |
| Settings | `configs/default.toml`, 600 plates of 96 wells (48,000 samples), `rate = 0` (as fast as possible) |

The models are trained on each machine during setup, so results on another
machine can differ slightly.

## Full pipeline

```powershell
.\run.ps1 -Headless -Plates 600
.\run.ps1 -Headless -Plates 600 -Device CPU
.\run.ps1 -Headless -Plates 600 -Clean
```

| Measurement | iGPU, faults on | CPU, faults on | iGPU, no faults |
| --- | --- | --- | --- |
| Steady-state throughput (samples/s) | 6,309 | 6,597 | 6,364 |
| Wall time, 600 plates | 7.6 s | 7.3 s | 7.6 s |
| Hit scoring precision | f16 | f32 | f16 |
| System CPU utilisation, average / peak | 7.5% / 10.0% | 9.0% / 12.3% | 8.2% / 12.9% |
| iGPU utilisation, average / peak | 20.6% / 31.3% | 0.05% / 0.24% | 20.0% / 30.9% |

Latency per plate (80 samples), P50 / P95 in milliseconds, iGPU run with
faults on:

| Stage | Device | P50 | P95 |
| --- | --- | --- | --- |
| Features | CPU | 1.5 | 3.3 |
| Auto-QC + drift | CPU | 14.4 | 15.4 |
| Hit scoring | iGPU | 2.4 | 4.0 |
| Fusion | CPU | 0.2 | 0.4 |
| End to end | – | 365 | 372 |

End-to-end latency includes the time a plate waits in the input buffer. At
`rate = 0` the source fills the buffer as fast as it can, so this number shows
the buffer depth, not processing time. With a paced source it drops to the
processing time: with `--rate 4000` (3,325 sample wells per second) the
end-to-end P50 / P95 is 8.7 / 17.2 ms with no back-pressure.

**Dashboard refresh lag.** With the dashboard open (`--rate 500`,
`refresh_ms = 1000`), the newest result was on screen 226 ms (P50) / 313 ms
(P95) after fusion produced it. This was measured in a browser on another
computer through an SSH tunnel.

### Findings

- **Auto-QC on the CPU limits throughput.** It takes about 14 ms per plate,
  mostly the Isolation Forest, and runs in one thread. The application process
  uses about 6.5% of the 16 logical CPUs, roughly one core. Hit scoring on the
  iGPU takes about 2.4 ms per plate in parallel, so moving it from the CPU to
  the iGPU does not change the end-to-end throughput.
- **The iGPU frees the CPU.** With hit scoring on the iGPU, system CPU use is
  lower (7.5% against 9.0% average), and the iGPU stays below a third of its
  capacity.
- **Same decisions on both devices.** CPU (f32) and iGPU (f16) runs gave the
  same dispositions and the same hit lists.
- **Headroom.** 6,300 samples/s is about 79 plates of 96 wells per second, far
  above the speed of a plate reader. The `--rate` option paces the simulator
  to a realistic speed for demonstrations.

## Hit scoring only

```powershell
.\run.ps1 -Benchmark
```

Each point is measured for 3 seconds after a warm-up. Latency is per inference
request.

| Batch size | CPU P50 / P95 (ms) | CPU samples/s | iGPU P50 / P95 (ms) | iGPU samples/s |
| --- | --- | --- | --- | --- |
| 1 | 0.035 / 0.037 | 27,859 | 0.144 / 0.157 | 6,883 |
| 16 | 0.057 / 0.065 | 245,143 | 0.138 / 0.148 | 114,695 |
| 64 | 0.137 / 0.150 | 426,966 | 0.138 / 0.148 | 458,393 |
| 256 | 0.455 / 0.519 | 534,272 | 0.154 / 0.166 | 1,649,007 |
| 1024 | 1.666 / 2.028 | 592,179 | 0.209 / 0.221 | 4,837,237 |

Ranking agreement over 50 plates: the top 10 of every plate is the same on
both devices, and the largest score difference is 0.0007.

### Findings

- The CPU is faster for single samples and small batches, where the iGPU's
  fixed cost per request (about 0.14 ms) dominates.
- From a batch size of 64 the iGPU is faster. At 1024 it is about 8 times the
  CPU throughput, with a lower and steadier latency.
- The default batch size of 256 gives the iGPU about 3 times the CPU
  throughput at 0.15 ms per request.

Results are written to `runs\benchmarks\benchmark-<UTC time>.json`.

## Reproduce

1. Close other applications that use the GPU.
2. Run the commands above from the application folder.
3. Compare with `summary.json` in each run folder. It records the hardware,
   software versions and settings of the run.
