# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line interface: ``python -m screening_tools <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from screening_tools import __version__
from screening_tools.config import APP_ROOT, ConfigError, load_config, without_faults

EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2


def _out_root(cfg: dict) -> Path:
    p = Path(cfg["run"]["output_dir"])
    return p if p.is_absolute() else APP_ROOT / p


def _config(args) -> dict:
    overrides: dict = {}
    if getattr(args, "plates", None) is not None:
        overrides.setdefault("run", {})["plates"] = args.plates
    if getattr(args, "seed", None) is not None:
        overrides.setdefault("run", {})["seed"] = args.seed
    if getattr(args, "run_id", None):
        overrides.setdefault("run", {})["run_id"] = args.run_id
    if getattr(args, "device", None):
        overrides.setdefault("hit_scoring", {})["device"] = args.device
    if getattr(args, "batch_size", None) is not None:
        overrides.setdefault("hit_scoring", {})["batch_size"] = args.batch_size
    if getattr(args, "rate", None) is not None:
        overrides.setdefault("ingestion", {})["rate"] = args.rate
    if getattr(args, "port", None) is not None:
        overrides.setdefault("dashboard", {})["port"] = args.port
    cfg = load_config(args.config, overrides)
    if getattr(args, "clean", False):
        cfg = without_faults(cfg)
    if not cfg["run"]["run_id"]:
        cfg["run"]["run_id"] = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return cfg


def cmd_setup(args) -> int:
    from screening_tools.model_store import ModelStore
    from screening_tools.setup_models import run_setup

    cfg = _config(args)
    manifest = run_setup(cfg, ModelStore())
    print(json.dumps({"hit_model": manifest["hit_model"]["test_metrics"], "anomaly_model": manifest["anomaly_model"]["test_metrics"]}, indent=2))
    return EXIT_OK


def cmd_simulate(args) -> int:
    from screening_tools.ingestion import write_run_file

    cfg = _config(args)
    out = Path(args.out) if args.out else _out_root(cfg) / "recordings" / f"{cfg['run']['run_id']}.jsonl"
    truth = out.with_name(out.stem + ".truth.jsonl")
    n = write_run_file(cfg, out, truth)
    print(f"wrote {n:,} records for {cfg['run']['plates']} plates to {out}")
    print(f"ground truth (demo-mode evaluation only): {truth}")
    return EXIT_OK


def _print_summary(summary_path: Path) -> None:
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    m = s["metrics"]
    print()
    print(f"Run {s['run_id']} {s['status']}: {m['plates']} plates, {m['samples']:,} samples in {m['wall_seconds']:.1f} s")
    print(f"  devices: hit scoring on {s['devices']['hit_scoring']}; features, auto-QC and fusion on CPU")
    tp = m["throughput_samples_per_s"]
    print(f"  steady-state throughput: {tp:,.0f} samples/s" if tp else "  steady-state throughput: not enough plates after warm-up")
    for stage, p in m["latency"].items():
        if p["count"]:
            print(f"  latency per plate {stage:12s} P50 {p['p50_ms']:8.2f} ms  P95 {p['p95_ms']:8.2f} ms")
    u = m["utilisation"]
    if u["cpu_percent"]:
        print(f"  CPU utilisation avg {u['cpu_percent']['avg']}% peak {u['cpu_percent']['peak']}%")
    print(f"  iGPU utilisation avg {u['gpu_percent']['avg']}% peak {u['gpu_percent']['peak']}%" if u["gpu_percent"] else f"  iGPU utilisation: {u['gpu_note']}")
    print(f"  backpressure events: {m['backpressure_events']} (buffer peak {m['buffer_peak_depth']}/{m['buffer_size']})")
    d = s["dispositions"]
    print(f"  candidates: {d['candidate']}, QC-flagged candidates: {d['candidate_qc_flagged']}")
    print(f"  QC events: {sum(s['qc_event_counts'].values())} " + json.dumps(s["qc_event_counts"]))
    e = s["evaluation"]
    if e and e.get("plates_evaluated"):
        print(f"  demo-mode ground truth: tier 1 precision {e['tier1']['precision']} recall {e['tier1']['recall']}; tier 1+2 recall {e['tier1_and_2']['recall']}")
        for name, f in e["faults"].items():
            print(f"    {name}: {json.dumps(f)}")
        print(f"    events on {e['plates_without_fault']} fault-free plates: {json.dumps(e['events_on_plates_without_fault'])}")
    print(f"  outputs: {summary_path.parent}")


def cmd_run(args) -> int:
    from screening_tools.export import write_summary
    from screening_tools.ingestion import ReplaySource, SimulatorSource
    from screening_tools.model_store import ModelStore
    from screening_tools.pipeline import Pipeline, PipelineError

    cfg = _config(args)
    out_dir = _out_root(cfg) / cfg["run"]["run_id"]
    if out_dir.exists():
        print(f"error: {out_dir} already exists; choose another --run-id", file=sys.stderr)
        return EXIT_USAGE
    if args.replay:
        source = ReplaySource(Path(args.replay))
        truth = Path(args.truth) if args.truth else None
        mode = "replay"
    else:
        truth = out_dir / "ground_truth.jsonl"
        source = SimulatorSource(cfg, truth)
        mode = "simulator"
    pipe = Pipeline(cfg, source, ModelStore(), out_dir, truth)
    server = None
    if not args.headless:
        from screening_tools.dashboard.server import DashboardServer

        port = cfg["dashboard"]["port"]
        try:
            server = DashboardServer(pipe, "127.0.0.1", port, cfg["dashboard"]["refresh_ms"])
        except OSError as exc:
            print(f"error: cannot open the dashboard on port {port} ({exc.strerror}). Use --port <n> or close the other program.", file=sys.stderr)
            return EXIT_ERROR
        server.start()
        print(f"dashboard: {server.url}")
        if not args.no_browser:
            webbrowser.open(server.url)
    print(f"run {cfg['run']['run_id']}: {cfg['run']['plates'] if mode == 'simulator' else 'replayed'} plates, hit scoring on {pipe.scorer.device} ({pipe.scorer.device_name})")
    t0 = time.perf_counter()
    status = EXIT_OK
    try:
        pipe.run()
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        status = EXIT_ERROR
    path = write_summary(pipe, time.perf_counter() - t0, mode)
    _print_summary(path)
    if server is not None:
        try:
            print("\nThe dashboard stays open for review. Press Ctrl+C to exit.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
    return status


def cmd_benchmark(args) -> int:
    from screening_tools.benchmark import run_benchmark, write_benchmark
    from screening_tools.model_store import ModelStore

    cfg = _config(args)
    devices = [d.strip().upper() for d in args.devices.split(",") if d.strip()]
    sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    print(f"benchmarking hit scoring on {', '.join(devices)}; batch sizes {sizes}; {args.seconds} s per point")
    result = run_benchmark(cfg, ModelStore(), devices, sizes, args.seconds, args.bench_plates, args.top_n)
    for dev, why in result["skipped_devices"].items():
        print(f"  skipped {dev}: {why}")
    if result["ranking_agreement"]:
        for k, v in result["ranking_agreement"].items():
            print(f"  ranking agreement {k}: mean top-{v['top_n']} overlap {v['mean_top_n_overlap']}, min {v['min_top_n_overlap']}, max |score difference| {v['max_abs_score_difference']}")
    path = write_benchmark(result, _out_root(cfg) / "benchmarks")
    print(f"wrote {path}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m screening_tools", description="Screening Tools: data management and operational efficiency for screening assays.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--config", help="TOML file merged over configs/default.toml")
        sp.add_argument("--seed", type=int, help="override run.seed")

    sp = sub.add_parser("setup", help="build and check the models (run once)")
    common(sp)
    sp.set_defaults(func=cmd_setup)

    sp = sub.add_parser("simulate", help="write a simulated run to a file for replay")
    common(sp)
    sp.add_argument("--plates", type=int)
    sp.add_argument("--run-id")
    sp.add_argument("--clean", action="store_true", help="switch off every fault scenario")
    sp.add_argument("--out", help="output file (default runs/recordings/<run-id>.jsonl)")
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("run", help="run the pipeline with the dashboard (or --headless)")
    common(sp)
    sp.add_argument("--plates", type=int)
    sp.add_argument("--run-id")
    sp.add_argument("--clean", action="store_true", help="switch off every fault scenario")
    sp.add_argument("--device", choices=["CPU", "GPU"], help="hit-scoring device")
    sp.add_argument("--batch-size", type=int)
    sp.add_argument("--rate", type=float, help="wells per second read from the source, controls included (0 = as fast as possible)")
    sp.add_argument("--replay", help="replay a file written by 'simulate' instead of the live simulator")
    sp.add_argument("--truth", help="ground-truth file for a replay (enables demo-mode metrics)")
    sp.add_argument("--headless", action="store_true", help="no dashboard; print the summary and exit")
    sp.add_argument("--port", type=int)
    sp.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("benchmark", help="compare hit scoring on CPU and iGPU")
    common(sp)
    sp.add_argument("--devices", default="CPU,GPU")
    sp.add_argument("--batch-sizes", default="1,16,64,256,1024")
    sp.add_argument("--seconds", type=float, default=3.0, help="measurement time per point")
    sp.add_argument("--bench-plates", type=int, default=50, help="plates scored for the ranking check")
    sp.add_argument("--top-n", type=int, default=10)
    sp.set_defaults(func=cmd_benchmark)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # noqa: BLE001 - one clear line instead of a traceback for known errors
        from screening_tools.hit_scoring import DeviceUnavailable
        from screening_tools.model_store import ModelError

        if isinstance(exc, (DeviceUnavailable, ModelError, FileNotFoundError)):
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR
        raise
