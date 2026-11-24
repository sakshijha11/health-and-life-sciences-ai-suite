# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Setup step: build the models locally (shared by ITEP-97834 and ITEP-97836).

1. Generate a clean training dataset from the simulator with a fixed seed.
2. Split it by plate into training, validation and test sets.
3. Compute features with the shared feature library.
4. Train the anomaly model (Isolation Forest) and the hit model (MLP).
5. Evaluate both on the test set.
6. Rebuild the hit model as an OpenVINO graph and check it matches scikit-learn.
7. Write the models and a manifest to ``models/``.
"""

from __future__ import annotations

import copy
import platform
import time
import warnings
from datetime import datetime, timezone

import numpy as np

from screening_tools import __version__
from screening_tools.config import without_faults
from screening_tools.features import ANOMALY_FEATURES, FEATURE_VERSION, Standardizer, assemble, compute, hit_feature_names
from screening_tools.model_store import ANOMALY_MODEL, ANOMALY_SCALER, HIT_SCALER, HIT_XML, ModelStore
from screening_tools.simulator import FAULT_WELL_ARTIFACT, LIBRARY_SEED, STREAM_TRAINING, Simulator


def _dataset(cfg: dict, n_plates: int, stream: int) -> dict:
    sim = Simulator(cfg, stream=stream, plates=n_plates)
    n_desc = cfg["simulator"]["descriptors"]
    hit_x, anom_x, y_act, y_nov, plate_of, artifact = [], [], [], [], [], []
    for plate in sim:
        pf = compute(assemble(plate.context, plate.samples, plate.telemetry, n_desc), cfg)
        truth = plate.truth["samples"]
        hit_x.append(pf.hit_inputs())
        anom_x.append(pf.anomaly_inputs())
        y_act.append([t["is_active"] for t in truth])
        y_nov.append([t["is_novel"] for t in truth])
        artifact.append([t["fault"] == FAULT_WELL_ARTIFACT for t in truth])
        plate_of.append(np.full(len(truth), plate.context.plate_index))
    return {
        "hit_x": np.vstack(hit_x),
        "anom_x": np.vstack(anom_x),
        "y": np.column_stack([np.concatenate(y_act), np.concatenate(y_nov)]).astype(np.int64),
        "plate": np.concatenate(plate_of),
        "artifact": np.concatenate(artifact),
    }


def _precision_recall(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    tp = int((pred & truth).sum())
    precision = tp / int(pred.sum()) if pred.any() else 0.0
    recall = tp / int(truth.sum()) if truth.any() else 0.0
    return precision, recall


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def train_hit_model(cfg: dict, train: dict, val: dict, log=print):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier

    m = cfg["model"]
    scaler = Standardizer.fit(train["hit_x"])
    xt, xv = scaler.transform(train["hit_x"]), scaler.transform(val["hit_x"])
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(m["hidden_layers"]),
        activation="relu",
        solver="adam",
        learning_rate_init=m["learning_rate"],
        batch_size=m["batch_size"],
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=cfg["run"]["seed"],
    )
    best = (float("inf"), None, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for epoch in range(m["epochs"]):
            clf.fit(xt, train["y"])
            loss = _log_loss(clf.predict_proba(xv), val["y"])
            if loss < best[0]:
                best = (loss, ([c.copy() for c in clf.coefs_], [b.copy() for b in clf.intercepts_]), epoch + 1)
            log(f"  epoch {epoch + 1:3d}/{m['epochs']}  train loss {clf.loss_:.4f}  val loss {loss:.4f}")
    clf.coefs_, clf.intercepts_ = best[1]
    log(f"  best epoch {best[2]} (val loss {best[0]:.4f})")
    return clf, scaler, {"best_epoch": best[2], "val_log_loss": round(best[0], 5)}


def evaluate_hit(clf, scaler: Standardizer, data: dict, threshold: float) -> dict:
    from sklearn.metrics import roc_auc_score

    p = clf.predict_proba(scaler.transform(data["hit_x"]))
    act, nov = data["y"][:, 0].astype(bool), data["y"][:, 1].astype(bool)
    precision, recall = _precision_recall(p[:, 0] >= threshold, act)
    out = {
        "activity_auc": round(float(roc_auc_score(act, p[:, 0])), 4),
        "activity_precision": round(precision, 4),
        "activity_recall": round(recall, 4),
        "samples": int(act.size),
        "actives": int(act.sum()),
    }
    if act.sum() and nov[act].any() and not nov[act].all():
        out["novelty_auc_among_actives"] = round(float(roc_auc_score(nov[act], p[act, 1])), 4)
    return out


def train_anomaly_model(cfg: dict, train: dict):
    """Fit the Isolation Forest on clean wells.

    Actives are rare (a few percent), so on their own they look unusual and the
    forest would flag real hits. They are repeated until they make up about a
    third of the training set: a high activity with clean measurements is then
    normal, and only unusual measurement conditions score as anomalies.
    """
    from sklearn.ensemble import IsolationForest

    x = train["anom_x"]
    active = train["y"][:, 0].astype(bool)
    if active.any():
        reps = max(1, int(round(0.5 * (~active).sum() / active.sum())))
        x = np.vstack([x[~active], np.repeat(x[active], reps, axis=0)])
    scaler = Standardizer.fit(x)
    model = IsolationForest(n_estimators=cfg["model"]["anomaly_trees"], random_state=cfg["run"]["seed"], n_jobs=1)
    model.fit(scaler.transform(x))
    return model, scaler


def evaluate_anomaly(model, scaler: Standardizer, clean: dict, faulty: dict, threshold: float) -> dict:
    s_clean = -model.score_samples(scaler.transform(clean["anom_x"]))
    s_faulty = -model.score_samples(scaler.transform(faulty["anom_x"]))
    art = faulty["artifact"]
    act = clean["y"][:, 0].astype(bool)
    return {
        "threshold": threshold,
        "clean_false_alarm_rate": round(float((s_clean > threshold).mean()), 5),
        "clean_active_false_alarm_rate": round(float((s_clean[act] > threshold).mean()), 5) if act.any() else None,
        "clean_score_p50": round(float(np.percentile(s_clean, 50)), 4),
        "clean_score_p99": round(float(np.percentile(s_clean, 99)), 4),
        "clean_score_p99_9": round(float(np.percentile(s_clean, 99.9)), 4),
        "artifact_detection_rate": round(float((s_faulty[art] > threshold).mean()), 4) if art.any() else None,
        "artifacts": int(art.sum()),
    }


def check_openvino(store: ModelStore, clf, scaler: Standardizer, x: np.ndarray) -> dict:
    import openvino as ov

    core = ov.Core()
    compiled = core.compile_model(core.read_model(str(store.path(HIT_XML))), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    xs = scaler.transform(x)
    ov_out = compiled(xs)[0]
    ref = clf.predict_proba(xs)
    diff = float(np.abs(ov_out - ref).max())
    if diff > 1e-4:
        raise RuntimeError(f"OpenVINO output differs from scikit-learn by {diff:.2e}")
    return {"max_abs_diff_cpu_fp32": diff, "checked_samples": int(x.shape[0])}


def run_setup(cfg: dict, store: ModelStore, log=print) -> dict:
    import joblib
    import openvino as ov
    import sklearn

    from screening_tools.hit_scoring import build_ov_model

    t0 = time.perf_counter()
    m = cfg["model"]
    clean_cfg = without_faults(cfg)
    n_train, n_val, n_test = m["train_plates"], m["val_plates"], m["test_plates"]
    log(f"Generating {n_train + n_val + n_test} clean training plates (seed {cfg['run']['seed']}) ...")
    data = _dataset(clean_cfg, n_train + n_val + n_test, STREAM_TRAINING)
    split = {
        "train": data["plate"] < n_train,
        "val": (data["plate"] >= n_train) & (data["plate"] < n_train + n_val),
        "test": data["plate"] >= n_train + n_val,
    }
    parts = {k: {f: v[mask] for f, v in data.items()} for k, mask in split.items()}

    log("Training hit model (MLP, activity + novelty heads) ...")
    clf, hit_scaler, train_info = train_hit_model(cfg, parts["train"], parts["val"], log)
    hit_metrics = evaluate_hit(clf, hit_scaler, parts["test"], cfg["hit_scoring"]["hit_threshold"])
    log(f"  test: {hit_metrics}")

    log("Training anomaly model (Isolation Forest) ...")
    iforest, anomaly_scaler = train_anomaly_model(cfg, parts["train"])
    faulty_cfg = copy.deepcopy(cfg)
    faulty = _dataset(faulty_cfg, n_test, STREAM_TRAINING + 100)
    anomaly_metrics = evaluate_anomaly(iforest, anomaly_scaler, parts["test"], faulty, cfg["qc"]["anomaly_threshold"])
    log(f"  test: {anomaly_metrics}")

    store.directory.mkdir(parents=True, exist_ok=True)
    ov.save_model(build_ov_model(clf.coefs_, clf.intercepts_), str(store.path(HIT_XML)), compress_to_fp16=False)
    hit_scaler.save(store.path(HIT_SCALER))
    anomaly_scaler.save(store.path(ANOMALY_SCALER))
    joblib.dump(iforest, store.path(ANOMALY_MODEL))
    ov_check = check_openvino(store, clf, hit_scaler, parts["test"]["hit_x"][:2048])
    log(f"OpenVINO check: {ov_check}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "app_version": __version__,
        "seed": cfg["run"]["seed"],
        "library_seed": LIBRARY_SEED,
        "descriptors": cfg["simulator"]["descriptors"],
        "plates": {"train": n_train, "val": n_val, "test": n_test},
        "hit_model": {
            "inputs": hit_feature_names(cfg["simulator"]["descriptors"]),
            "outputs": ["activity_probability", "novelty_probability"],
            "hidden_layers": list(m["hidden_layers"]),
            "parameters": int(sum(c.size for c in clf.coefs_) + sum(b.size for b in clf.intercepts_)),
            "training": train_info,
            "test_metrics": hit_metrics,
            "openvino_check": ov_check,
        },
        "anomaly_model": {"inputs": list(ANOMALY_FEATURES), "trees": m["anomaly_trees"], "test_metrics": anomaly_metrics},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "openvino": ov.get_version(),
        },
        "feature_version_used": FEATURE_VERSION,
        "setup_seconds": round(time.perf_counter() - t0, 1),
    }
    store.write_manifest(manifest)
    log(f"Models written to {store.directory} in {manifest['setup_seconds']} s")
    return manifest
