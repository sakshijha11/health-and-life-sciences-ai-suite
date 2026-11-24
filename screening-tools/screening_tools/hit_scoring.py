# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""iGPU hit scoring with OpenVINO (ITEP-97836).

The hit model is a two-headed MLP (activity, novelty). It is trained with
scikit-learn in the setup step and rebuilt here as an OpenVINO graph from
its weights, so the runtime needs no PyTorch and no conversion tool.

The device is requested explicitly. If it is not available the scorer stops
with a clear error; there is no silent fallback to the CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from screening_tools.config import APP_ROOT
from screening_tools.model_store import HIT_XML, ModelStore

OUTPUT_NAME = "probabilities"  # (N, 2): activity, novelty


class DeviceUnavailable(RuntimeError):
    pass


def build_ov_model(coefs: list[np.ndarray], intercepts: list[np.ndarray]):
    """Build an OpenVINO model equivalent to a scikit-learn MLP with ReLU hidden
    layers and a logistic (sigmoid) multi-label output."""
    import openvino as ov
    import openvino.opset13 as ops

    n_in = coefs[0].shape[0]
    x_in = ops.parameter([-1, n_in], np.float32, name="features")
    x = x_in
    last = len(coefs) - 1
    for i, (w, b) in enumerate(zip(coefs, intercepts)):
        x = ops.matmul(x, ops.constant(w.astype(np.float32)), False, False)
        x = ops.add(x, ops.constant(b.astype(np.float32)))
        x = ops.relu(x) if i < last else ops.sigmoid(x)
    x.output(0).get_tensor().set_names({OUTPUT_NAME})
    return ov.Model([x], [x_in], "hit_mlp")


def device_name(core, device: str) -> str:
    try:
        return str(core.get_property(device, "FULL_DEVICE_NAME"))
    except Exception:  # noqa: BLE001 - informational only
        return device


class HitScorer:
    """Runs the hit model on one device with a fixed batch size."""

    def __init__(self, store: ModelStore, device: str, batch_size: int, cache_dir: Path | None = APP_ROOT / "cache"):
        import openvino as ov

        core = ov.Core()
        available = core.available_devices
        if device not in available:
            raise DeviceUnavailable(f"OpenVINO device {device!r} is not available (found: {', '.join(available) or 'none'}). Check the Intel graphics driver, or set hit_scoring.device = \"CPU\".")
        if cache_dir is not None and device == "GPU":
            cache_dir.mkdir(parents=True, exist_ok=True)
            core.set_property({"CACHE_DIR": str(cache_dir)})
        model = core.read_model(str(store.path(HIT_XML)))
        self.n_inputs = int(model.input(0).get_partial_shape()[1].get_length())
        model.reshape({0: [batch_size, self.n_inputs]})
        self.compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
        self.request = self.compiled.create_infer_request()
        self.device = device
        self.device_name = device_name(core, device)
        self.batch_size = batch_size
        try:
            p = self.compiled.get_property("INFERENCE_PRECISION_HINT")
            self.precision = p.to_string() if hasattr(p, "to_string") else str(p)
        except Exception:  # noqa: BLE001
            self.precision = "unknown"
        self._buf = np.zeros((batch_size, self.n_inputs), dtype=np.float32)

    def infer_batch(self, x: np.ndarray) -> np.ndarray:
        """Infer exactly one batch (``x`` has at most batch_size rows)."""
        n = x.shape[0]
        self._buf[:n] = x
        if n < self.batch_size:
            self._buf[n:] = 0.0
        out = self.request.infer({0: self._buf})
        return np.array(next(iter(out.values()))[:n], dtype=np.float32)

    def score(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return activity and novelty probabilities for every row of ``x``."""
        if x.shape[0] == 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        parts = [self.infer_batch(x[i : i + self.batch_size]) for i in range(0, x.shape[0], self.batch_size)]
        p = np.vstack(parts)
        return p[:, 0], p[:, 1]
