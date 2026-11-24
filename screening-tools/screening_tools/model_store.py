# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model files produced by the local setup step.

Only files written by ``python -m screening_tools setup`` on this machine are
loaded. The manifest records a SHA-256 for every file and the feature-list
version; the runtime refuses files whose hash or feature version differs.
The anomaly model is a scikit-learn object stored with joblib (pickle), so
the hash check runs before it is unpickled.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from screening_tools.config import APP_ROOT
from screening_tools.features import FEATURE_VERSION, Standardizer

DEFAULT_MODEL_DIR = APP_ROOT / "models"
MANIFEST = "manifest.json"
HIT_XML = "hit_mlp.xml"
HIT_BIN = "hit_mlp.bin"
HIT_SCALER = "hit_scaler.json"
ANOMALY_MODEL = "anomaly_iforest.joblib"
ANOMALY_SCALER = "anomaly_scaler.json"
MODEL_FILES = (HIT_XML, HIT_BIN, HIT_SCALER, ANOMALY_MODEL, ANOMALY_SCALER)


class ModelError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ModelStore:
    directory: Path = DEFAULT_MODEL_DIR

    def path(self, name: str) -> Path:
        return self.directory / name

    def write_manifest(self, manifest: dict) -> None:
        manifest = dict(manifest)
        manifest["feature_version"] = FEATURE_VERSION
        manifest["files"] = {name: sha256(self.path(name)) for name in MODEL_FILES}
        self.path(MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def manifest(self) -> dict:
        mpath = self.path(MANIFEST)
        if not mpath.is_file():
            raise ModelError(f"no models in {self.directory}. Run: python -m screening_tools setup")
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        if manifest.get("feature_version") != FEATURE_VERSION:
            raise ModelError(
                f"models were built with feature version {manifest.get('feature_version')}, "
                f"this code uses {FEATURE_VERSION}. Run: python -m screening_tools setup"
            )
        for name in MODEL_FILES:
            p = self.path(name)
            if not p.is_file():
                raise ModelError(f"missing model file {p}. Run: python -m screening_tools setup")
            if sha256(p) != manifest["files"].get(name):
                raise ModelError(f"{p} does not match the manifest written by setup; refusing to load it")
        return manifest

    def check(self, cfg: dict) -> dict:
        manifest = self.manifest()
        if manifest["descriptors"] != cfg["simulator"]["descriptors"]:
            raise ModelError("simulator.descriptors differs from the value the models were built with. Re-run setup.")
        return manifest

    def load_anomaly(self):
        import joblib

        self.manifest()  # verifies hashes before unpickling
        return joblib.load(self.path(ANOMALY_MODEL)), Standardizer.load(self.path(ANOMALY_SCALER))

    def load_hit_scaler(self) -> Standardizer:
        return Standardizer.load(self.path(HIT_SCALER))
