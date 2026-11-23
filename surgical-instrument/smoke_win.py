"""PTL smoke test: app train/export path on torch-xpu with a synthetic dataset.

Usage:  set APP=<path>\\surgical-instrument  &&  python smoke.py [epochs] [n_train]
"""
import os
import random
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

APP = Path(os.environ.get("APP", ".")).resolve()
ROOT = Path(tempfile.gettempdir()) / "si_smoke"


def make_dataset(n_train: int) -> Path:
    ds = ROOT / "ds"
    rng = random.Random(0)
    for split, n in (("train", n_train), ("val", 32)):
        (ds / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.full((640, 640, 3), rng.randint(40, 120), np.uint8)
            w, h = rng.randint(60, 200), rng.randint(60, 200)
            x, y = rng.randint(0, 640 - w), rng.randint(0, 640 - h)
            cv2.ellipse(img, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, (60, 60, 220), -1)
            cv2.imwrite(str(ds / "images" / split / f"{i}.jpg"), img)
            (ds / "labels" / split / f"{i}.txt").write_text(
                f"0 {(x + w / 2) / 640:.6f} {(y + h / 2) / 640:.6f} {w / 640:.6f} {h / 640:.6f}\n"
            )
    data_yaml = ds / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(
        {"path": str(ds), "train": "images/train", "val": "images/val", "names": {0: "Polyp"}}))
    return data_yaml


def main() -> None:
    sys.path.insert(0, str(APP))
    import torch
    from backend.bootstrap import train as T  # applies xpu_compat before ultralytics
    from backend.bootstrap.export import export_model

    print("torch", torch.__version__, "| xpu:", torch.xpu.is_available(),
          torch.xpu.get_device_name(0) if torch.xpu.is_available() else "-")
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    data_yaml = make_dataset(n_train)

    cfg = yaml.safe_load((APP / "backend/config/model.yaml").read_text())
    tcfg = dict(cfg["train"])
    tcfg.update(epochs=epochs, workers=0 if os.name == "nt" else 2, cache=False)
    print("train cfg:", {k: tcfg[k] for k in ("device", "epochs", "batch", "imgsz", "amp", "workers")})

    def progress(e):
        if e.get("state") in ("starting", "epoch_end", "done"):
            print("PROGRESS", e, flush=True)

    res = T.train_model("yolo11n", tcfg, data_yaml, ROOT / "artifacts", progress=progress)
    print("TRAIN OK", res)
    ir = export_model(Path(res["best_pt"]), cfg["export"], ROOT / "ir")
    print("EXPORT OK", ir, sorted(p.name for p in ir.iterdir()))


if __name__ == "__main__":
    main()
