"""Cell-classification agent skill (interim feature rules).

Third and final model step of the HCI microscopy DAG. It measures each
segmentation instance and assigns a class label behind a stable I/O contract,
so the planner (ITEP-96810/96811) can call it as a local tool and the Geti
ResNet classifier (ITEP-96798) can be swapped in later without touching the
report/dataframe step.

Contract:
    input  : 8-bit RGB image (path or HxWx3 uint8) + instances (optional)
    output : {model, input{...}, output{...}, instances:[{bbox_xyxy_px,
             detection_class_id, label, class_id, score, features{...}}, ...]}

Swap point: replace `load_classifier()` only.
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Protocol, Sequence, TypedDict, Union

import numpy as np
from PIL import Image

# Make the sibling segmentation skill importable for composition (ITEP-96811).
_SEG_DIR = Path(__file__).resolve().parents[1] / "segmentation"
if str(_SEG_DIR) not in sys.path:
    sys.path.insert(0, str(_SEG_DIR))

import segment_skill  # noqa: E402  (provides rle_decode + the segment() compose step)

# Interim morphology taxonomy. Geti ResNet (96798) replaces these labels.
CLASS_NAMES = {0: "small-cell", 1: "cell", 2: "elongated", 3: "cluster"}


class Features(TypedDict):
    area_px: int
    mean_intensity: float
    fill_ratio: float
    aspect_ratio: float


class Classification(TypedDict):
    bbox_xyxy_px: list[float]
    detection_class_id: int
    label: str
    class_id: int
    score: float
    features: Features


class Classifier(Protocol):
    def classify(
        self, image: np.ndarray, instances: Sequence[dict[str, Any]]
    ) -> list[Classification]:
        ...


# --------------------------------------------------------------------------- #
# Interim backend: measure morphology/intensity, label by transparent rules.
# --------------------------------------------------------------------------- #
def _measure(gray: np.ndarray, mask: np.ndarray, bbox: Sequence[float]) -> Features:
    area = int(mask.sum())
    x1, y1, x2, y2 = bbox
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    mean_int = float(gray[mask].mean()) if area else 0.0
    return {
        "area_px": area,
        "mean_intensity": round(mean_int, 2),
        "fill_ratio": round(area / (bw * bh), 4),
        "aspect_ratio": round(bw / bh, 4),
    }


def _label(f: Features) -> tuple[str, int, float]:
    """Return (label, class_id, interim_score). Score is a rule margin in [0,1]
    and becomes the model softmax after the swap to Geti ResNet."""
    aspect = f["aspect_ratio"] or 1.0
    fill = f["fill_ratio"]
    area = f["area_px"]
    if aspect >= 1.8 or aspect <= 0.55:
        score = min(1.0, abs(math.log(aspect)) / math.log(1.8))
        return "elongated", 2, round(score, 4)
    if fill >= 0.6 and area >= 5000:
        return "cluster", 3, round(min(1.0, fill), 4)
    if area < 1500:
        return "small-cell", 0, round(min(1.0, 0.5 + 0.5 * (1 - fill)), 4)
    return "cell", 1, round(min(1.0, 0.5 + 0.5 * fill), 4)


class _FeatureClassifier:
    """Interim classifier: measurements + rules. Swapped for Geti ResNet (96798)."""

    def classify(
        self, image: np.ndarray, instances: Sequence[dict[str, Any]]
    ) -> list[Classification]:
        gray = np.asarray(Image.fromarray(image).convert("L"))
        out: list[Classification] = []
        for inst in instances:
            mask = segment_skill.rle_decode(inst["mask_rle"])
            feats = _measure(gray, mask, inst["bbox_xyxy_px"])
            label, class_id, score = _label(feats)
            out.append(
                {
                    "bbox_xyxy_px": inst["bbox_xyxy_px"],
                    "detection_class_id": int(inst.get("class_id", -1)),
                    "label": label,
                    "class_id": class_id,
                    "score": score,
                    "features": feats,
                }
            )
        return out


def load_classifier() -> Classifier:
    """Return the interim feature-rule classifier.

    THIS IS THE SWAP POINT. To move to production, return the Geti ResNet
    classifier (ITEP-96798) that implements `.classify(image, instances) ->
    list of {label, class_id, score, features}` and reads 16-bit intensity
    (ITEP-96793) under the mask.
    """
    return _FeatureClassifier()


# --------------------------------------------------------------------------- #
def _load_rgb(image: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise ValueError("RGB array must be 8-bit (uint8)")
        return Image.fromarray(image).convert("RGB")
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def _get_instances(
    rgb: Image.Image, threshold: float, model_size: str, detector: Any, segmenter: Any
) -> list[dict[str, Any]]:
    """Run detection + segmentation to seed classification (composition, 96811)."""
    contract = segment_skill.segment(
        rgb,
        threshold=threshold,
        model_size=model_size,
        detector=detector,
        segmenter=segmenter,
    )
    return contract["instances"]


def build_contract(
    model_name: str, image: Image.Image, instances: list[Classification]
) -> dict[str, Any]:
    return {
        "model": model_name,
        "input": {
            "format": "RGB",
            "width": image.width,
            "height": image.height,
            "bit_depth": 8,
            "note": "Features measured on the 8-bit projection; production reads "
            "16-bit intensity (ITEP-96793) at the same mask pixels here.",
        },
        "output": {
            "num_instances": len(instances),
            "fields": [
                "bbox_xyxy_px",
                "detection_class_id",
                "label",
                "class_id",
                "score",
                "features",
            ],
            "classes": {str(k): v for k, v in CLASS_NAMES.items()},
        },
        "instances": instances,
    }


def classify(
    image: Union[str, Path, np.ndarray, Image.Image],
    instances: list[dict[str, Any]] | None = None,
    threshold: float = 0.5,
    model_size: str = "base",
    detector: Any = None,
    segmenter: Any = None,
    classifier: Classifier | None = None,
) -> dict[str, Any]:
    """Run classification and return the I/O contract dict.

    If `instances` is None, the detection + segmentation skills run internally
    (full end-to-end pipeline). `classifier` can be injected for tests or to
    supply the production model; otherwise the interim classifier is loaded.
    """
    rgb = _load_rgb(image)
    arr = np.asarray(rgb)
    if instances is None:
        instances = _get_instances(rgb, threshold, model_size, detector, segmenter)

    clf = classifier if classifier is not None else load_classifier()
    results = clf.classify(arr, instances)
    model_name = "injected" if classifier is not None else "feature-rules (interim)"
    return build_contract(model_name, rgb, results)


# --------------------------------------------------------------------------- #
# Dataframe (CSV) + visualization helpers for the meeting demo.
# --------------------------------------------------------------------------- #
def to_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for i, inst in enumerate(contract["instances"]):
        x1, y1, x2, y2 = inst["bbox_xyxy_px"]
        rows.append(
            {
                "instance": i,
                "label": inst["label"],
                "class_id": inst["class_id"],
                "score": inst["score"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                **inst["features"],
            }
        )
    return rows


def write_csv(contract: dict[str, Any], out_path: Union[str, Path]) -> None:
    rows = to_rows(contract)
    if not rows:
        Path(out_path).write_text("")
        return
    with open(out_path, "w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}  ({len(rows)} rows)")


_COLORS = {
    0: (80, 180, 255),
    1: (80, 255, 120),
    2: (255, 210, 60),
    3: (255, 80, 80),
}


def draw_overlay(
    image: Union[str, Path, np.ndarray, Image.Image],
    contract: dict[str, Any],
    out_path: Union[str, Path],
) -> None:
    from PIL import ImageDraw

    img = _load_rgb(image)
    draw = ImageDraw.Draw(img)
    for inst in contract["instances"]:
        color = _COLORS.get(inst["class_id"], (255, 255, 255))
        x1, y1, x2, y2 = inst["bbox_xyxy_px"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1 + 4, y1 + 4), f'{inst["label"]} {inst["score"]:.2f}', fill=color)
    img.save(out_path)
    print(f"Wrote {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cell-classification agent skill (interim rules)")
    p.add_argument("--image", required=True, type=Path, help="8-bit RGB image path")
    p.add_argument("--instances", type=Path, help="segmentation contract JSON (optional)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--model-size", default="base")
    p.add_argument("--out", type=Path, default=Path("classifications.json"))
    p.add_argument("--csv", type=Path, help="write the per-cell dataframe as CSV")
    p.add_argument("--overlay", type=Path, help="write a labeled overlay PNG")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    insts = None
    if args.instances:
        insts = json.loads(args.instances.read_text())["instances"]
    contract = classify(
        args.image, instances=insts, threshold=args.threshold, model_size=args.model_size
    )
    args.out.write_text(json.dumps(contract, indent=2))
    print(f"Wrote {args.out}  ({contract['output']['num_instances']} instances)")
    if args.csv:
        write_csv(contract, args.csv)
    if args.overlay:
        draw_overlay(args.image, contract, args.overlay)


if __name__ == "__main__":
    main()
