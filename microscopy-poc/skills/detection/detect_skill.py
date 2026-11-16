"""Cell-detection agent skill (interim RF-DETR).

This wraps a detector behind a stable I/O contract so the planner agent
(ITEP-96810/96811) can call it as a local tool, and so the Geti domain-trained
detector (or D-FINE-M, per ITEP-96794) can be swapped in later without touching
any downstream skill.

Contract:
    input  : 8-bit RGB image (path or HxWx3 uint8 array) + threshold
    output : {model, input{...}, output{...}, detections:[{bbox_xyxy_px,
             class_id, confidence}, ...]}  (boxes in pixel xyxy)

Swap point: replace `load_detector()` only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol, Sequence, TypedDict, Union

import numpy as np
from PIL import Image

# RF-DETR size name -> rfdetr class name (interim weights are COCO-pretrained).
_RFDETR_SIZES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


class Detection(TypedDict):
    bbox_xyxy_px: list[float]
    class_id: int
    confidence: float


class DetectorResult(Protocol):
    """Shape of a supervision.Detections result (duck-typed)."""

    xyxy: Sequence[Sequence[float]]
    confidence: Sequence[float]
    class_id: Sequence[int]


class Detector(Protocol):
    def predict(self, image: Any, threshold: float) -> DetectorResult: ...


def load_detector(model_size: str = "base") -> Detector:
    """Return the interim RF-DETR detector.

    THIS IS THE SWAP POINT. To move to the production model, return the
    Geti-trained detector (or D-FINE-M) that implements `.predict(image,
    threshold) -> result with .xyxy/.confidence/.class_id`.
    """
    import rfdetr

    cls_name = _RFDETR_SIZES.get(model_size, "RFDETRBase")
    return getattr(rfdetr, cls_name)()


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


def result_to_detections(result: DetectorResult) -> list[Detection]:
    """Normalize any detector result into the stable detection list."""
    boxes: list[Detection] = []
    for xyxy, conf, class_id in zip(
        result.xyxy, result.confidence, result.class_id
    ):
        boxes.append(
            {
                "bbox_xyxy_px": [round(float(v), 2) for v in xyxy],
                "class_id": int(class_id),
                "confidence": round(float(conf), 4),
            }
        )
    return boxes


def build_contract(
    model_name: str, image: Image.Image, detections: list[Detection]
) -> dict[str, Any]:
    return {
        "model": model_name,
        "input": {
            "format": "RGB",
            "width": image.width,
            "height": image.height,
            "bit_depth": 8,
            "note": "Detection runs on 8-bit RGB; 16-bit intensity (ITEP-96793) "
            "is read in the classification step, not here.",
        },
        "output": {
            "box_coord_space": "pixel_xyxy",
            "num_detections": len(detections),
            "fields": ["bbox_xyxy_px", "class_id", "confidence"],
        },
        "detections": detections,
    }


def detect(
    image: Union[str, Path, np.ndarray, Image.Image],
    threshold: float = 0.5,
    model_size: str = "base",
    detector: Detector | None = None,
) -> dict[str, Any]:
    """Run detection and return the I/O contract dict.

    `detector` can be injected (for tests or to supply the production model);
    otherwise the interim RF-DETR detector is loaded.
    """
    rgb = _load_rgb(image)
    det = detector if detector is not None else load_detector(model_size)
    result = det.predict(rgb, threshold=threshold)
    detections = result_to_detections(result)
    model_name = f"rfdetr-{model_size}" if detector is None else "injected"
    return build_contract(model_name, rgb, detections)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cell-detection agent skill (interim RF-DETR)")
    p.add_argument("--image", required=True, type=Path, help="8-bit RGB image path")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--model-size", default="base", choices=list(_RFDETR_SIZES))
    p.add_argument("--out", type=Path, default=Path("detections.json"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    contract = detect(args.image, threshold=args.threshold, model_size=args.model_size)
    print(json.dumps(contract, indent=2))
    args.out.write_text(json.dumps(contract, indent=2))
    print(f"\nWrote {args.out}  ({contract['output']['num_detections']} detections)")


if __name__ == "__main__":
    main()
