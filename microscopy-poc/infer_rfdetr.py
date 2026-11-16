"""Interim RF-DETR inference test for the HLS microscopy pipeline.

Goal (per architect): run RF-DETR in our own framework on a sample image,
print its input/output contract, and optionally export to ONNX/OpenVINO so we
can wrap it as an Agent Skill. This is bootstrap/learning only — Geti will
supply the production models later; we keep this script's I/O as the skill's
contract so the official model can be swapped in behind it.

Setup:
    python -m venv .venv && source .venv/bin/activate
    pip install rfdetr supervision pillow

Run:
    python infer_rfdetr.py --image sample.jpg --threshold 0.5
    python infer_rfdetr.py --image sample.jpg --export onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interim RF-DETR inference test")
    p.add_argument("--image", required=True, type=Path, help="Path to input image")
    p.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    p.add_argument(
        "--model",
        default="base",
        choices=["base", "large"],
        help="RF-DETR size to load (interim; Geti sizes nano-2XL later)",
    )
    p.add_argument(
        "--export",
        choices=["onnx"],
        default=None,
        help="Optionally export the model (ONNX) for OpenVINO conversion",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("rfdetr_io_contract.json"),
        help="Where to write the observed I/O contract",
    )
    return p.parse_args()


def load_model(size: str):
    # RF-DETR (Roboflow). Interim = public COCO-pretrained weights.
    from rfdetr import RFDETRBase, RFDETRLarge

    return (RFDETRLarge if size == "large" else RFDETRBase)()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    image = Image.open(args.image).convert("RGB")
    model = load_model(args.model)

    # supervision.Detections: xyxy (pixel coords), confidence, class_id
    detections = model.predict(image, threshold=args.threshold)

    boxes = []
    for xyxy, conf, class_id in zip(
        detections.xyxy, detections.confidence, detections.class_id
    ):
        boxes.append(
            {
                "bbox_xyxy_px": [round(float(v), 2) for v in xyxy],
                "class_id": int(class_id),
                "confidence": round(float(conf), 4),
            }
        )

    # This is the contract the detection Agent Skill must expose/consume.
    contract = {
        "model": f"rfdetr-{args.model}",
        "input": {
            "format": "RGB",
            "width": image.width,
            "height": image.height,
            "bit_depth": 8,
            "note": "Interim test uses 8-bit RGB. Confirm raw/16-bit per Friday PoC.",
        },
        "output": {
            "box_coord_space": "pixel_xyxy",
            "num_detections": len(boxes),
            "fields": ["bbox_xyxy_px", "class_id", "confidence"],
        },
        "detections": boxes,
    }

    print(json.dumps(contract, indent=2))
    args.out.write_text(json.dumps(contract, indent=2))
    print(f"\nI/O contract written to {args.out}")

    if args.export == "onnx":
        # Produces an ONNX graph we can convert with OpenVINO `ovc`.
        model.export()
        print("Exported ONNX (see RF-DETR output dir). Next: ovc <model>.onnx")


if __name__ == "__main__":
    main()
