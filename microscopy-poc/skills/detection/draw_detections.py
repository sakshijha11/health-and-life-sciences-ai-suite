"""Draw detection boxes onto an image — the meeting visual.

    python draw_detections.py --image ../sample.jpg --out detections_overlay.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

import detect_skill

_COLORS = [
    (255, 80, 80),
    (80, 180, 255),
    (80, 255, 120),
    (255, 210, 60),
    (200, 120, 255),
    (255, 140, 0),
]


def draw(image_path: Path, threshold: float, model_size: str, out: Path) -> None:
    contract = detect_skill.detect(
        image_path, threshold=threshold, model_size=model_size
    )
    img = Image.open(image_path).convert("RGB")
    canvas = ImageDraw.Draw(img)
    for i, det in enumerate(contract["detections"]):
        color = _COLORS[i % len(_COLORS)]
        x1, y1, x2, y2 = det["bbox_xyxy_px"]
        canvas.rectangle([x1, y1, x2, y2], outline=color, width=4)
        canvas.text(
            (x1 + 4, y1 + 4),
            f'id{det["class_id"]}  {det["confidence"]:.2f}',
            fill=color,
        )
    img.save(out)
    print(f"Wrote {out}  ({contract['output']['num_detections']} boxes)")


def main() -> None:
    p = argparse.ArgumentParser(description="Draw detection boxes (meeting visual)")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--model-size", default="base", choices=list(detect_skill._RFDETR_SIZES))
    p.add_argument("--out", type=Path, default=Path("detections_overlay.png"))
    args = p.parse_args()
    draw(args.image, args.threshold, args.model_size, args.out)


if __name__ == "__main__":
    main()
