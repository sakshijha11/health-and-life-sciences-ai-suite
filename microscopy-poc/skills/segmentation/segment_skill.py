"""Cell-segmentation agent skill (interim classical Otsu).

Second model step of the HCI microscopy DAG. It takes detection boxes and
returns one instance mask per box behind a stable I/O contract, so the planner
(ITEP-96810/96811) can call it as a local tool and the production segmenter
(MaskRCNN-SwinT, ITEP-96796, or Geti SAM, ITEP-96797) can be swapped in later
without touching any downstream skill.

Contract:
    input  : 8-bit RGB image (path or HxWx3 uint8) + detections (optional)
    output : {model, input{...}, output{...}, instances:[{bbox_xyxy_px,
             class_id, area_px, mask_rle{size,counts}}, ...]}  (COCO RLE masks)

Swap point: replace `load_segmenter()` only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Protocol, Sequence, TypedDict, Union

import numpy as np
from PIL import Image

# Make the sibling detection skill importable for composition (ITEP-96811).
_DETECTION_DIR = Path(__file__).resolve().parents[1] / "detection"
if str(_DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(_DETECTION_DIR))


class MaskRLE(TypedDict):
    size: list[int]  # [height, width]
    counts: list[int]  # COCO column-major run lengths, starting with a 0-run


class Instance(TypedDict):
    bbox_xyxy_px: list[float]
    class_id: int
    area_px: int
    mask_rle: MaskRLE


class Segmenter(Protocol):
    def segment(
        self, image: np.ndarray, boxes: Sequence[Sequence[float]]
    ) -> list[np.ndarray]:
        """Return one full-image HxW boolean mask per input box."""
        ...


# --------------------------------------------------------------------------- #
# COCO run-length encoding (column-major / Fortran order), pure numpy.
# --------------------------------------------------------------------------- #
def rle_encode(mask: np.ndarray) -> MaskRLE:
    h, w = mask.shape
    flat = np.asarray(mask, dtype=bool).flatten(order="F").astype(np.uint8)
    change = np.where(np.diff(flat) != 0)[0] + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0] == 1:  # COCO counts start with a run of zeros
        counts = [0] + counts
    return {"size": [int(h), int(w)], "counts": [int(c) for c in counts]}


def rle_decode(rle: MaskRLE) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    value = 0
    for count in rle["counts"]:
        if value == 1:
            flat[idx : idx + count] = 1
        idx += count
        value ^= 1
    return flat.reshape((h, w), order="F").astype(bool)


# --------------------------------------------------------------------------- #
# Interim backend: Otsu threshold inside each detection box.
# --------------------------------------------------------------------------- #
def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (total - omega)
    numer = (mu_t * omega - mu) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 0, numer / denom, 0.0)
    return int(np.argmax(sigma_b))


class _ClassicalSegmenter:
    """Interim segmenter: per-box Otsu foreground. Swapped for 96796/96797."""

    def segment(
        self, image: np.ndarray, boxes: Sequence[Sequence[float]]
    ) -> list[np.ndarray]:
        gray = np.asarray(Image.fromarray(image).convert("L"))
        h, w = gray.shape
        masks: list[np.ndarray] = []
        for x1, y1, x2, y2 in boxes:
            xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
            xi2, yi2 = min(w, int(round(x2))), min(h, int(round(y2)))
            full = np.zeros((h, w), dtype=bool)
            if xi2 > xi1 and yi2 > yi1:
                crop = gray[yi1:yi2, xi1:xi2]
                fg = crop > _otsu_threshold(crop)
                if fg.mean() > 0.5:  # objects are usually the minority region
                    fg = ~fg
                full[yi1:yi2, xi1:xi2] = fg
            masks.append(full)
        return masks


def load_segmenter() -> Segmenter:
    """Return the interim classical segmenter.

    THIS IS THE SWAP POINT. To move to production, return the MaskRCNN-SwinT
    (ITEP-96796) or Geti SAM (ITEP-96797) segmenter that implements
    `.segment(image, boxes) -> list[HxW bool mask]`.
    """
    return _ClassicalSegmenter()


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


def _get_detections(
    rgb: Image.Image, threshold: float, model_size: str, detector: Any
) -> list[dict[str, Any]]:
    """Run the detection skill to seed segmentation (composition, 96811)."""
    import detect_skill

    contract = detect_skill.detect(
        rgb, threshold=threshold, model_size=model_size, detector=detector
    )
    return contract["detections"]


def build_contract(
    model_name: str, image: Image.Image, instances: list[Instance]
) -> dict[str, Any]:
    return {
        "model": model_name,
        "input": {
            "format": "RGB",
            "width": image.width,
            "height": image.height,
            "bit_depth": 8,
            "note": "Masks come from an 8-bit RGB projection; quantitative "
            "16-bit intensity (ITEP-96793) is read under the mask in the "
            "classification step, not here.",
        },
        "output": {
            "mask_format": "coco_rle",
            "num_instances": len(instances),
            "fields": ["bbox_xyxy_px", "class_id", "area_px", "mask_rle"],
        },
        "instances": instances,
    }


def segment(
    image: Union[str, Path, np.ndarray, Image.Image],
    detections: list[dict[str, Any]] | None = None,
    threshold: float = 0.5,
    model_size: str = "base",
    detector: Any = None,
    segmenter: Segmenter | None = None,
) -> dict[str, Any]:
    """Run segmentation and return the I/O contract dict.

    If `detections` is None, the detection skill is run internally. `segmenter`
    can be injected (for tests or to supply the production model); otherwise the
    interim classical segmenter is loaded.
    """
    rgb = _load_rgb(image)
    arr = np.asarray(rgb)
    if detections is None:
        detections = _get_detections(rgb, threshold, model_size, detector)

    seg = segmenter if segmenter is not None else load_segmenter()
    boxes = [d["bbox_xyxy_px"] for d in detections]
    masks = seg.segment(arr, boxes)

    instances: list[Instance] = []
    for det, mask in zip(detections, masks):
        instances.append(
            {
                "bbox_xyxy_px": det["bbox_xyxy_px"],
                "class_id": int(det.get("class_id", -1)),
                "area_px": int(mask.sum()),
                "mask_rle": rle_encode(mask),
            }
        )

    model_name = "injected" if segmenter is not None else "classical-otsu (interim)"
    return build_contract(model_name, rgb, instances)


# --------------------------------------------------------------------------- #
# Visualization helper (for the meeting demo).
# --------------------------------------------------------------------------- #
_COLORS = [
    (255, 80, 80),
    (80, 180, 255),
    (80, 255, 120),
    (255, 210, 60),
    (200, 120, 255),
    (255, 140, 0),
]


def draw_overlay(
    image: Union[str, Path, np.ndarray, Image.Image],
    contract: dict[str, Any],
    out_path: Union[str, Path],
) -> None:
    from PIL import ImageDraw

    base = _load_rgb(image).convert("RGBA")
    for i, inst in enumerate(contract["instances"]):
        color = _COLORS[i % len(_COLORS)]
        mask = rle_decode(inst["mask_rle"])
        layer = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        layer[mask] = [color[0], color[1], color[2], 110]
        base = Image.alpha_composite(base, Image.fromarray(layer, "RGBA"))
    draw = ImageDraw.Draw(base)
    for i, inst in enumerate(contract["instances"]):
        color = _COLORS[i % len(_COLORS)] + (255,)
        x1, y1, x2, y2 = inst["bbox_xyxy_px"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1 + 3, y1 + 3), f'id{inst["class_id"]} {inst["area_px"]}px', fill=color)
    base.convert("RGB").save(out_path)
    print(f"Wrote {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cell-segmentation agent skill (interim Otsu)")
    p.add_argument("--image", required=True, type=Path, help="8-bit RGB image path")
    p.add_argument("--detections", type=Path, help="detection contract JSON (optional)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--model-size", default="base")
    p.add_argument("--out", type=Path, default=Path("instances.json"))
    p.add_argument("--overlay", type=Path, help="write a colored mask overlay PNG")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dets = None
    if args.detections:
        dets = json.loads(args.detections.read_text())["detections"]
    contract = segment(
        args.image, detections=dets, threshold=args.threshold, model_size=args.model_size
    )
    args.out.write_text(json.dumps(contract, indent=2))
    print(f"Wrote {args.out}  ({contract['output']['num_instances']} instances)")
    if args.overlay:
        draw_overlay(args.image, contract, args.overlay)


if __name__ == "__main__":
    main()
