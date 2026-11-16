---
name: cell-detection
description: >
  Detect cells/objects in a microscopy image and return bounding boxes.
  Interim implementation uses RF-DETR (Roboflow); the Geti domain-trained
  detector (or D-FINE-M) swaps in behind the same input/output contract.
when_to_use: >
  Use as the FIRST model step of the HCI analysis pipeline, after an image
  reader (OME-Zarr 96803 / OME-TIFF 96804) and OpenCV preprocess (96802) have
  produced an 8-bit RGB tile. Downstream: segmentation (96796/96797) consumes
  these boxes to crop ROIs; classification (96798) labels each ROI.
target_platform: Xeon
target_os: Ubuntu 24.04
status: interim (RF-DETR COCO weights) — pending Geti domain weights (~Dec preview)
related_tickets:
  - ITEP-96794 (pipeline — detection node, listed as D-FINE-M)
  - ITEP-96799 (run whole DAG as a skill)
  - ITEP-96802 (OpenCV preprocess — upstream)
  - ITEP-96803 / ITEP-96804 (readers — upstream)
  - "detection skill has NO dedicated ticket yet — propose creating one"
---

# Cell Detection Agent Skill

## What it does
Runs an object detector on one 8-bit RGB microscopy tile and returns a list of
bounding boxes with class id and confidence. This is the detection node of the
microscopy DAG (`ITEP-96794`), exposed as a composable agent skill.

## Input contract
| Field | Type | Notes |
|-------|------|-------|
| `image` | path to image **or** HxWx3 uint8 RGB array | 8-bit RGB. Upstream reader+OpenCV produce this from OME-Zarr/OME-TIFF. |
| `threshold` | float (default 0.5) | Minimum confidence to keep a detection. |
| `model_size` | `nano\|small\|medium\|base\|large` | Interim RF-DETR size. Geti sizes nano–2XL later. |

**16-bit note:** scientific intensity (`ITEP-96793`) is 16-bit; this skill
intentionally consumes an 8-bit RGB projection for *detection*. Quantitative
intensity must be read from the 16-bit source in the classification step, not here.

## Output contract (stable — the swap boundary)
```json
{
  "model": "rfdetr-base",
  "input":  { "format": "RGB", "width": 1351, "height": 1350, "bit_depth": 8 },
  "output": { "box_coord_space": "pixel_xyxy", "num_detections": 4,
              "fields": ["bbox_xyxy_px", "class_id", "confidence"] },
  "detections": [
    { "bbox_xyxy_px": [70.56, 248.39, 648.98, 929.04], "class_id": 18, "confidence": 0.9248 }
  ]
}
```
- `bbox_xyxy_px`: `[x1, y1, x2, y2]` in **pixel** coordinates of the input tile.
- `class_id` / `confidence`: integer label, float score.

Any detector (RF-DETR now, Geti/D-FINE-M later) that emits this JSON is a valid
drop-in — the planner, segmentation, and classification skills do not change.

## How to run
```bash
# produces detections + writes the I/O contract JSON
python detect_skill.py --image ../sample.jpg --threshold 0.5 --out detections.json
```

## Validation
```bash
python test_detect_skill.py        # or: pytest test_detect_skill.py
```

## Swap policy (interim → production)
1. Keep this output contract frozen.
2. Replace `load_detector()` with the Geti-trained detector (or D-FINE-M).
3. Re-run `test_detect_skill.py`; the contract test must still pass.
4. Deployment path already proven: PyTorch → ONNX → OpenVINO IR (FP16) → CPU/Xeon.
