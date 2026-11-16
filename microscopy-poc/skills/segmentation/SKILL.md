---
name: cell-segmentation
description: >
  Produce instance masks for cells/objects in a microscopy image. Interim
  implementation uses classical thresholding (Otsu) inside each detection box;
  MaskRCNN-SwinT (ITEP-96796) or Geti SAM (ITEP-96797) swaps in behind the
  same input/output contract.
when_to_use: >
  Use as the SECOND model step of the HCI analysis pipeline, after detection
  has produced boxes. It crops each ROI and emits a pixel mask per instance.
  Downstream: classification (96798) applies the mask and labels each cell;
  the DAG/report step measures mask area + intensity.
target_platform: Xeon
target_os: Ubuntu 24.04
status: interim (classical Otsu) — pending MaskRCNN-SwinT / Geti SAM weights (~Dec preview)
related_tickets:
  - ITEP-96796 (MaskRCNN-SwinT segmentation skill — this skill, production backend)
  - ITEP-96797 (Geti SAM segmentation — alternate backend, same contract)
  - ITEP-96811 (compose segmentation after detection — demonstrated here)
  - ITEP-96794 (pipeline — segmentation node)
  - ITEP-96798 (classification — downstream consumer of masks)
---

# Cell Segmentation Agent Skill

## What it does
Takes one 8-bit RGB tile plus detection boxes and returns one **instance mask**
per box. This is the segmentation node of the microscopy DAG (`ITEP-96794`),
exposed as a composable agent skill. It can consume detections from the
detection skill (skill **composition**, `ITEP-96811`) or run detection itself.

## Input contract
| Field | Type | Notes |
|-------|------|-------|
| `image` | path **or** HxWx3 uint8 RGB array | 8-bit RGB (same as detection). |
| `detections` | list of `{bbox_xyxy_px, class_id, ...}` *(optional)* | Output of the detection skill. If omitted, detection runs internally. |
| `threshold` | float (default 0.5) | Only used when detection runs internally. |
| `model_size` | `nano\|small\|medium\|base\|large` | Only used when detection runs internally. |

## Output contract (stable — the swap boundary)
```json
{
  "model": "classical-otsu (interim)",
  "input":  { "format": "RGB", "width": 1351, "height": 1350, "bit_depth": 8 },
  "output": { "mask_format": "coco_rle", "num_instances": 4,
              "fields": ["bbox_xyxy_px", "class_id", "area_px", "mask_rle"] },
  "instances": [
    { "bbox_xyxy_px": [70.6, 248.4, 649.0, 929.0], "class_id": 18,
      "area_px": 118442,
      "mask_rle": { "size": [1350, 1351], "counts": [12003, 44, 1306, 51, ...] } }
  ]
}
```
- `mask_rle`: **COCO run-length encoding** (column-major `counts`, starting with a
  0-run). Standard, compact, and decodes to a full-image `H×W` boolean mask.
- `area_px`: number of foreground pixels in the mask.
- `bbox_xyxy_px` / `class_id`: carried through from the detection that seeded it.

Any segmenter (classical now, MaskRCNN-SwinT / Geti SAM later) that emits this
JSON is a drop-in — classification, the planner, and the report step don't change.

## How to run
```bash
# compose detection -> segmentation, write contract + a visual overlay PNG
python segment_skill.py --image ../sample.jpg --threshold 0.5 \
    --out instances.json --overlay segmentation_overlay.png
```

## Validation
```bash
python test_segment_skill.py       # or: pytest test_segment_skill.py
```

## Swap policy (interim → production)
1. Keep this output contract frozen (COCO RLE).
2. Replace `load_segmenter()` with MaskRCNN-SwinT (`96796`) or Geti SAM (`96797`).
3. Re-run `test_segment_skill.py`; the contract + RLE round-trip tests must pass.
4. Deployment path already proven: PyTorch → ONNX → OpenVINO IR (FP16) → CPU/Xeon.
