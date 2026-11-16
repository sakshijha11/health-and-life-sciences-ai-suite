---
name: cell-classification
description: >
  Label each segmented cell/object and extract per-instance morphology +
  intensity features. Interim implementation uses transparent feature rules;
  the Geti ResNet classifier (ITEP-96798) swaps in behind the same contract.
when_to_use: >
  Use as the THIRD and final model step of the HCI analysis pipeline, after
  segmentation has produced instance masks. It measures each mask and assigns a
  class label, producing the per-cell rows of the final dataframe.
target_platform: Xeon
target_os: Ubuntu 24.04
status: interim (feature rules) — pending Geti ResNet weights (~Dec preview)
related_tickets:
  - ITEP-96798 (Geti ResNet classification skill — this skill, production backend)
  - ITEP-96811 (compose segmentation -> classification — demonstrated here)
  - ITEP-96794 (pipeline — classification node + dataframe/report output)
  - ITEP-96793 (16-bit training — intensity under the mask is the 16-bit hook)
---

# Cell Classification Agent Skill

## What it does
Takes one 8-bit RGB tile plus segmentation instances (masks) and returns, per
instance, a **class label + score + measured features** (area, mean intensity,
fill ratio, aspect ratio). This is the classification node of the microscopy DAG
(`ITEP-96794`) and produces the per-cell rows of the final dataframe. It can
consume instances from the segmentation skill (composition, `ITEP-96811`) or run
the whole detection -> segmentation chain itself.

## Input contract
| Field | Type | Notes |
|-------|------|-------|
| `image` | path **or** HxWx3 uint8 RGB array | 8-bit RGB (same as upstream). |
| `instances` | list of `{bbox_xyxy_px, class_id, mask_rle, ...}` *(optional)* | Output of the segmentation skill. If omitted, detection + segmentation run internally. |
| `threshold` / `model_size` | float / str | Only used when upstream runs internally. |

**16-bit hook:** `mean_intensity` is measured here, under the mask. The interim
reads the 8-bit projection; the production step reads the **16-bit source**
(`ITEP-96793`) at the same mask pixels — this skill is the designed place for it.

## Output contract (stable — the swap boundary)
```json
{
  "model": "feature-rules (interim)",
  "input":  { "format": "RGB", "width": 1351, "height": 1350, "bit_depth": 8 },
  "output": {
    "num_instances": 4,
    "fields": ["bbox_xyxy_px", "detection_class_id", "label", "class_id", "score", "features"],
    "classes": { "0": "small-cell", "1": "cell", "2": "elongated", "3": "cluster" }
  },
  "instances": [
    { "bbox_xyxy_px": [70.6, 248.4, 649.0, 929.0], "detection_class_id": 18,
      "label": "cluster", "class_id": 3, "score": 0.71,
      "features": { "area_px": 118442, "mean_intensity": 142.3,
                    "fill_ratio": 0.33, "aspect_ratio": 0.85 } }
  ]
}
```
- `label` / `class_id` / `score`: the classification (score is an interim rule
  margin; it becomes the model softmax after the swap).
- `features`: the quantitative measurements that flow into the final dataframe.

Any classifier (feature rules now, Geti ResNet later) that emits this JSON is a
drop-in — the report/dataframe step does not change.

## How to run
```bash
# run the WHOLE pipeline (detection -> segmentation -> classification) on an image
python classify_skill.py --image ../sample.jpg --threshold 0.5 \
    --out classifications.json --csv results.csv --overlay classified_overlay.png

# or the one-command end-to-end demo:
python ../run_pipeline.py --image ../../sample.jpg
```

## Validation
```bash
python test_classify_skill.py      # or: pytest test_classify_skill.py
```

## Swap policy (interim → production)
1. Keep this output contract frozen.
2. Replace `load_classifier()` with the Geti ResNet classifier (`96798`).
3. Point `mean_intensity` at the 16-bit source (`96793`) at the same mask pixels.
4. Re-run `test_classify_skill.py`; the contract + CSV tests must still pass.
5. Deployment path already proven: PyTorch → ONNX → OpenVINO IR (FP16) → CPU/Xeon.
