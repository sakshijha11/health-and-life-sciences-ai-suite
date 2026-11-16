"""Tests for the cell-classification agent skill.

Runs with pytest OR directly:  python test_classify_skill.py

- Unit tests (always run): the real (download-free) feature classifier labels
  synthetic instances; CSV export + injection path are checked too.
- Integration test (auto-skips): if `rfdetr` + a sample image are present, run
  the full detection -> segmentation -> classification pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import classify_skill
import segment_skill


def _instance_from_mask(mask: np.ndarray, bbox, class_id=5) -> dict:
    return {
        "bbox_xyxy_px": list(map(float, bbox)),
        "class_id": class_id,
        "mask_rle": segment_skill.rle_encode(mask),
    }


class _FakeClassifier:
    def classify(self, image, instances):  # noqa: ANN001
        return [
            {
                "bbox_xyxy_px": i["bbox_xyxy_px"],
                "detection_class_id": int(i.get("class_id", -1)),
                "label": "cell",
                "class_id": 1,
                "score": 0.5,
                "features": {
                    "area_px": 1,
                    "mean_intensity": 0.0,
                    "fill_ratio": 0.0,
                    "aspect_ratio": 1.0,
                },
            }
            for i in instances
        ]


def test_classifies_instances() -> None:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[40:200, 60:120] = 200  # bright region so mean_intensity > 0

    mask = np.zeros((240, 320), dtype=bool)
    mask[40:200, 60:120] = True  # 160x60 filled -> elongated (aspect ~0.375)
    instances = [_instance_from_mask(mask, [60, 40, 120, 200])]

    contract = classify_skill.classify(img, instances=instances)
    assert contract["model"] == "feature-rules (interim)"
    assert contract["output"]["num_instances"] == 1

    inst = contract["instances"][0]
    assert set(inst) == {
        "bbox_xyxy_px",
        "detection_class_id",
        "label",
        "class_id",
        "score",
        "features",
    }
    assert inst["label"] in classify_skill.CLASS_NAMES.values()
    assert inst["features"]["area_px"] == 160 * 60
    assert inst["features"]["mean_intensity"] > 0.0
    assert 0.0 <= inst["score"] <= 1.0


def test_injected_classifier_and_csv(tmp_path=None) -> None:
    import tempfile

    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 10:20] = True
    instances = [_instance_from_mask(mask, [10, 10, 20, 20])]

    contract = classify_skill.classify(
        img, instances=instances, classifier=_FakeClassifier()
    )
    assert contract["model"] == "injected"

    rows = classify_skill.to_rows(contract)
    assert rows[0]["label"] == "cell"
    assert {"instance", "label", "class_id", "score", "x1", "area_px"} <= set(rows[0])

    out = Path(tempfile.mkdtemp()) / "results.csv"
    classify_skill.write_csv(contract, out)
    text = out.read_text()
    assert "label" in text.splitlines()[0]
    assert "cell" in text


def test_rejects_non_uint8_array() -> None:
    bad = np.zeros((8, 8, 3), dtype=np.float32)
    try:
        classify_skill.classify(bad, instances=[])
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-uint8 RGB array")


def test_integration_end_to_end() -> None:
    """Auto-skips unless rfdetr + a sample image are available."""
    try:
        import rfdetr  # noqa: F401
    except Exception:
        print("SKIP integration: rfdetr not installed")
        return
    sample = Path(__file__).resolve().parents[2] / "sample.jpg"
    if not sample.exists():
        print(f"SKIP integration: {sample} not found")
        return
    contract = classify_skill.classify(sample, threshold=0.5, model_size="base")
    n = contract["output"]["num_instances"]
    assert n == len(contract["instances"])
    for inst in contract["instances"]:
        assert inst["label"] in classify_skill.CLASS_NAMES.values()
        assert 0.0 <= inst["score"] <= 1.0
    print(f"OK integration (det->seg->cls): {n} classified cells")


if __name__ == "__main__":
    test_classifies_instances()
    print("PASS test_classifies_instances")
    test_injected_classifier_and_csv()
    print("PASS test_injected_classifier_and_csv")
    test_rejects_non_uint8_array()
    print("PASS test_rejects_non_uint8_array")
    test_integration_end_to_end()
    print("DONE")
