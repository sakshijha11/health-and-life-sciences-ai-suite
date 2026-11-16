"""Tests for the cell-segmentation agent skill.

Runs with pytest OR directly:  python test_segment_skill.py

- Unit tests (always run): RLE round-trip + contract/serialization with fake
  detections and a fake segmenter (no model, no network).
- Integration test (auto-skips): if `rfdetr` + a sample image are present, run
  the real detection -> segmentation composition and validate the contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import segment_skill


class _FakeSegmenter:
    def segment(self, image, boxes):  # noqa: ANN001
        h, w = image.shape[:2]
        masks = []
        for x1, y1, x2, y2 in boxes:
            m = np.zeros((h, w), dtype=bool)
            m[int(y1) : int(y2), int(x1) : int(x2)] = True
            masks.append(m)
        return masks


def test_rle_roundtrip() -> None:
    rng = np.random.default_rng(0)
    mask = rng.integers(0, 2, size=(37, 41), dtype=np.uint8).astype(bool)
    restored = segment_skill.rle_decode(segment_skill.rle_encode(mask))
    assert restored.shape == mask.shape
    assert np.array_equal(restored, mask)


def test_contract_shape_and_serialization() -> None:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    detections = [
        {"bbox_xyxy_px": [10.0, 20.0, 110.0, 120.0], "class_id": 3},
        {"bbox_xyxy_px": [5.0, 6.0, 25.0, 26.0], "class_id": 7},
    ]
    contract = segment_skill.segment(
        img, detections=detections, segmenter=_FakeSegmenter()
    )

    assert contract["model"] == "injected"
    assert contract["input"]["width"] == 320 and contract["input"]["height"] == 240
    assert contract["output"]["mask_format"] == "coco_rle"
    assert contract["output"]["num_instances"] == 2

    inst = contract["instances"][0]
    assert set(inst) == {"bbox_xyxy_px", "class_id", "area_px", "mask_rle"}
    assert inst["class_id"] == 3
    # 100x100 filled box -> 10000 px, decodes to the same mask
    assert inst["area_px"] == 10000
    decoded = segment_skill.rle_decode(inst["mask_rle"])
    assert decoded.shape == (240, 320)
    assert int(decoded.sum()) == 10000


def test_rejects_non_uint8_array() -> None:
    bad = np.zeros((8, 8, 3), dtype=np.float32)
    try:
        segment_skill.segment(bad, detections=[], segmenter=_FakeSegmenter())
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-uint8 RGB array")


def test_integration_real_pipeline() -> None:
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
    contract = segment_skill.segment(sample, threshold=0.5, model_size="base")
    n = contract["output"]["num_instances"]
    assert contract["model"] == "classical-otsu (interim)"
    assert n == len(contract["instances"])
    for inst in contract["instances"]:
        mask = segment_skill.rle_decode(inst["mask_rle"])
        assert mask.shape == (contract["input"]["height"], contract["input"]["width"])
        assert int(mask.sum()) == inst["area_px"]
    print(f"OK integration: {n} instance masks")


if __name__ == "__main__":
    test_rle_roundtrip()
    print("PASS test_rle_roundtrip")
    test_contract_shape_and_serialization()
    print("PASS test_contract_shape_and_serialization")
    test_rejects_non_uint8_array()
    print("PASS test_rejects_non_uint8_array")
    test_integration_real_pipeline()
    print("DONE")
