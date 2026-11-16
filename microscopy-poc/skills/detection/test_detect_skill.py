"""Tests for the cell-detection agent skill.

Runs with pytest OR directly:  python test_detect_skill.py

- Unit test (always runs): a fake detector proves the contract/serialization
  without loading a model or using the network.
- Integration test (auto-skips): if `rfdetr` is installed and a sample image is
  present, run the real interim detector and validate the same contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import detect_skill


class _FakeResult:
    """Mimics a supervision.Detections result."""

    def __init__(self) -> None:
        self.xyxy = np.array([[10.0, 20.0, 110.0, 220.0], [5.5, 6.5, 50.5, 60.5]])
        self.confidence = np.array([0.91, 0.42])
        self.class_id = np.array([3, 7])


class _FakeDetector:
    def predict(self, image, threshold: float):  # noqa: ANN001
        return _FakeResult()


def test_contract_shape_and_serialization() -> None:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    contract = detect_skill.detect(img, threshold=0.3, detector=_FakeDetector())

    assert contract["input"] == {
        "format": "RGB",
        "width": 320,
        "height": 240,
        "bit_depth": 8,
        "note": contract["input"]["note"],
    }
    assert contract["output"]["box_coord_space"] == "pixel_xyxy"
    assert contract["output"]["num_detections"] == 2

    first = contract["detections"][0]
    assert set(first) == {"bbox_xyxy_px", "class_id", "confidence"}
    assert first["bbox_xyxy_px"] == [10.0, 20.0, 110.0, 220.0]
    assert isinstance(first["class_id"], int) and first["class_id"] == 3
    assert isinstance(first["confidence"], float) and first["confidence"] == 0.91


def test_rejects_non_uint8_array() -> None:
    bad = np.zeros((8, 8, 3), dtype=np.float32)
    try:
        detect_skill.detect(bad, detector=_FakeDetector())
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-uint8 RGB array")


def test_integration_real_rfdetr() -> None:
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
    contract = detect_skill.detect(sample, threshold=0.5, model_size="base")
    assert contract["model"] == "rfdetr-base"
    assert contract["output"]["fields"] == ["bbox_xyxy_px", "class_id", "confidence"]
    assert contract["output"]["num_detections"] == len(contract["detections"])
    print(f"OK integration: {contract['output']['num_detections']} detections")


if __name__ == "__main__":
    test_contract_shape_and_serialization()
    print("PASS test_contract_shape_and_serialization")
    test_rejects_non_uint8_array()
    print("PASS test_rejects_non_uint8_array")
    test_integration_real_rfdetr()
    print("DONE")
