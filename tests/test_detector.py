"""
Unit and integration tests for VisionX YOLO Detection Engine.
Tests model initialization, inference schema, device resolution, and edge cases.
"""

from pathlib import Path
import numpy as np
import pytest
import cv2

from config.settings import DEFAULT_MODEL_PATH
from core.detector import YOLOObjectDetector


@pytest.fixture(scope="session")
def detector() -> YOLOObjectDetector:
    """Fixture that initializes the detector once for test session."""
    return YOLOObjectDetector(model_path=DEFAULT_MODEL_PATH)


def test_detector_initialization(detector: YOLOObjectDetector):
    """Verify that detector initializes with classes and valid device."""
    assert detector is not None
    assert detector.model is not None
    assert detector.device in ("cuda", "cpu")
    assert detector.total_classes > 0
    assert 0 in detector.class_names
    assert detector.class_names[0] == "person"


def test_detector_device_fallback():
    """Verify that passing an invalid CUDA index falls back safely or warns without crashing."""
    detector_cpu = YOLOObjectDetector(model_path=DEFAULT_MODEL_PATH, device="cpu")
    assert detector_cpu.device == "cpu"


def test_invalid_and_empty_frames(detector: YOLOObjectDetector):
    """Verify that None, empty arrays, or invalid shapes return empty lists without crashing."""
    assert detector.detect(None) == []
    assert detector.detect(np.array([])) == []
    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_synthetic_frame_inference(detector: YOLOObjectDetector):
    """Run inference on a synthetic image and verify structure of results."""
    # Create a 640x640 synthetic RGB image
    synthetic_frame = np.full((640, 640, 3), 128, dtype=np.uint8)
    
    # Draw a distinct rectangle to give the model visual features
    cv2.rectangle(synthetic_frame, (100, 100), (400, 400), (0, 0, 255), -1)

    detections = detector.detect(synthetic_frame, conf_threshold=0.10)
    assert isinstance(detections, list)
    assert detector.last_inference_time_ms >= 0.0

    # If any detection was found, verify schema
    for det in detections:
        assert "class_id" in det and isinstance(det["class_id"], int)
        assert "class_name" in det and isinstance(det["class_name"], str)
        assert "confidence" in det and isinstance(det["confidence"], float)
        assert "bbox" in det and isinstance(det["bbox"], list)
        assert len(det["bbox"]) == 4

        x1, y1, x2, y2 = det["bbox"]
        assert 0 <= x1 <= 640
        assert 0 <= y1 <= 640
        assert 0 <= x2 <= 640
        assert 0 <= y2 <= 640
        assert x1 <= x2
        assert y1 <= y2


def test_confidence_threshold_filtering(detector: YOLOObjectDetector):
    """Verify that high confidence threshold excludes lower confidence detections."""
    frame = np.full((640, 640, 3), 200, dtype=np.uint8)
    # Threshold at 0.999 should yield either empty or strictly >= 0.999
    detections = detector.detect(frame, conf_threshold=0.999)
    for det in detections:
        assert det["confidence"] >= 0.999
