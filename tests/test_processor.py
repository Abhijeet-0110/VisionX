"""
Integration tests for VisionX FrameProcessor.
Verifies end-to-end frame processing, annotation, and telemetry outputs.
"""

import numpy as np
import pytest
from core.processor import FrameProcessor


@pytest.fixture(scope="session")
def processor() -> FrameProcessor:
    """Fixture that initializes the shared processor once."""
    return FrameProcessor()


def test_processor_initialization(processor: FrameProcessor):
    """Verify processor components are correctly linked."""
    assert processor.detector is not None
    assert processor.visualizer is not None
    assert processor.analytics is not None
    assert processor.device in ("cuda", "cpu")


def test_process_empty_frame(processor: FrameProcessor):
    """Verify empty frame returns safe defaults."""
    frame, dets, metrics = processor.process_frame(None)
    assert frame is None
    assert dets == []
    assert "fps" in metrics


def test_process_valid_frame(processor: FrameProcessor):
    """Verify processing a synthetic frame produces an annotated image and metrics."""
    synthetic = np.full((480, 640, 3), 100, dtype=np.uint8)
    annotated, dets, metrics = processor.process_frame(synthetic, conf_threshold=0.5)

    assert annotated is not None
    assert annotated.shape == (480, 640, 3)
    assert isinstance(dets, list)
    assert "fps" in metrics
    assert "inference_time_ms" in metrics
    assert "total_objects" in metrics
    assert "class_counts" in metrics
