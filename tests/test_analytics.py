"""
Unit tests for VisionX AnalyticsEngine.
Tests FPS calculation, per-frame counting, and session statistics.
"""

import time
import pytest
from core.analytics import AnalyticsEngine


def test_analytics_initialization():
    """Verify engine initializes with clean counters."""
    engine = AnalyticsEngine()
    assert engine.total_frames_processed == 0
    assert engine.peak_detections_in_frame == 0
    assert len(engine.frame_timestamps) == 0


def test_object_counting():
    """Verify category counts are accurate for each frame."""
    engine = AnalyticsEngine()
    fake_detections = [
        {"class_name": "person", "confidence": 0.95},
        {"class_name": "cup", "confidence": 0.88},
        {"class_name": "person", "confidence": 0.72},
        {"class_name": "laptop", "confidence": 0.85},
    ]

    metrics = engine.update(fake_detections, inference_time_ms=15.2)

    assert metrics["total_objects"] == 4
    assert metrics["class_counts"]["person"] == 2
    assert metrics["class_counts"]["cup"] == 1
    assert metrics["class_counts"]["laptop"] == 1
    assert metrics["inference_time_ms"] == 15.2
    assert engine.total_frames_processed == 1


def test_fps_calculation():
    """Verify moving average FPS computation across multiple frame updates."""
    engine = AnalyticsEngine(window_size=10)
    
    # Simulate 5 frames with 20ms delays (~50 FPS)
    for _ in range(5):
        time.sleep(0.02)
        metrics = engine.update([], inference_time_ms=5.0)

    assert metrics["fps"] > 0.0
    assert metrics["average_fps"] > 0.0
    assert metrics["total_frames"] == 5


def test_analytics_reset():
    """Verify resetting analytics clears all state."""
    engine = AnalyticsEngine()
    engine.update([{"class_name": "chair", "confidence": 0.9}], inference_time_ms=10.0)
    assert engine.total_frames_processed == 1

    engine.reset()
    assert engine.total_frames_processed == 0
    assert len(engine.frame_timestamps) == 0
    assert len(engine.cumulative_class_counts) == 0
