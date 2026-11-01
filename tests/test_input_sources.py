"""
Unit tests for VisionX input sources (image, camera, and video handling).
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from input_sources.image import load_image_from_bytes, load_image_from_file, save_image
from input_sources.video import create_video_writer


def test_image_load_and_save(tmp_path: Path):
    """Test saving a synthetic image and reloading it."""
    synthetic = np.full((100, 100, 3), 200, dtype=np.uint8)
    test_file = tmp_path / "test_sample.jpg"

    assert save_image(synthetic, test_file) is True
    assert test_file.exists()

    loaded = load_image_from_file(test_file)
    assert loaded is not None
    assert loaded.shape == (100, 100, 3)


def test_image_from_bytes():
    """Test decoding an image from raw memory bytes."""
    synthetic = np.zeros((50, 50, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".png", synthetic)
    byte_data = buffer.tobytes()

    decoded = load_image_from_bytes(byte_data)
    assert decoded is not None
    assert decoded.shape == (50, 50, 3)

    # Empty bytes test
    assert load_image_from_bytes(b"") is None


def test_video_writer_creation(tmp_path: Path):
    """Test initializing a video writer object."""
    output_video = tmp_path / "test_out.mp4"
    writer = create_video_writer(output_video, fps=30.0, width=640, height=480)
    assert writer is not None
    writer.release()
