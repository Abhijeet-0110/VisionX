"""
VisionX - Video Input Source & Processing Manager
Handles opening, reading metadata, frame extraction, and annotated video saving.
"""

from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger("VisionX.VideoSource")


class VideoSource:
    """
    Manages reading recorded video files, inspecting metadata,
    and reading sequential frames.
    """

    def __init__(self, video_path: Union[str, Path]) -> None:
        """
        Initialize video source with target file path.
        """
        self.video_path = Path(video_path)
        self.capture: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self._open()

    def _open(self) -> None:
        """Open the video file and inspect properties."""
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file not found at: {self.video_path}")

        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video file: {self.video_path}")

        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or 30.0
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Opened video '{self.video_path.name}': {self.width}x{self.height} @ "
            f"{self.fps:.2f} FPS ({self.total_frames} total frames)"
        )

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator yielding (frame_index, frame) for each frame in the video.
        """
        if self.capture is None or not self.capture.isOpened():
            return

        frame_idx = 0
        while True:
            ret, frame = self.capture.read()
            if not ret or frame is None:
                break
            yield frame_idx, frame
            frame_idx += 1

    def release(self) -> None:
        """Release video resources."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        logger.info(f"Released video capture: {self.video_path.name}")

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def create_video_writer(
    output_path: Union[str, Path],
    fps: float,
    width: int,
    height: int,
    fourcc: str = "mp4v",
) -> cv2.VideoWriter:
    """
    Factory function to initialize a cv2.VideoWriter instance.

    Args:
        output_path: Destination path for output video.
        fps: Target frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fourcc: 4-character codec string ('mp4v', 'avc1', 'XVID').

    Returns:
        cv2.VideoWriter: Initialized video writer.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(out_file), fourcc_code, fps, (width, height))
    return writer
