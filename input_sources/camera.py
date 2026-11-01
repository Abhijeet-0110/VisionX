"""
VisionX - Camera Input Manager
Safely manages webcam initialization, continuous frame acquisition,
device resolution configuration, and resource cleanup.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from config.settings import (
    CAMERA_FPS,
    CAMERA_FRAME_HEIGHT,
    CAMERA_FRAME_WIDTH,
    DEFAULT_CAMERA_INDEX,
)
from utils.logger import get_logger

logger = get_logger("VisionX.Camera")


class CameraManager:
    """
    Manages OpenCV video capture devices (built-in webcam or external USB camera).
    Implements safe resource allocation and deterministic cleanup.
    """

    def __init__(
        self,
        camera_index: int = DEFAULT_CAMERA_INDEX,
        width: int = CAMERA_FRAME_WIDTH,
        height: int = CAMERA_FRAME_HEIGHT,
        target_fps: int = CAMERA_FPS,
    ) -> None:
        """
        Initialize camera parameters.

        Args:
            camera_index: OpenCV device index (default 0 for built-in laptop webcam).
            width: Desired capture frame width.
            height: Desired capture frame height.
            target_fps: Desired capture frame rate.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.capture: Optional[cv2.VideoCapture] = None
        self._is_opened = False

    def start(self) -> bool:
        """
        Open the video capture device and configure requested parameters.

        Returns:
            bool: True if camera initialized successfully, False otherwise.
        """
        if self._is_opened and self.capture is not None and self.capture.isOpened():
            logger.info("Camera is already active.")
            return True

        logger.info(f"Opening camera at index {self.camera_index}...")

        # Use CAP_DSHOW on Windows for faster device acquisition if available
        self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        # Fallback to default backend if DSHOW fails
        if not self.capture.isOpened():
            logger.warning("DirectShow backend failed, trying default OpenCV capture backend...")
            self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            logger.error(
                f"Failed to open camera index {self.camera_index}. "
                "Check if another application is holding the camera or if permissions are denied."
            )
            self._is_opened = False
            return False

        # Set capture properties
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Retrieve actual applied properties
        actual_w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)

        logger.info(f"Camera opened successfully: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
        self._is_opened = True
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next video frame from the capture stream.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (True, frame) if successful, (False, None) otherwise.
        """
        if not self._is_opened or self.capture is None or not self.capture.isOpened():
            return False, None

        ret, frame = self.capture.read()
        if not ret or frame is None:
            logger.warning("Camera stream returned an empty frame or was disconnected.")
            return False, None

        return True, frame

    def release(self) -> None:
        """
        Release hardware camera resources.
        """
        if self.capture is not None:
            logger.info("Releasing camera capture resources...")
            self.capture.release()
            self.capture = None
        self._is_opened = False

    @property
    def is_opened(self) -> bool:
        """Check if the camera is currently opened."""
        return self._is_opened and self.capture is not None and self.capture.isOpened()

    def __enter__(self) -> "CameraManager":
        """Support context management protocol."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure camera is released upon exit from context."""
        self.release()
