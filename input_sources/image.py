"""
VisionX - Image Input Source Manager
Handles reading, validating, and saving still image inputs from disk or memory buffers.
"""

from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger("VisionX.ImageSource")


def load_image_from_file(file_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load an image from a local file path into a BGR NumPy array.

    Args:
        file_path: Local path to the image file.

    Returns:
        Optional[np.ndarray]: Loaded BGR image, or None if reading failed.
    """
    path = Path(file_path)
    if not path.is_file():
        logger.error(f"Image file does not exist at: {path}")
        return None

    try:
        # cv2.imread loads in BGR
        image = cv2.imread(str(path))
        if image is None or image.size == 0:
            logger.error(f"Failed to decode image from: {path}")
            return None
        logger.info(f"Loaded image '{path.name}' ({image.shape[1]}x{image.shape[0]}, {image.shape[2]} channels)")
        return image
    except Exception as e:
        logger.error(f"Error reading image '{path}': {e}")
        return None


def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode an image from raw memory bytes (such as from Streamlit file uploader).

    Args:
        image_bytes: Raw bytes of the image file (JPEG, PNG, etc.).

    Returns:
        Optional[np.ndarray]: Decoded BGR NumPy array, or None if invalid.
    """
    if not image_bytes:
        logger.warning("Empty byte buffer received in load_image_from_bytes().")
        return None

    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            logger.error("Failed to decode image from byte buffer.")
            return None
        return image
    except Exception as e:
        logger.error(f"Error decoding image bytes: {e}")
        return None


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> bool:
    """
    Save a NumPy BGR image to the specified output path.

    Args:
        image: BGR NumPy array.
        output_path: Target filesystem path (.jpg, .png).

    Returns:
        bool: True if successfully written, False otherwise.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        logger.error("Cannot save invalid or empty image.")
        return False

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        success = cv2.imwrite(str(out_file), image)
        if success:
            logger.info(f"Saved image to: {out_file}")
            return True
        logger.error(f"cv2.imwrite failed for: {out_file}")
        return False
    except Exception as e:
        logger.error(f"Exception saving image to '{out_file}': {e}")
        return False
