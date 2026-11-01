"""
VisionX - Central Configuration Settings
Defines file paths, detection parameters, hardware device preferences, and UI constants.
"""

from pathlib import Path
import torch

# Base Directory Paths (Resolved relative to project root)
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
CORE_DIR = BASE_DIR / "core"
INPUT_SOURCES_DIR = BASE_DIR / "input_sources"
MODELS_DIR = BASE_DIR / "models"
UTILS_DIR = BASE_DIR / "utils"
TESTS_DIR = BASE_DIR / "tests"
OUTPUTS_DIR = BASE_DIR / "outputs"
SCREENSHOTS_DIR = OUTPUTS_DIR / "screenshots"
VIDEOS_DIR = OUTPUTS_DIR / "videos"

# Ensure output directories exist at runtime
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Default Model Configuration
DEFAULT_MODEL_NAME = "yolo26n.pt"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME
FALLBACK_MODEL_NAME = "yolov8n.pt"
FALLBACK_MODEL_PATH = MODELS_DIR / FALLBACK_MODEL_NAME

# Detection Hyperparameters
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
MIN_CONFIDENCE_THRESHOLD = 0.10
MAX_CONFIDENCE_THRESHOLD = 1.00
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMAGE_SIZE = 640

# Hardware Acceleration
def get_default_device() -> str:
    """
    Returns 'cuda' if NVIDIA GPU acceleration is available via PyTorch,
    otherwise safely falls back to 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_default_device()

# Camera Configuration
DEFAULT_CAMERA_INDEX = 0
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
CAMERA_FPS = 30

# UI Theme & Styling Palette (Dark Theme)
COLOR_BACKGROUND = (26, 26, 26)       # Charcoal / near-black
COLOR_PRIMARY_ACCENT = (255, 128, 0)  # Electric Blue (BGR format for OpenCV: B=255, G=128, R=0)
COLOR_SECONDARY_ACCENT = (255, 255, 0)# Cyan (BGR format for OpenCV)
COLOR_SUCCESS = (50, 205, 50)         # Green status
COLOR_WARNING = (0, 165, 255)         # Amber / Orange
COLOR_ERROR = (0, 0, 255)             # Red status
COLOR_TEXT_PRIMARY = (255, 255, 255)  # White text
COLOR_BOX_DEFAULT = (255, 140, 0)     # Vibrant Cyan/Blue default bbox
