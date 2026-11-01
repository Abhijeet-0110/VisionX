"""
VisionX - Central YOLO Object Detection Engine
Loads pretrained YOLO weights once and provides high-performance,
device-accelerated inference with structured detection outputs.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from ultralytics import YOLO

from config.settings import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MODEL_PATH,
    get_default_device,
)
from utils.logger import get_logger

logger = get_logger("VisionX.Detector")


class YOLOObjectDetector:
    """
    Centralized YOLO Object Detection Engine for VisionX.
    
    Responsible for loading the neural network weights into memory once,
    managing GPU/CPU execution contexts, running inference on single frames,
    and transforming raw tensor outputs into structured Python dictionaries.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        img_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        """
        Initialize the YOLO object detection engine.

        Args:
            model_path: Path to the model weights (.pt). Defaults to config.settings.DEFAULT_MODEL_PATH.
            device: 'cuda' or 'cpu'. Defaults to automatic detection.
            conf_threshold: Minimum confidence score [0.0, 1.0] to keep a detection.
            iou_threshold: Intersection over Union threshold for Non-Maximum Suppression (NMS).
            img_size: Image size for model input inference (e.g., 640).
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.img_size = int(img_size)
        self.last_inference_time_ms: float = 0.0

        # Validate and configure inference device
        self.device = self._resolve_device(device)

        # Load model weights
        self.model = self._load_model()

        # Cache class names mapping: int -> str
        self.class_names: Dict[int, str] = self._extract_class_names()

        # Warm up the model to prevent initial latency spikes
        self._warmup()

    def _resolve_device(self, preferred_device: Optional[str]) -> str:
        """
        Resolve and validate the execution device (CUDA vs CPU).
        """
        if preferred_device is None:
            resolved = get_default_device()
        else:
            resolved = preferred_device.lower().strip()

        if resolved.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning(
                    f"CUDA requested ('{resolved}'), but PyTorch reports CUDA is unavailable. "
                    "Falling back to CPU inference."
                )
                return "cpu"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA acceleration active on device: {gpu_name} ({resolved})")
            return resolved

        logger.info("Inference configured on CPU.")
        return "cpu"

    def _load_model(self) -> YOLO:
        """
        Instantiate the Ultralytics YOLO model from disk.
        """
        if not self.model_path.exists():
            logger.warning(
                f"Model weights file not found at '{self.model_path}'. "
                f"Ultralytics will attempt to auto-download '{self.model_path.name}'."
            )

        try:
            logger.info(f"Loading YOLO model from: {self.model_path}")
            model = YOLO(str(self.model_path))
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model from '{self.model_path}': {e}")
            raise RuntimeError(f"Could not initialize YOLO model: {e}") from e

    def _extract_class_names(self) -> Dict[int, str]:
        """
        Extract class ID to class name mapping from the loaded model.
        """
        if hasattr(self.model, "names") and isinstance(self.model.names, dict):
            return {int(k): str(v) for k, v in self.model.names.items()}
        return {}

    def _warmup(self) -> None:
        """
        Execute a single dummy inference pass to allocate GPU tensors and initialize CUDA contexts.
        """
        try:
            logger.info(f"Warming up YOLO model on {self.device.upper()}...")
            dummy_frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            self.model.predict(
                source=dummy_frame,
                device=self.device,
                imgsz=self.img_size,
                verbose=False,
            )
            logger.info("Model warmup completed successfully.")
        except Exception as e:
            logger.warning(f"Model warmup failed (non-critical): {e}")

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform object detection on a single image frame.

        Args:
            frame: Input image as a NumPy BGR or RGB array (H, W, C).
            conf_threshold: Optional override for the confidence threshold.

        Returns:
            List[Dict[str, Any]]: List of structured detection records conforming to:
                {
                    "class_id": int,
                    "class_name": str,
                    "confidence": float,
                    "bbox": [x1, y1, x2, y2]  # Pixel coordinates
                }
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("Empty or invalid frame passed to detect(). Returning empty detections.")
            return []

        active_conf = float(conf_threshold) if conf_threshold is not None else self.conf_threshold

        start_time = time.perf_counter()

        # Run inference
        results = self.model.predict(
            source=frame,
            conf=active_conf,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )

        # Synchronize CUDA if on GPU to get exact inference timing
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

        self.last_inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[Dict[str, Any]] = []

        if not results:
            return detections

        first_result = results[0]
        boxes = first_result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # Extract tensor elements to CPU numpy arrays for parsing
        xyxy_coords = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()

        frame_height, frame_width = frame.shape[:2]

        for i in range(len(boxes)):
            conf = float(confidences[i])
            if conf < active_conf:
                continue

            cls_id = int(class_ids[i])
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

            # Ensure bounding box coordinates are clamped within frame dimensions
            x1 = max(0, min(int(round(xyxy_coords[i][0])), frame_width - 1))
            y1 = max(0, min(int(round(xyxy_coords[i][1])), frame_height - 1))
            x2 = max(0, min(int(round(xyxy_coords[i][2])), frame_width - 1))
            y2 = max(0, min(int(round(xyxy_coords[i][3])), frame_height - 1))

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return detections

    @property
    def model_name(self) -> str:
        """Returns the base filename of the loaded model weights."""
        return self.model_path.name

    @property
    def total_classes(self) -> int:
        """Returns the number of recognizable classes in the model."""
        return len(self.class_names)
