"""
VisionX - Central Frame Processor Orchestrator
Coordinates detection inference, analytics tracking, and visual annotation
into a single, unified pipeline consumed by both Desktop and Web applications.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from config.settings import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_MODEL_PATH
from core.analytics import AnalyticsEngine
from core.detector import YOLOObjectDetector
from core.visualizer import DetectionVisualizer
from utils.logger import get_logger

logger = get_logger("VisionX.Processor")


class FrameProcessor:
    """
    Unified Pipeline Orchestrator.
    Encapsulates detection, performance metrics, and frame annotation.
    Guarantees consistent inference behavior across Desktop (OpenCV) and Web (Streamlit).
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """
        Initialize the shared pipeline components.

        Args:
            model_path: Path to YOLO weights (.pt).
            device: 'cuda' or 'cpu' (None for auto-detection).
            conf_threshold: Initial confidence threshold.
        """
        logger.info("Initializing VisionX Unified FrameProcessor...")
        self.detector = YOLOObjectDetector(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
        )
        self.visualizer = DetectionVisualizer()
        self.analytics = AnalyticsEngine()
        logger.info("FrameProcessor initialization complete.")

    def process_frame(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        render_boxes: bool = True,
        render_hud: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process a single video frame end-to-end.

        Args:
            frame: Input BGR NumPy array from camera or video.
            conf_threshold: Optional confidence threshold override.
            render_boxes: Whether to draw bounding boxes and labels on output frame.
            render_hud: Whether to render the technical HUD status banner.

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
                - annotated_frame: The rendered BGR image.
                - detections: List of structured detection records.
                - metrics: Dictionary of telemetry statistics (FPS, counts, latency).
        """
        if frame is None or frame.size == 0:
            return frame, [], self.analytics.update([], 0.0)

        # 1. Inference Engine Pass
        detections = self.detector.detect(frame, conf_threshold=conf_threshold)

        # 2. Analytics Telemetry Update
        metrics = self.analytics.update(
            detections=detections,
            inference_time_ms=self.detector.last_inference_time_ms,
        )

        # 3. Visual Annotation
        annotated_frame = frame
        if render_boxes:
            annotated_frame = self.visualizer.draw_detections(annotated_frame, detections)

        if render_hud:
            annotated_frame = self.visualizer.draw_telemetry_hud(
                frame=annotated_frame,
                fps=metrics["fps"],
                inference_time_ms=metrics["inference_time_ms"],
                device=self.detector.device,
                object_counts=metrics["class_counts"],
            )

        return annotated_frame, detections, metrics

    def reset_analytics(self) -> None:
        """Reset historical session metrics."""
        self.analytics.reset()

    @property
    def device(self) -> str:
        """Current inference device."""
        return self.detector.device

    @property
    def model_name(self) -> str:
        """Loaded model filename."""
        return self.detector.model_name
