"""
VisionX - Detection Visualizer and Annotation Engine
Draws high-contrast bounding boxes, class labels, confidence scores,
and telemetry overlays (FPS, device, object counts) conforming to the UI design standards.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from config.settings import (
    COLOR_BACKGROUND,
    COLOR_ERROR,
    COLOR_PRIMARY_ACCENT,
    COLOR_SECONDARY_ACCENT,
    COLOR_SUCCESS,
    COLOR_TEXT_PRIMARY,
)


class DetectionVisualizer:
    """
    Renders visual detections and performance statistics on image frames.
    """

    # Distinct high-contrast palette for different object classes (BGR format)
    PALETTE: List[Tuple[int, int, int]] = [
        (255, 128, 0),   # Electric Blue (Cyan-blue)
        (0, 255, 255),   # Yellow
        (50, 205, 50),   # Lime Green
        (255, 0, 128),   # Magenta
        (0, 165, 255),   # Vibrant Orange
        (255, 255, 0),   # Cyan
        (180, 105, 255), # Hot Pink
        (255, 191, 0),   # Deep Sky Blue
        (0, 215, 255),   # Gold
        (147, 20, 255),  # Deep Pink
    ]

    def __init__(
        self,
        box_thickness: int = 2,
        font_scale: float = 0.55,
        font_thickness: int = 1,
    ) -> None:
        """
        Initialize visualization parameters.
        """
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Return a deterministic color for a given class ID."""
        return self.PALETTE[class_id % len(self.PALETTE)]

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        draw_labels: bool = True,
        draw_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes and class labels for all detected objects.

        Args:
            frame: BGR NumPy array.
            detections: List of detection records conforming to schema.
            draw_labels: Whether to render class names.
            draw_confidence: Whether to append confidence percentage.

        Returns:
            np.ndarray: Annotated BGR frame.
        """
        if frame is None or frame.size == 0:
            return frame

        annotated = frame.copy()

        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            class_id = det.get("class_id", 0)
            class_name = det.get("class_name", f"Class {class_id}")
            confidence = det.get("confidence", 0.0)

            color = self._get_class_color(class_id)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.box_thickness)

            if not draw_labels:
                continue

            # Format label text
            if draw_confidence:
                label_text = f"{class_name} {int(confidence * 100)}%"
            else:
                label_text = class_name

            # Calculate text dimensions
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, self.font, self.font_scale, self.font_thickness
            )

            # Position label background tag above the box if room allows, else inside
            tag_y1 = max(0, y1 - text_h - 6)
            tag_y2 = tag_y1 + text_h + 6
            tag_x2 = min(annotated.shape[1] - 1, x1 + text_w + 8)

            # Draw filled tag background
            cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), color, -1)

            # Text label in high-contrast black or white
            # For brighter palette colors, dark text is most legible
            text_color = (0, 0, 0)
            cv2.putText(
                annotated,
                label_text,
                (x1 + 4, tag_y2 - baseline - 2),
                self.font,
                self.font_scale,
                text_color,
                self.font_thickness,
                lineType=cv2.LINE_AA,
            )

        return annotated

    def draw_telemetry_hud(
        self,
        frame: np.ndarray,
        fps: float = 0.0,
        inference_time_ms: float = 0.0,
        device: str = "CPU",
        object_counts: Optional[Dict[str, int]] = None,
        title: str = "VISIONX AI",
    ) -> np.ndarray:
        """
        Draw a technical heads-up display (HUD) badge on the top edge of the frame.
        """
        if frame is None or frame.size == 0:
            return frame

        annotated = frame.copy()
        frame_h, frame_w = annotated.shape[:2]

        # Draw top semi-transparent HUD banner
        hud_height = 42
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (frame_w, hud_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Bottom border for HUD banner
        cv2.line(annotated, (0, hud_height), (frame_w, hud_height), COLOR_PRIMARY_ACCENT, 1)

        # Left Item: Project Branding Badge
        cv2.putText(
            annotated,
            title,
            (14, 27),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            COLOR_SECONDARY_ACCENT,
            1,
            lineType=cv2.LINE_AA,
        )

        # Center Items: FPS and Latency
        device_upper = device.upper()
        device_color = COLOR_SUCCESS if "CUDA" in device_upper else COLOR_WARNING
        metrics_text = f"FPS: {fps:.1f}  |  Inference: {inference_time_ms:.1f}ms  |  "
        
        cv2.putText(
            annotated,
            metrics_text,
            (160, 26),
            self.font,
            0.50,
            COLOR_TEXT_PRIMARY,
            1,
            lineType=cv2.LINE_AA,
        )

        # Device badge
        (metrics_w, _), _ = cv2.getTextSize(metrics_text, self.font, 0.50, 1)
        cv2.putText(
            annotated,
            f"Device: {device_upper}",
            (160 + metrics_w, 26),
            self.font,
            0.50,
            device_color,
            1,
            lineType=cv2.LINE_AA,
        )

        # Right Item: Total detections counter
        total_objects = sum(object_counts.values()) if object_counts else 0
        counter_text = f"Objects: {total_objects}"
        (c_w, _), _ = cv2.getTextSize(counter_text, self.font, 0.55, 1)
        cv2.putText(
            annotated,
            counter_text,
            (frame_w - c_w - 15, 27),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            COLOR_PRIMARY_ACCENT,
            1,
            lineType=cv2.LINE_AA,
        )

        # Bottom Left: Category breakdown pills if objects exist
        if object_counts:
            pill_y = frame_h - 16
            pill_x = 14
            breakdown_str = "Detections: " + ", ".join(f"{k}: {v}" for k, v in object_counts.items())
            
            # Background pill
            (b_w, b_h), _ = cv2.getTextSize(breakdown_str, self.font, 0.45, 1)
            pill_overlay = annotated.copy()
            cv2.rectangle(
                pill_overlay,
                (pill_x - 6, pill_y - b_h - 6),
                (pill_x + b_w + 6, pill_y + 4),
                (20, 20, 20),
                -1,
            )
            cv2.addWeighted(pill_overlay, 0.70, annotated, 0.30, 0, annotated)
            cv2.putText(
                annotated,
                breakdown_str,
                (pill_x, pill_y - 2),
                self.font,
                0.45,
                COLOR_SECONDARY_ACCENT,
                1,
                lineType=cv2.LINE_AA,
            )

        return annotated
