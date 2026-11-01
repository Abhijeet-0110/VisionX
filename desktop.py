"""
VisionX - Desktop Application
Real-Time Object Detection using OpenCV and the shared VisionX FrameProcessor.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import cv2
import numpy as np

from config.settings import (
    COLOR_PRIMARY_ACCENT,
    COLOR_SUCCESS,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    SCREENSHOTS_DIR,
)
from core.processor import FrameProcessor
from input_sources.camera import CameraManager
from input_sources.image import save_image
from utils.logger import get_logger

logger = get_logger("VisionX.Desktop")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the desktop application."""
    parser = argparse.ArgumentParser(
        description="VisionX - Real-Time Object Detection & Recognition (Desktop)"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help="Camera device index (default: 0 for built-in webcam)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold between 0.10 and 1.00 (default: 0.50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cuda' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to YOLO model weights file (.pt)",
    )
    return parser.parse_args()


def run_desktop_app(
    camera_index: int = DEFAULT_CAMERA_INDEX,
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    device: str = None,
    model_path: str = str(DEFAULT_MODEL_PATH),
) -> None:
    """
    Main execution loop for the VisionX Desktop Application.
    """
    print("\n" + "=" * 65)
    print("           VISIONX - REAL-TIME OBJECT DETECTION (DESKTOP)        ")
    print("=" * 65)
    print(f"[*] Camera Index        : {camera_index}")
    print(f"[*] Model Weights       : {model_path}")
    print(f"[*] Initial Confidence  : {conf_threshold:.2f}")
    print("=" * 65)
    print("Controls:")
    print("  [Q] or [ESC] : Quit application")
    print("  [S]          : Save annotated screenshot to outputs/screenshots/")
    print("  [+] / [=]    : Increase confidence threshold (+0.05)")
    print("  [-]          : Decrease confidence threshold (-0.05)")
    print("  [H]          : Toggle HUD telemetry banner")
    print("  [B]          : Toggle Bounding Boxes")
    print("  [R]          : Reset performance analytics")
    print("=" * 65 + "\n")

    # Initialize the shared detection pipeline
    try:
        processor = FrameProcessor(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
        )
    except Exception as e:
        logger.error(f"Failed to initialize detection pipeline: {e}")
        print(f"\n[ERROR] Detection pipeline initialization failed: {e}")
        sys.exit(1)

    # Initialize camera manager
    camera = CameraManager(camera_index=camera_index)
    if not camera.start():
        logger.error(f"Cannot access camera index {camera_index}.")
        print(f"\n[ERROR] Could not open camera {camera_index}.")
        print("[!] Troubleshooting Checklist:")
        print("  1. Verify your laptop webcam privacy slider/shutter is open.")
        print("  2. Ensure no other application (Zoom, Teams, Browser) is using the camera.")
        print("  3. Check Windows Camera Privacy Settings (Settings -> Privacy & Security -> Camera).")
        sys.exit(1)

    window_name = "VisionX - Real-Time Object Detection (OpenCV Interface)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    current_conf = conf_threshold
    show_hud = True
    show_boxes = True
    notification_msg: str = ""
    notification_expiry: float = 0.0

    try:
        while True:
            success, raw_frame = camera.read_frame()
            if not success or raw_frame is None:
                logger.warning("Dropped or empty camera frame. Retrying...")
                # Show an error placeholder frame if stream disconnects
                error_canvas = np.full((720, 1280, 3), 30, dtype=np.uint8)
                cv2.putText(
                    error_canvas,
                    "CAMERA STREAM INTERRUPTED - WAITING...",
                    (320, 360),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow(window_name, error_canvas)
                if cv2.waitKey(100) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                continue

            # Process frame using the shared VisionX core
            annotated_frame, detections, metrics = processor.process_frame(
                frame=raw_frame,
                conf_threshold=current_conf,
                render_boxes=show_boxes,
                render_hud=show_hud,
            )

            # Draw temporary on-screen notification if active
            import time
            current_time = time.time()
            if notification_msg and current_time < notification_expiry:
                cv2.putText(
                    annotated_frame,
                    notification_msg,
                    (20, annotated_frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.65,
                    COLOR_SUCCESS,
                    2,
                    lineType=cv2.LINE_AA,
                )

            # Display frame
            cv2.imshow(window_name, annotated_frame)

            # Handle user keyboard input
            key = cv2.waitKey(1) & 0xFF

            # Exit condition
            if key in (ord("q"), ord("Q"), 27):
                logger.info("Exit command received. Shutting down...")
                break

            # Save Screenshot
            elif key in (ord("s"), ord("S")):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_filename = f"visionx_screenshot_{timestamp}.jpg"
                screenshot_path = SCREENSHOTS_DIR / screenshot_filename
                if save_image(annotated_frame, screenshot_path):
                    notification_msg = f"[SAVED] Screenshot: {screenshot_filename}"
                    notification_expiry = current_time + 3.0
                    print(f"[*] Screenshot saved: {screenshot_path}")

            # Increase confidence threshold
            elif key in (ord("+"), ord("=")):
                current_conf = min(0.95, round(current_conf + 0.05, 2))
                notification_msg = f"Confidence Threshold: {int(current_conf * 100)}%"
                notification_expiry = current_time + 2.0

            # Decrease confidence threshold
            elif key in (ord("-"), ord("_")):
                current_conf = max(0.10, round(current_conf - 0.05, 2))
                notification_msg = f"Confidence Threshold: {int(current_conf * 100)}%"
                notification_expiry = current_time + 2.0

            # Toggle HUD
            elif key in (ord("h"), ord("H")):
                show_hud = not show_hud
                notification_msg = f"HUD Display: {'ENABLED' if show_hud else 'DISABLED'}"
                notification_expiry = current_time + 2.0

            # Toggle Bounding Boxes
            elif key in (ord("b"), ord("B")):
                show_boxes = not show_boxes
                notification_msg = f"Bounding Boxes: {'ENABLED' if show_boxes else 'DISABLED'}"
                notification_expiry = current_time + 2.0

            # Reset Analytics
            elif key in (ord("r"), ord("R")):
                processor.reset_analytics()
                notification_msg = "Performance Analytics Reset"
                notification_expiry = current_time + 2.0

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        # Deterministic resource cleanup
        camera.release()
        cv2.destroyAllWindows()
        print("\n[*] VisionX Desktop Application closed cleanly.")


if __name__ == "__main__":
    args = parse_arguments()
    run_desktop_app(
        camera_index=args.camera_index,
        conf_threshold=args.conf,
        device=args.device,
        model_path=args.model,
    )
