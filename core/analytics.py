"""
VisionX - Analytics and Performance Engine
Computes real-time FPS, moving average frame rates, per-frame class counts,
and session-wide telemetry metrics.
"""

from collections import Counter, deque
import time
from typing import Any, Dict, List, Optional


class AnalyticsEngine:
    """
    Computes per-frame statistics and maintains rolling averages for frame rates,
    inference latency, and detected object classifications.
    """

    def __init__(self, window_size: int = 30) -> None:
        """
        Initialize the analytics engine.

        Args:
            window_size: Number of recent frames used to compute the smoothed rolling FPS.
        """
        self.window_size = window_size
        self.frame_timestamps = deque(maxlen=window_size)
        self.inference_latencies = deque(maxlen=window_size)
        
        # Session Counters
        self.total_frames_processed: int = 0
        self.session_start_time: float = time.perf_counter()
        self.last_frame_time: Optional[float] = None
        self.peak_detections_in_frame: int = 0
        self.cumulative_class_counts: Counter = Counter()

    def update(
        self,
        detections: List[Dict[str, Any]],
        inference_time_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Record a newly processed frame, update moving averages, and return
        the current telemetry snapshot.

        Args:
            detections: List of detection records in current frame.
            inference_time_ms: Elapsed inference time for this frame in milliseconds.

        Returns:
            Dict[str, Any]: Comprehensive metrics dictionary containing:
                - "fps": Smoothed rolling frames per second
                - "instant_fps": Instantaneous FPS from the last frame interval
                - "average_fps": Lifetime session average FPS
                - "inference_time_ms": Latency for this frame
                - "avg_inference_time_ms": Rolling average inference latency
                - "total_objects": Total objects detected in this frame
                - "class_counts": Per-class count breakdown in this frame {str: int}
                - "total_frames": Total frames processed this session
        """
        now = time.perf_counter()
        self.total_frames_processed += 1

        # Calculate instantaneous FPS
        instant_fps = 0.0
        if self.last_frame_time is not None:
            delta = now - self.last_frame_time
            if delta > 0.0:
                instant_fps = 1.0 / delta

        self.last_frame_time = now
        self.frame_timestamps.append(now)
        self.inference_latencies.append(inference_time_ms)

        # Calculate smoothed rolling FPS
        rolling_fps = 0.0
        if len(self.frame_timestamps) > 1:
            elapsed = self.frame_timestamps[-1] - self.frame_timestamps[0]
            if elapsed > 0.0:
                rolling_fps = (len(self.frame_timestamps) - 1) / elapsed

        # Calculate session lifetime average FPS
        session_elapsed = now - self.session_start_time
        lifetime_avg_fps = (
            self.total_frames_processed / session_elapsed if session_elapsed > 0 else 0.0
        )

        # Compute per-frame object counts
        current_counts = Counter()
        for det in detections:
            cls_name = det.get("class_name", "unknown")
            current_counts[cls_name] += 1
            self.cumulative_class_counts[cls_name] += 1

        total_current_objects = sum(current_counts.values())
        if total_current_objects > self.peak_detections_in_frame:
            self.peak_detections_in_frame = total_current_objects

        avg_inference_ms = (
            sum(self.inference_latencies) / len(self.inference_latencies)
            if self.inference_latencies
            else inference_time_ms
        )

        return {
            "fps": round(rolling_fps, 1),
            "instant_fps": round(instant_fps, 1),
            "average_fps": round(lifetime_avg_fps, 1),
            "inference_time_ms": round(inference_time_ms, 1),
            "avg_inference_time_ms": round(avg_inference_ms, 1),
            "total_objects": total_current_objects,
            "class_counts": dict(current_counts),
            "total_frames": self.total_frames_processed,
        }

    def reset(self) -> None:
        """Reset session statistics and clear historical windows."""
        self.frame_timestamps.clear()
        self.inference_latencies.clear()
        self.total_frames_processed = 0
        self.session_start_time = time.perf_counter()
        self.last_frame_time = None
        self.peak_detections_in_frame = 0
        self.cumulative_class_counts.clear()

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Produce a high-level summary of the entire session.
        """
        now = time.perf_counter()
        total_time = max(0.001, now - self.session_start_time)
        return {
            "total_frames": self.total_frames_processed,
            "elapsed_seconds": round(total_time, 2),
            "overall_avg_fps": round(self.total_frames_processed / total_time, 1),
            "peak_detections_in_frame": self.peak_detections_in_frame,
            "cumulative_classes_detected": dict(self.cumulative_class_counts),
        }
