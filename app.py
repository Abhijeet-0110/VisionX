"""
VisionX - Streamlit Web Dashboard
Professional Computer Vision and Object Recognition Web Interface.
Shares the exact same FrameProcessor as the Desktop application.
"""

from datetime import datetime
from pathlib import Path
import tempfile
import time
from typing import Dict, List, Optional
import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_webrtc import (
        RTCConfiguration,
        VideoProcessorBase,
        WebRtcMode,
        webrtc_streamer,
    )
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

from config.settings import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    SCREENSHOTS_DIR,
    VIDEOS_DIR,
    get_default_device,
)
from core.processor import FrameProcessor
from input_sources.camera import CameraManager
from input_sources.image import load_image_from_bytes, save_image
from input_sources.video import VideoSource, create_video_writer
from utils.logger import get_logger

logger = get_logger("VisionX.Web")

# Set Page Config
st.set_page_config(
    page_title="VisionX - AI Object Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Charcoal Dark Theme, Electric Blue, Cyan accents)
CUSTOM_CSS = """
<style>
    /* Dark Theme Base Styling */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #00d2ff !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Headings and Accents */
    h1, h2, h3 {
        color: #f0f6fc !important;
        font-weight: 600 !important;
    }
    
    .visionx-title {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .badge-cuda {
        background-color: #10b981;
        color: #042f2e;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.75rem;
    }

    .badge-cpu {
        background-color: #f59e0b;
        color: #451a03;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.75rem;
    }

    /* Primary Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #1E88E5 0%, #00acc1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_shared_processor(
    model_name: str,
    device: str,
    conf_threshold: float,
) -> FrameProcessor:
    """
    Cached singleton factory for the shared detection engine.
    Ensures the YOLO weights are loaded once in memory across page re-renders.
    """
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        model_path = DEFAULT_MODEL_PATH
    return FrameProcessor(
        model_path=model_path,
        device=device,
        conf_threshold=conf_threshold,
    )


# Sidebar Configuration & Navigation
with st.sidebar:
    st.markdown('<div class="visionx-title">VISIONX</div>', unsafe_allow_html=True)
    st.caption("Real-Time Object Detection & Recognition")
    st.markdown("---")

    navigation = st.radio(
        "Navigation",
        [
            "📊 Dashboard Overview",
            "🎥 Live Webcam Detection",
            "🖼️ Image Detection",
            "🎬 Video Detection",
            "📈 Analytics & Telemetry",
            "⚙️ Settings",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("Inference Settings")
    
    # Model Selector
    available_models = [f.name for f in MODELS_DIR.glob("*.pt")]
    if not available_models:
        available_models = [DEFAULT_MODEL_NAME]
    
    selected_model = st.selectbox(
        "Detection Model",
        available_models,
        index=0,
    )

    # Hardware Device Preference
    default_dev = get_default_device()
    selected_device = st.selectbox(
        "Inference Hardware",
        ["cuda", "cpu"] if default_dev == "cuda" else ["cpu"],
        index=0,
        help="CUDA leverages your NVIDIA GPU; CPU runs on system processor threads.",
    )

    # Confidence Threshold Slider
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05,
        help="Filter detections below this confidence score.",
    )

    # Visualization Options
    show_hud = st.checkbox("Show Performance HUD", value=True)
    show_boxes = st.checkbox("Show Bounding Boxes", value=True)

    st.markdown("---")
    # Hardware Status Pill
    if selected_device == "cuda":
        st.markdown('Hardware: <span class="badge-cuda">CUDA ACCELERATED</span>', unsafe_allow_html=True)
    else:
        st.markdown('Hardware: <span class="badge-cpu">CPU MODE</span>', unsafe_allow_html=True)

# Initialize the shared processor instance
processor = get_shared_processor(selected_model, selected_device, conf_threshold)


# ==========================================
# 1. DASHBOARD OVERVIEW
# ==========================================
if navigation == "📊 Dashboard Overview":
    st.markdown("## 📊 VisionX System Overview")
    st.markdown(
        "VisionX is a high-performance computer vision system combining deep neural networks "
        "with hardware-accelerated real-time inference. Both the desktop and web interfaces "
        "run on the **same unified detection engine**."
    )

    # Status Cards Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Model Architecture", selected_model.replace(".pt", "").upper())
    with c2:
        st.metric("Active Device", processor.device.upper())
    with c3:
        st.metric("Recognizable Classes", f"{processor.detector.total_classes} Categories")
    with c4:
        st.metric("Default Confidence", f"{int(conf_threshold * 100)}%")

    st.markdown("---")
    st.markdown("### Core Capabilities")

    col_a, col_b = st.columns(2)
    with col_a:
        st.info(
            "**🎥 Live Video Streaming**\n\n"
            "Captures video from your webcam, runs GPU tensor inference, and displays annotated "
            "bounding boxes, class labels, and confidence percentages in real time."
        )
        st.success(
            "**🖼️ Still Image Detection**\n\n"
            "Upload high-resolution images, adjust detection thresholds on the fly, inspect class "
            "breakdowns, and export publication-ready annotated images."
        )
    with col_b:
        st.warning(
            "**🎬 Recorded Video Processing**\n\n"
            "Process recorded video footage frame-by-frame, monitor throughput FPS, and generate "
            "annotated MP4 video outputs."
        )
        st.info(
            "**📈 Performance Analytics**\n\n"
            "Auditable telemetry measuring frame processing intervals, inference latency in "
            "milliseconds, moving average FPS, and object category distributions."
        )


# ==========================================
# 2. LIVE WEBCAM DETECTION
# ==========================================
elif navigation == "🎥 Live Webcam Detection":
    st.markdown("## 🎥 Real-Time Live Webcam Detection")
    st.markdown(
        "Select your preferred streaming mode below. "
        "**Browser WebRTC** captures through your web browser, while **Local OpenCV Stream** "
        "accesses the built-in webcam directly."
    )

    stream_mode = st.radio(
        "Stream Mode",
        ["Browser WebRTC Stream", "Local OpenCV Stream"],
        horizontal=True,
    )

    if stream_mode == "Browser WebRTC Stream":
        if not WEBRTC_AVAILABLE:
            st.error("streamlit-webrtc package is not installed.")
        else:
            class VisionXVideoTransformer(VideoProcessorBase):
                def __init__(self) -> None:
                    self.processor = processor

                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img = frame.to_ndarray(format="bgr24")
                    annotated, _, _ = self.processor.process_frame(
                        frame=img,
                        conf_threshold=conf_threshold,
                        render_boxes=show_boxes,
                        render_hud=show_hud,
                    )
                    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

            webrtc_streamer(
                key="visionx-live-stream",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=VisionXVideoTransformer,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

    else:
        # Local OpenCV Stream Loop
        col_ctrl, col_view = st.columns([1, 4])
        with col_ctrl:
            start_btn = st.button("▶️ Start Camera", use_container_width=True)
            stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)
            snap_btn = st.button("📸 Capture Screenshot", use_container_width=True)

        with col_view:
            frame_placeholder = st.empty()
            metric_placeholder = st.empty()

        if start_btn:
            st.session_state["local_camera_active"] = True

        if stop_btn:
            st.session_state["local_camera_active"] = False

        if st.session_state.get("local_camera_active", False):
            camera = CameraManager()
            if not camera.start():
                st.error("Failed to open local camera. Ensure webcam permissions are allowed and not in use.")
                st.session_state["local_camera_active"] = False
            else:
                try:
                    while st.session_state.get("local_camera_active", False):
                        ret, raw_frame = camera.read_frame()
                        if not ret or raw_frame is None:
                            break

                        annotated, dets, metrics = processor.process_frame(
                            frame=raw_frame,
                            conf_threshold=conf_threshold,
                            render_boxes=show_boxes,
                            render_hud=show_hud,
                        )

                        # Convert BGR to RGB for Streamlit rendering
                        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        metric_placeholder.markdown(
                            f"**FPS:** `{metrics['fps']:.1f}` | "
                            f"**Inference:** `{metrics['inference_time_ms']:.1f}ms` | "
                            f"**Objects Detected:** `{metrics['total_objects']}`"
                        )
                finally:
                    camera.release()


# ==========================================
# 3. IMAGE DETECTION
# ==========================================
elif navigation == "🖼️ Image Detection":
    st.markdown("## 🖼️ Still Image Detection & Inspection")
    st.markdown("Upload any image file (JPG, PNG, JPEG) to run object recognition.")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        bgr_image = load_image_from_bytes(file_bytes)

        if bgr_image is not None:
            # Process using shared FrameProcessor
            annotated_bgr, detections, metrics = processor.process_frame(
                frame=bgr_image,
                conf_threshold=conf_threshold,
                render_boxes=show_boxes,
                render_hud=show_hud,
            )

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Detections", metrics["total_objects"])
            with m2:
                st.metric("Inference Latency", f"{metrics['inference_time_ms']:.1f} ms")
            with m3:
                st.metric("Hardware Device", processor.device.upper())
            with m4:
                st.metric("Image Resolution", f"{bgr_image.shape[1]}x{bgr_image.shape[0]}")

            # Visual Comparison Tabs
            tab_ann, tab_orig = st.tabs(["🎯 Annotated Result", "📷 Original Image"])
            with tab_ann:
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)
            with tab_orig:
                orig_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                st.image(orig_rgb, use_container_width=True)

            # Detailed Detection Table
            if detections:
                st.markdown("### 📋 Detected Objects Breakdown")
                table_data = []
                for idx, det in enumerate(detections, 1):
                    table_data.append(
                        {
                            "ID": idx,
                            "Class": det["class_name"].capitalize(),
                            "Confidence": f"{det['confidence'] * 100:.1f}%",
                            "Bounding Box (X1, Y1, X2, Y2)": str(det["bbox"]),
                        }
                    )
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)

                # Download Annotated Image
                _, encoded_img = cv2.imencode(".jpg", annotated_bgr)
                st.download_button(
                    label="💾 Download Annotated Image",
                    data=encoded_img.tobytes(),
                    file_name=f"visionx_detected_{uploaded_file.name}",
                    mime="image/jpeg",
                )
            else:
                st.warning("No objects recognized above the current confidence threshold.")


# ==========================================
# 4. VIDEO DETECTION
# ==========================================
elif navigation == "🎬 Video Detection":
    st.markdown("## 🎬 Recorded Video Detection")
    st.markdown("Upload a video to analyze frames and export an annotated output video.")

    uploaded_video = st.file_uploader(
        "Upload a video file (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        tfile.close()

        video_source = VideoSource(tfile.name)
        st.info(
            f"Video Info: `{video_source.width}x{video_source.height}` @ "
            f"`{video_source.fps:.1f} FPS` ({video_source.total_frames} total frames)"
        )

        if st.button("🚀 Process Video Frames"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            preview_placeholder = st.empty()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_video_path = VIDEOS_DIR / f"visionx_annotated_{timestamp}.mp4"

            writer = create_video_writer(
                output_path=output_video_path,
                fps=video_source.fps,
                width=video_source.width,
                height=video_source.height,
            )

            total = max(1, video_source.total_frames)
            start_proc_time = time.perf_counter()

            for idx, frame in video_source.frames():
                annotated_frame, dets, metrics = processor.process_frame(
                    frame=frame,
                    conf_threshold=conf_threshold,
                    render_boxes=show_boxes,
                    render_hud=show_hud,
                )
                writer.write(annotated_frame)

                # Update preview every 10 frames
                if idx % 10 == 0 or idx == total - 1:
                    rgb_preview = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    preview_placeholder.image(rgb_preview, use_container_width=True)
                    progress = min(1.0, (idx + 1) / total)
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Processing frame {idx + 1}/{total} | "
                        f"Inference: {metrics['inference_time_ms']:.1f}ms | Objects: {metrics['total_objects']}"
                    )

            writer.release()
            video_source.release()
            total_elapsed = time.perf_counter() - start_proc_time
            avg_throughput = total / max(0.001, total_elapsed)

            st.success(
                f"✅ Video processing complete in {total_elapsed:.1f}s "
                f"({avg_throughput:.1f} FPS throughput). Saved to `{output_video_path.name}`."
            )

            if output_video_path.exists():
                with open(output_video_path, "rb") as vf:
                    st.download_button(
                        label="💾 Download Annotated Video",
                        data=vf.read(),
                        file_name=output_video_path.name,
                        mime="video/mp4",
                    )


# ==========================================
# 5. ANALYTICS & TELEMETRY
# ==========================================
elif navigation == "📈 Analytics & Telemetry":
    st.markdown("## 📈 Performance & Telemetry Analytics")
    summary = processor.analytics.get_session_summary()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Frames Processed", summary["total_frames"])
    with col2:
        st.metric("Session Elapsed Time", f"{summary['elapsed_seconds']:.1f} s")
    with col3:
        st.metric("Average System FPS", f"{summary['overall_avg_fps']:.1f}")

    st.markdown("---")
    st.markdown("### Cumulative Class Distribution")

    class_counts = summary["cumulative_classes_detected"]
    if class_counts:
        df_classes = pd.DataFrame(
            list(class_counts.items()), columns=["Object Category", "Total Count"]
        ).sort_values(by="Total Count", ascending=False)

        st.bar_chart(df_classes.set_index("Object Category"))
        st.dataframe(df_classes, use_container_width=True)
    else:
        st.info("No detections recorded in this session yet. Run Live, Image, or Video detection to generate telemetry.")

    if st.button("🔄 Reset Analytics Session"):
        processor.reset_analytics()
        st.success("Session analytics reset successfully.")


# ==========================================
# 6. SETTINGS
# ==========================================
elif navigation == "⚙️ Settings":
    st.markdown("## ⚙️ VisionX Application Settings")

    st.markdown("### Model Information")
    st.code(
        f"Weights File : {processor.model_name}\n"
        f"Active Device: {processor.device.upper()}\n"
        f"Classes Count: {processor.detector.total_classes}\n"
        f"Base Path    : {MODELS_DIR}",
        language="yaml",
    )

    st.markdown("### Output Storage Locations")
    st.write(f"- **Screenshots:** `{SCREENSHOTS_DIR}`")
    st.write(f"- **Annotated Videos:** `{VIDEOS_DIR}`")

    if st.button("🧹 Clear Temporary Screenshots and Videos"):
        for f in SCREENSHOTS_DIR.glob("*.jpg"):
            f.unlink()
        for f in VIDEOS_DIR.glob("*.mp4"):
            f.unlink()
        st.success("Cleaned generated outputs.")
