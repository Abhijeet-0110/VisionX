# VisionX: Real-Time Object Detection and Recognition System

[![Python Version](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%20CUDA%2012.8-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-YOLO26n-00FFFF.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**VisionX** is a modular, high-performance computer vision system engineered for real-time object detection and recognition. Built using PyTorch, Ultralytics YOLO, OpenCV, and Streamlit, it provides both a low-latency OpenCV **Desktop Application** and an interactive browser-based **Web Dashboard**—both backed by the **same centralized detection and analytics engine**.

---

## Architecture Overview

```text
VISIONX
│
├── Desktop Application (OpenCV Interface: desktop.py)
│
├── Web Application (Streamlit Dashboard: app.py)
│
├── Shared Application Core
│   ├── FrameProcessor (core/processor.py)      <-- Unified orchestrator
│   ├── YOLOObjectDetector (core/detector.py)   <-- GPU/CPU inference
│   ├── DetectionVisualizer (core/visualizer.py)<-- Bounding boxes & HUD
│   └── AnalyticsEngine (core/analytics.py)     <-- Rolling FPS & counters
│
├── Input Management
│   ├── CameraManager (input_sources/camera.py) <-- Webcam capture
│   ├── ImageSource (input_sources/image.py)    <-- Still images
│   └── VideoSource (input_sources/video.py)    <-- Video files
│
└── Models (models/yolo26n.pt)                  <-- Pretrained weights
```

### Key Architectural Principle
Inference logic is **never duplicated**. Whether frames arrive from the local laptop webcam, an uploaded image, a recorded video file, or a browser WebRTC stream, they are processed through `FrameProcessor.process_frame()`, guaranteeing uniform detection thresholds, coordinate scaling, and performance metrics across all user interfaces.

---

## Features

- **⚡ Hardware Acceleration**: Native NVIDIA CUDA acceleration with seamless automatic CPU fallback.
- **👁️ Live Camera Detection**: Real-time object recognition from your laptop webcam with moving-average FPS calculation.
- **🖼️ Still Image Inference**: Upload JPG/PNG images, adjust confidence thresholds interactively, view detailed coordinate tables, and export annotated outputs.
- **🎬 Recorded Video Inference**: Frame-by-frame batch processing of MP4/AVI videos with a real-time progress monitor and downloadable annotated video export.
- **📊 Real-Time Analytics**:
  - Per-frame category counts (e.g., `{"person": 2, "cell phone": 1}`).
  - Rolling moving-average FPS tracker (eliminates 1-frame spikes).
  - High-precision CUDA-synchronized inference latency measurement (in milliseconds).
  - Session-wide cumulative category distribution charts.
- **🎨 Technical Dark Theme**: Charcoal aesthetic with electric blue and cyan telemetry badges.

---

## Verified Hardware & Environment Configuration

| Component | Tested Specification |
| :--- | :--- |
| **Operating System** | Windows 11 (64-bit) |
| **Python** | 3.14.7 in `.venv` |
| **GPU** | NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM) |
| **NVIDIA Driver** | 616.56 |
| **CUDA Runtime** | 12.8 (PyTorch bundled) |
| **PyTorch Build** | `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128` |
| **OpenCV** | 5.0.0 |
| **Ultralytics** | 8.4.197 |

---

## Quickstart Guide

### 1. Clone & Enter Directory
```powershell
cd $HOME\Desktop\projects\VisionX
```

### 2. Activate Virtual Environment
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Verify System Health & CUDA Detection
```powershell
python system_check.py
```
*Expected Output:*
```text
============================================================
              VISIONX ENVIRONMENT DIAGNOSTIC               
============================================================
[Python] Executable: ...\.venv\Scripts\python.exe
[PyTorch] Version : 2.11.0+cu128
[PyTorch] CUDA Available : True
[PyTorch] Detected GPU 0 : NVIDIA GeForce RTX 5050 Laptop GPU
[PyTorch] Total VRAM     : 7.93 GB
[OpenCV] Version : 5.0.0
[Ultralytics] Version : 8.4.197
[Streamlit] Version : 1.66.1
============================================================
```

---

## Running the Applications

### A. Run Desktop Application (OpenCV)
Launch the standalone desktop preview:
```powershell
python desktop.py
```

#### Keyboard Controls:
| Key | Action |
| :--- | :--- |
| **`Q`** or **`ESC`** | Quit application and release camera |
| **`S`** | Capture and save annotated screenshot to `outputs/screenshots/` |
| **`+`** / **`=`** | Increase confidence threshold (+0.05) |
| **`-`** | Decrease confidence threshold (-0.05) |
| **`H`** | Toggle HUD telemetry banner on/off |
| **`B`** | Toggle bounding boxes on/off |
| **`R`** | Reset session FPS and performance analytics |

Optional command-line arguments:
```powershell
python desktop.py --camera-index 0 --conf 0.60 --device cuda
```

---

### B. Run Web Application (Streamlit Dashboard)
Launch the browser dashboard:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

#### Web Navigation Screens:
1. **📊 Dashboard Overview**: System diagnostic status, active device, and architecture summary.
2. **🎥 Live Webcam Detection**: Browser WebRTC or local direct webcam loop with live metrics.
3. **🖼️ Image Detection**: Drag-and-drop image upload, dual-tab visual comparison, class counts table, and annotated image download.
4. **🎬 Video Detection**: Process recorded video files frame-by-frame, preview detection progress, and download rendered MP4 video.
5. **📈 Analytics & Telemetry**: Cumulative detection charts, average throughput FPS, and session summaries.
6. **⚙️ Settings**: Model weights manager, device preference, and storage maintenance.

---

## Running Automated Tests

VisionX includes a comprehensive unit and integration test suite using **pytest**:
```powershell
pytest -v tests
```
*Test Coverage:*
- Model loading & class vocabulary verification
- Empty and corrupt frame validation
- Bounding-box coordinates clamping and schema compliance
- Confidence filtering logic
- Moving average FPS & category count calculations
- Image decoding and VideoWriter creation
- End-to-end `FrameProcessor` pipeline integration

---

## Output Directories

- `outputs/screenshots/`: Timestamped annotated image captures.
- `outputs/videos/`: Processed annotated video exports.
- `models/`: Pretrained model weight files (`.pt`).

---

## Privacy & Security

- **Local Inference**: All camera frames and uploaded media are processed locally in system RAM and GPU VRAM.
- **No External Telemetry**: No camera streams or images are transmitted to external cloud servers.
- **Artifact Control**: Screenshots and video recordings are stored locally on your machine only when you explicitly press Save or Download.
