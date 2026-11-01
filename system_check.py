"""
VisionX - System Diagnostics and Environment Verification
Verifies Python version, PyTorch, CUDA, GPU acceleration, OpenCV, and Ultralytics.
"""

import sys
import os

def check_environment():
    print("=" * 60)
    print("              VISIONX ENVIRONMENT DIAGNOSTIC               ")
    print("=" * 60)

    # 1. Python Environment
    print(f"[Python] Executable: {sys.executable}")
    print(f"[Python] Version   : {sys.version.split()[0]} ({sys.platform})")
    in_venv = sys.prefix != sys.base_prefix
    print(f"[Python] In Virtual Environment: {'YES (.venv)' if in_venv else 'NO (Global Python)'}")
    print("-" * 60)

    # 2. PyTorch & CUDA Check
    try:
        import torch
        print(f"[PyTorch] Version : {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"[PyTorch] CUDA Available : {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024 ** 3)
            print(f"[PyTorch] PyTorch Built CUDA : {cuda_version}")
            print(f"[PyTorch] GPU Count         : {device_count}")
            print(f"[PyTorch] Detected GPU 0    : {gpu_name}")
            print(f"[PyTorch] Total VRAM        : {vram_gb:.2f} GB")
        else:
            print("[PyTorch] Running on CPU only (CUDA not detected in PyTorch build).")
    except ImportError:
        print("[PyTorch] NOT INSTALLED yet.")

    print("-" * 60)

    # 3. OpenCV Check
    try:
        import cv2
        print(f"[OpenCV] Version : {cv2.__version__}")
    except ImportError:
        print("[OpenCV] NOT INSTALLED yet.")

    # 4. Ultralytics YOLO Check
    try:
        import ultralytics
        print(f"[Ultralytics] Version : {ultralytics.__version__}")
    except ImportError:
        print("[Ultralytics] NOT INSTALLED yet.")

    # 5. Streamlit Check
    try:
        import streamlit
        print(f"[Streamlit] Version : {streamlit.__version__}")
    except ImportError:
        print("[Streamlit] NOT INSTALLED yet.")

    print("=" * 60)

if __name__ == "__main__":
    check_environment()
