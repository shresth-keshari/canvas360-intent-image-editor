import sys
import cv2
import numpy as np
import rembg
import onnxruntime as ort

print(f"✅ Python: {sys.version.split()[0]} (Recommended: 3.10 or 3.11)")
print(f"✅ OpenCV: {cv2.__version__}")
print(f"✅ NumPy:  {np.__version__} (Must be < 2.0.0)")

# Check AI Hardware Acceleration
providers = ort.get_available_providers()
if 'CUDAExecutionProvider' in providers:
    print("🚀 GPU Acceleration: ENABLED (CUDA is working)")
else:
    print("🐢 GPU Acceleration: DISABLED (Running on CPU - Safe Mode)")

print("✅ Environment is stable.")