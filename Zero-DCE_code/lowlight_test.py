import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import platform

def setup_device():
    """
    Detect and return the best available device with priority:
    1. CUDA (NVIDIA GPU)
    2. DirectML (Windows - Intel/AMD/NVIDIA)
    3. Intel XPU (Linux - Intel GPU)
    4. MPS (Apple Silicon)
    5. CPU (Fallback)
    """
    device = torch.device("cpu")
    device_name = "CPU"

    # Check CUDA (NVIDIA GPUs) - All platforms
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA ({torch.cuda.get_device_name()})"
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name()}")

    # Check DirectML (Windows - Intel/AMD/NVIDIA GPUs)
    elif platform.system() == "Windows":
        try:
            import torch_directml
            dml_device = torch_directml.device()
            device = dml_device
            device_name = "DirectML (Windows GPU)"
            print("Using DirectML for GPU acceleration on Windows")
        except ImportError:
            print("torch-directml not available, skipping DirectML")

    # Check Intel XPU (Linux - Intel GPUs)
    elif hasattr(torch, 'xpu') and hasattr(torch.xpu, 'is_available') and torch.xpu.is_available():
        device = torch.device("xpu")
        device_name = f"Intel XPU"
        print(f"Using Intel Integrated GPU")

    # Check MPS (Apple Silicon) - macOS only
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon (MPS)"
        print("Using Apple Silicon MPS")

    else:
        print("Using CPU")

    print(f"Selected device: {device_name}")
    return device

def load_model(device):
    """Load and prepare the model once"""
    # For DirectML, we need to load on CPU first, then move to device
    if device.type == 'privateuseone':
        # DirectML requires special handling
        print("Loading model for DirectML device...")
        # Load to CPU first with weights_only=False
        state_dict = torch.load('snapshots/Epoch99.pth', map_location='cpu', weights_only=False)
        DCE_net = model.enhance_net_nopool()
        DCE_net.load_state_dict(state_dict)
        # Then move to DirectML device
        DCE_net = DCE_net.to(device)
    else:
        # Standard loading for other devices
        DCE_net = model.enhance_net_nopool().to(device)
        DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth', map_location=device, weights_only=False))

    # Intel GPU optimization (if using IPEX and available)
    if device.type == 'xpu':
        try:
            import intel_extension_for_pytorch as ipex
            DCE_net = ipex.optimize(DCE_net)
            print("Applied Intel IPEX optimizations")
        except ImportError:
            print("Intel IPEX not available, running without Intel optimizations")

    # Set to evaluation mode
    DCE_net.eval()

    return DCE_net

def lowlight(image_path, device, model):
    """Process a single image using the pre-initialized device and model"""

    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight)/255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.to(device).unsqueeze(0)

    start = time.time()
    _, enhanced_image, _ = model(data_lowlight)
    end_time = (time.time() - start)

    print(f"Processing time: {end_time:.2f} seconds")

    # Move tensor back to CPU for saving
    enhanced_image = enhanced_image.cpu()

    # Cross-platform path handling
    image_path = image_path.replace('test_data', 'result')
    result_path = image_path
    result_dir = os.path.dirname(result_path)

    # Create directory if it doesn't exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
    # Print system info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Platform details: {platform.platform()}")

    # Setup device and model ONCE
    device = setup_device()
    dce_model = load_model(device)

    print("Device and model initialized. Starting image processing...")

    with torch.no_grad():
        filePath = 'data/test_data/'

        # Cross-platform path handling
        if not os.path.exists(filePath):
            print(f"Error: Test data path '{filePath}' does not exist!")
            sys.exit(1)

        file_list = os.listdir(filePath)

        for file_name in file_list:
            # Cross-platform glob pattern
            pattern = os.path.join(filePath, file_name, "*")
            test_list = glob.glob(pattern)

            for image in test_list:
                print(f"Processing: {image}")
                lowlight(image, device, dce_model)

    print("All images processed!")