import torch
import subprocess
import sys

def check_cuda():
    print("Checking CUDA availability and configuration...")

    # Check if CUDA is available in PyTorch
    print(f"CUDA is available in PyTorch: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Check NVCC version
    try:
        nvcc_output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        print("\nNVCC version:")
        print(nvcc_output)
    except subprocess.CalledProcessError:
        print("\nNVCC not found. Make sure CUDA is installed and in your PATH.")
    except FileNotFoundError:
        print("\nNVCC not found. Make sure CUDA is installed and in your PATH.")

    # Check GPU devices
    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        print("\nGPU Information:")
        print(nvidia_smi_output)
    except subprocess.CalledProcessError:
        print("\nnvidia-smi command failed. Make sure your GPU drivers are installed correctly.")
    except FileNotFoundError:
        print("\nnvidia-smi not found. Make sure your GPU drivers are installed correctly.")

    # Test CUDA with PyTorch
    if torch.cuda.is_available():
        print("\nTesting CUDA with PyTorch:")
        x = torch.rand(5, 3)
        print(f"Random tensor: {x}")
        if torch.cuda.is_available():
            x = x.cuda()
            print(f"Tensor moved to CUDA: {x}")
        else:
            print("Failed to move tensor to CUDA")
    else:
        print("\nCannot test CUDA with PyTorch as it's not available.")

if __name__ == "__main__":
    check_cuda()