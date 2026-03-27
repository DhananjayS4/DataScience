import torch
import sys

print("--- GPU Diagnostic ---")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Current Device: {torch.cuda.current_device()}")
    
    # Test a small operation on GPU
    try:
        x = torch.tensor([1.0, 2.0]).to("cuda")
        print("✅ Success: Successfully moved a tensor to GPU!")
    except Exception as e:
        print(f"❌ Error: Failed to use GPU even though it is 'available'.")
        print(f"Technical Error: {e}")
else:
    print("❌ Error: CUDA is not available to PyTorch.")
    print("Suggestions:")
    print("1. Ensure NVIDIA drivers are up to date.")
    print("2. Reinstall torch with the correct CUDA version if needed.")
