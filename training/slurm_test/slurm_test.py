import torch

# Check if CUDA is available
print("CUDA Available:", torch.cuda.is_available())

# List all CUDA devices
if torch.cuda.is_available():
    print("Listing CUDA Devices:")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA devices are available.")
