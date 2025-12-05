import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version used by PyTorch: {torch.version.cuda}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# This is the most important number:
capability = torch.cuda.get_device_capability(0)
print(f"Compute Capability: {capability[0]}.{capability[1]}")