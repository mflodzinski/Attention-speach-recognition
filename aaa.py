import torch
from warp_rnnt import rnnt_loss
# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    print("CUDA (GPU) is available.")
    # Additional information about the GPU(s)
    print("GPU device count:", torch.cuda.device_count())
    print("Current GPU device:", torch.cuda.current_device())
    print("GPU device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA (GPU) is not available.")

print(len("she_had_your_dark_suit_in_greasy_wash_water_all_year"))