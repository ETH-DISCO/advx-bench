import torch

def get_device(enable_mps=False):
    if torch.backends.mps.is_available() and enable_mps:
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
