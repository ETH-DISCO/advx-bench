import torch


def get_device(enable_mps=False) -> str:
    if torch.backends.mps.is_available() and enable_mps:
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
