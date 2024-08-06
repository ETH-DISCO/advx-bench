import random
import secrets
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int = -1) -> None:
    if seed == -1:
        seed = secrets.randbelow(1_000_000_000)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_kodak_img(idx: int) -> Image.Image:
    assert 1 <= idx <= 24
    cwd = Path(__file__).parent.parent / "data" / "kodak" / f"kodim{idx:02d}.png"
    return Image.open(cwd)
