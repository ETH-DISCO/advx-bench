from pathlib import Path
from PIL import Image
import random
import secrets

import numpy as np
import torch


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


def get_random_kodak_image() -> Image.Image:
    img_idx = random.randint(1, 24)
    cwd = Path(__file__).parent.parent / 'data' / 'kodak' / f'kodim{img_idx:02d}.png'
    return Image.open(cwd)

random_img = get_random_kodak_image()
random_img.show()
