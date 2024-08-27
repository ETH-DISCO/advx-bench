import os
from pathlib import Path

import torch
from datasets import load_dataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


"""
validation
"""

# decide whether to train or validate based on whether the .pth file exists

outpath = Path.cwd() / "data" / "eval" / "eval_cls.csv"
dataset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=True).shuffle(seed=seed)
