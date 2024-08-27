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

seed = 41

outpath = Path.cwd() / "data" / "eval" / "eval_cls.csv"
dataset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=True).shuffle(seed=seed)
