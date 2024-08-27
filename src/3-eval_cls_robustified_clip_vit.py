import os
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


"""
validation
"""

seed = 41

outpath = Path.cwd() / "data" / "eval" / "eval_cls.csv"
dataset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=True).shuffle(seed=seed)

file_name = "robustified_clip_vit.pth"
model_id = "sueszli/robustified_clip_vit"
downloaded_file_path = hf_hub_download(repo_id=model_id, filename=file_name)
