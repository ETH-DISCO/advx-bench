import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets import load_dataset
from skimage.metrics import structural_similarity
from sklearn.metrics import average_precision_score
from torchvision.models import inception_v3
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToPILImage
from tqdm import tqdm

from models.utils import set_seed

import json
from pathlib import Path

from datasets import load_dataset

import csv
import gc
import itertools
import json
import os
import random
from pathlib import Path

import spacy
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from advx.masks import get_diamond_mask
from advx.perturb import get_fgsm_clipvit_imagenet
from advx.utils import add_overlay
from metrics.metrics import get_cosine_similarity, get_psnr, get_ssim
from models.cls import classify_clip
from utils import get_device

torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow TF32 on cudnn
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"














# def is_cached(path: Path, entry_id: dict) -> bool:
#     entry_id = entry_id.copy()

#     if not path.exists():
#         return False

#     with open(path, mode="r") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             if all(row[key] == str(value) for key, value in entry_id.items()):
#                 return True
#     return False


def get_coco_label(idx: int) -> str:
    datapath = Path.cwd() / "data" / "coco_labels.json"
    data = json.loads(datapath.read_text())
    return data[str(idx)]


def get_coco_labels() -> list[str]:
    datapath = Path.cwd() / "data" / "coco_labels.json"
    data = json.loads(datapath.read_text())
    return list(data.values())




def get_advx(img: Image.Image, label_id: int, combination: dict) -> Image.Image:
    combination = combination.copy()
    
    get_diamond_overlay = lambda img: add_overlay(img, overlay=get_diamond_mask(diamond_count=15, diamonds_per_row=10), opacity=160)
    img = get_diamond_overlay(img)

    return img



"""
config
"""


CONFIG = {
    "outpath": Path.cwd() / "data" / "eval" / "eval_cls.csv",
    "subset_size": 5,
}
COMBINATIONS = {
    # most effective from previous experiments
    # ...
}

"""
eval loop
"""

random_combinations = list(itertools.product(*COMBINATIONS.values()))
random.shuffle(random_combinations)
print(f"total iterations: {len(random_combinations)} * {CONFIG['subset_size']} = {len(random_combinations) * CONFIG['subset_size']}")

dataset = load_dataset("detection-datasets/coco", split="val", streaming=True).take(CONFIG["subset_size"]).shuffle(seed=41)
dataset = list(map(lambda x: (x["image_id"], x["image"].convert("RGB"), x["objects"]["category"], x["objects"]["caption"]), dataset))

if get_device() == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

for elem in dataset:
    print(elem)
    break

# for combination in tqdm(random_combinations, total=len(random_combinations)):
#     combination = dict(zip(COMBINATIONS.keys(), combination))

#     for id, x_image, label_id, caption in dataset:
#         entry_id = {
#             **combination,
#             "img_id": id,
#         }
#         if is_cached(CONFIG["outpath"], entry_id):
#             print(f"skipping {entry_id}")
#             continue

#         with torch.no_grad(), torch.amp.autocast(device_type=get_device(disable_mps=True), enabled="cuda" == get_device()):
#             advx_image = get_advx(x_image, label_id, combination)

#             transform = transforms.Compose([transforms.Resize((256, 256)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
#             x: torch.Tensor = transform(x_image).unsqueeze(0)
#             advx_x: torch.Tensor = transform(advx_image).unsqueeze(0)

#             def get_acc_boolmask(img: Image.Image) -> list[bool]:
#                 preds = list(zip(range(len(labels)), classify_clip(img, labels)))
#                 preds.sort(key=lambda x: x[1], reverse=True)
#                 top5_keys, top5_vals = zip(*preds[:5])
#                 top5_mask = [label_id == key for key in top5_keys]
#                 return top5_mask

#             x_acc5 = get_acc_boolmask(x_image)
#             advx_acc5 = get_acc_boolmask(advx_image)

#         results = {
#             **entry_id,
#             # semantic similarity
#             "cosine_sim": get_cosine_similarity(x_image, advx_image),
#             "psnr": get_psnr(x, advx_x),
#             "ssim": get_ssim(x, advx_x),
#             # accuracy
#             "label": get_imagenet_label(label_id),
#             "x_acc1": 1 if x_acc5[0] else 0,
#             "advx_acc1": 1 if advx_acc5[0] else 0,
#             "x_acc5": 1 if any(x_acc5) else 0,
#             "advx_acc5": 1 if any(advx_acc5) else 0,
#         }

#         with open(CONFIG["outpath"], mode="a") as f:
#             writer = csv.DictWriter(f, fieldnames=results.keys())
#             if CONFIG["outpath"].stat().st_size == 0:
#                 writer.writeheader()
#             writer.writerow(results)

#         torch.cuda.empty_cache()
#         gc.collect()











# set_seed(41)

# for quality in range(1, 9):
#     codec = Bmshj2018Codec(quality=quality)
#     subset_size = 25
#     dataset = CocoDataset().get_generator(size=subset_size)
#     det_model = YOLODetector()

#     inception = inception_v3(pretrained=True, transform_input=False).eval()
#     inception_transform = Compose([Resize(299), CenterCrop(299), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#     real_features = []
#     fake_features = []

#     for id, image, boxes, labels in tqdm(dataset, total=subset_size):
#         transform = transforms.Compose(
#             [
#                 transforms.Resize((256, 256)),
#                 transforms.Grayscale(num_output_channels=3),  # grayscale to rgb
#                 transforms.ToTensor(),
#             ]
#         )
#         x = transform(image).unsqueeze(0)

#         time_start = time.time()
#         compressed = codec.compress(x)
#         time_compression = time.time() - time_start
#         x_hat = codec.decompress(compressed)
#         time_decompression = time.time() - time_start - time_compression
#         with torch.no_grad():
#             real_feature = inception(inception_transform(x)).squeeze().cpu().numpy()
#             fake_feature = inception(inception_transform(x_hat)).squeeze().cpu().numpy()
#             real_features.append(real_feature)
#             fake_features.append(fake_feature)

#         def filter_detections(probs, boxes, labels, min_prob=0.5, max_prob=0.95):
#             filtered = [(prob, box, label) for prob, box, label in zip(probs, boxes, labels) if min_prob <= prob <= max_prob]
#             if not filtered:
#                 return [], [], []
#             return map(list, zip(*filtered))

#         x_boxes, x_probs, x_labels = det_model.detect(ToPILImage()(x_hat.squeeze()))
#         x_hat_boxes, x_hat_probs, x_hat_labels = det_model.detect(ToPILImage()(x.squeeze()))
#         x_probs_50_95, x_boxes_50_95, x_labels_50_95 = filter_detections(x_probs, x_boxes, x_labels)
#         x_hat_probs_50_95, x_hat_boxes_50_95, x_hat_labels_50_95 = filter_detections(x_hat_probs, x_hat_boxes, x_hat_labels)

#         metrics = {
#             # compression
#             "quality": quality,
#             "bpp": compressed["bpp"],
#             "bitrate": torch.sum(torch.log2(codec.encode(x)["likelihoods"])).item() / (x.size(2) * x.size(3)),
#             # time
#             "time_compression": time_compression,
#             "time_decompression": time_decompression,
#             # semantic similarity
#             "latent_cosine_similarity": F.cosine_similarity(codec.encode(x)["latent"].view(1, -1), codec.encode(x_hat)["latent"].view(1, -1)).item(),
#             "psnr": (20 * torch.log10(1.0 / torch.sqrt(torch.mean((x - x_hat) ** 2)))).item(),
#             "ssim": structural_similarity(np.array(x.squeeze().permute(1, 2, 0).cpu().numpy()), np.array(x_hat.squeeze().permute(1, 2, 0).cpu().numpy()), multichannel=True, channel_axis=2, data_range=1.0),
#             # accuracy
#             "img_id": id,
#             "ground_truth_labels": labels,
#             "ground_truth_boxes": boxes,
#             "ap_x": average_precision_score([1 if label in labels else 0 for label in x_labels], x_probs) if len(x_labels) > 0 else 0.0,
#             "ap_x_hat": average_precision_score([1 if label in labels else 0 for label in x_hat_labels], x_hat_probs) if len(x_hat_labels) > 0 else 0.0,
#             "ap_x_50_95": average_precision_score([1 if label in labels else 0 for label in x_labels_50_95], x_probs_50_95) if len(x_labels_50_95) > 0 else 0.0,
#             "ap_x_hat_50_95": average_precision_score([1 if label in labels else 0 for label in x_hat_labels_50_95], x_hat_probs_50_95) if len(x_hat_labels_50_95) > 0 else 0.0,
#             "iou_x": max([get_iou(box, box_) for box in boxes for box_ in x_boxes]) if len(x_boxes) > 0 else 0.0,
#             "iou_x_hat": max([get_iou(box, box_) for box in boxes for box_ in x_hat_boxes]) if len(x_hat_boxes) > 0 else 0.0,
#         }
#         with open(outpath, mode="a") as f:
#             writer = csv.DictWriter(f, fieldnames=metrics.keys())
#             if outpath.stat().st_size == 0:
#                 writer.writeheader()
#             writer.writerow(metrics)

#     with open(fidkidpath, mode="a") as f:
#         metrics = {
#             "quality": quality,
#             "fid": get_fid(real_features, fake_features),
#             "kid": get_kid(real_features, fake_features),
#         }
#         writer = csv.DictWriter(f, fieldnames=metrics.keys())
#         if fidkidpath.stat().st_size == 0:
#             writer.writeheader()
#         writer.writerow(metrics)
