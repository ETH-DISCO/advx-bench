import csv
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from skimage.metrics import structural_similarity
from torchvision.transforms import ToPILImage
from tqdm import tqdm


"""
dataset
"""


def get_coco_generator(self, size: int) -> Iterator[tuple[Image.Image, list[float], list[float], list[str]]]:
    dataset = load_dataset("detection-datasets/coco", split="val", streaming=True)
    
    def _coco_id_to_name(coco_id: int) -> str:
        mapping = { 0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush" }
        return mapping[coco_id]

    for i, elem in enumerate(dataset):
        if i == size:
            break
        image = elem["image"].convert("RGB")
        boxes = elem["objects"]["bbox"]
        probs = [1.0] * len(boxes)  # max confidence
        labels = elem["objects"]["category"]

        labels = [_coco_id_to_name(label) for label in labels]
        yield image, boxes, probs, labels


"""
utils
"""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_iou(box1: list[float], box2: list[float]) -> float:
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xi1, yi1 = max(x1, x1_), max(y1, y1_)
    xi2, yi2 = min(x2, x2_), min(y2, y2_)
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    return intersection / (box1_area + box2_area - intersection)


"""
main
"""


outpath = Path.cwd() / "eval" / "eval_det.csv"

if __name__ == "__main__":
    set_seed(41)

    if outpath.exists():
        outpath.unlink()

    for quality in range(1, 9):
        dataset_size = 25
        dataset = get_coco_generator(dataset_size)

        codec = Bmshj2018Codec(quality=quality)
        det_model = YOLODetector()

        for image, boxes, probs, labels in tqdm(dataset):
            transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.Grayscale(num_output_channels=3),  # grayscale to rgb
                    transforms.ToTensor(),
                ]
            )
            x = transform(image).unsqueeze(0)

            time_start = time.time()
            compressed = codec.compress(x)
            time_compression = time.time() - time_start
            x_hat = codec.decompress(compressed)
            time_decompression = time.time() - time_start - time_compression

            x_boxes, x_probs, x_labels = det_model.detect(ToPILImage()(x_hat.squeeze()))
            x_hat_boxes, x_hat_probs, x_hat_labels = det_model.detect(ToPILImage()(x.squeeze()))

            # filter by probability for mAP@50:95

            metrics = {
                "quality": quality,
                "time_compression": time_compression,
                "time_decompression": time_decompression,
                "ssim": structural_similarity(np.array(x.squeeze().permute(1, 2, 0).cpu().numpy()), np.array(x_hat.squeeze().permute(1, 2, 0).cpu().numpy()), multichannel=True, channel_axis=2, data_range=1.0),
                "bpp": compressed["bpp"],
                "latent_cosine_similarity": F.cosine_similarity(codec.encode(x)["latent"].view(1, -1), codec.encode(x_hat)["latent"].view(1, -1)).item(),
                "psnr": (20 * torch.log10(1.0 / torch.sqrt(torch.mean((x - x_hat) ** 2)))).item(),
                "bitrate": torch.sum(torch.log2(codec.encode(x)["likelihoods"])).item() / (x.size(2) * x.size(3)),

                # "ap_x": average_precision_score([1 if label in labels else 0 for label in x_labels], x_probs) if len(x_labels) > 0 else 0.0,
                # "ap_x_hat": average_precision_score([1 if label in labels else 0 for label in x_hat_labels], x_hat_probs) if len(x_hat_labels) > 0 else 0.0,
                # "iou_x": max([get_iou(box, box_) for box in boxes for box_ in x_boxes]) if len(x_boxes) > 0 else 0.0,
                # "iou_x_hat": max([get_iou(box, box_) for box in boxes for box_ in x_hat_boxes]) if len(x_hat_boxes) > 0 else 0.0,
            }

            with open(outpath, mode="a") as f:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                if outpath.stat().st_size == 0:
                    writer.writeheader()
                writer.writerow(metrics)
