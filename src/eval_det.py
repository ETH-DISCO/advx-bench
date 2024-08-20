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


def get_coco_generator(size: int, seed: int):
    subset = load_dataset("detection-datasets/coco", split="val", streaming=True).take(size).shuffle(seed=41)
    for elem in subset:
        yield elem["image_id"], elem["image"].convert("RGB"), elem["objects"]["bbox"], elem["objects"]["category"]


def get_coco_label(idx: int) -> str:
    datapath = Path.cwd() / "data" / "coco_labels.json"
    data = json.loads(datapath.read_text())
    return data[str(idx)]


if __name__ == "__main__":
    set_seed(41)

    outpath = Path.cwd() / "eval" / "eval_det.csv"
    fidkidpath = Path.cwd() / "eval" / "eval_det_fidkid.csv"
    outpath.unlink(missing_ok=True)
    fidkidpath.unlink(missing_ok=True)

    for quality in range(1, 9):
        codec = Bmshj2018Codec(quality=quality)
        subset_size = 25
        dataset = CocoDataset().get_generator(size=subset_size)
        det_model = YOLODetector()

        inception = inception_v3(pretrained=True, transform_input=False).eval()
        inception_transform = Compose([Resize(299), CenterCrop(299), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        real_features = []
        fake_features = []

        for id, image, boxes, labels in tqdm(dataset, total=subset_size):
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
            with torch.no_grad():
                real_feature = inception(inception_transform(x)).squeeze().cpu().numpy()
                fake_feature = inception(inception_transform(x_hat)).squeeze().cpu().numpy()
                real_features.append(real_feature)
                fake_features.append(fake_feature)

            def filter_detections(probs, boxes, labels, min_prob=0.5, max_prob=0.95):
                filtered = [(prob, box, label) for prob, box, label in zip(probs, boxes, labels) if min_prob <= prob <= max_prob]
                if not filtered:
                    return [], [], []
                return map(list, zip(*filtered))

            x_boxes, x_probs, x_labels = det_model.detect(ToPILImage()(x_hat.squeeze()))
            x_hat_boxes, x_hat_probs, x_hat_labels = det_model.detect(ToPILImage()(x.squeeze()))
            x_probs_50_95, x_boxes_50_95, x_labels_50_95 = filter_detections(x_probs, x_boxes, x_labels)
            x_hat_probs_50_95, x_hat_boxes_50_95, x_hat_labels_50_95 = filter_detections(x_hat_probs, x_hat_boxes, x_hat_labels)

            metrics = {
                # compression
                "quality": quality,
                "bpp": compressed["bpp"],
                "bitrate": torch.sum(torch.log2(codec.encode(x)["likelihoods"])).item() / (x.size(2) * x.size(3)),
                # time
                "time_compression": time_compression,
                "time_decompression": time_decompression,
                # semantic similarity
                "latent_cosine_similarity": F.cosine_similarity(codec.encode(x)["latent"].view(1, -1), codec.encode(x_hat)["latent"].view(1, -1)).item(),
                "psnr": (20 * torch.log10(1.0 / torch.sqrt(torch.mean((x - x_hat) ** 2)))).item(),
                "ssim": structural_similarity(np.array(x.squeeze().permute(1, 2, 0).cpu().numpy()), np.array(x_hat.squeeze().permute(1, 2, 0).cpu().numpy()), multichannel=True, channel_axis=2, data_range=1.0),
                # accuracy
                "img_id": id,
                "ground_truth_labels": labels,
                "ground_truth_boxes": boxes,
                "ap_x": average_precision_score([1 if label in labels else 0 for label in x_labels], x_probs) if len(x_labels) > 0 else 0.0,
                "ap_x_hat": average_precision_score([1 if label in labels else 0 for label in x_hat_labels], x_hat_probs) if len(x_hat_labels) > 0 else 0.0,
                "ap_x_50_95": average_precision_score([1 if label in labels else 0 for label in x_labels_50_95], x_probs_50_95) if len(x_labels_50_95) > 0 else 0.0,
                "ap_x_hat_50_95": average_precision_score([1 if label in labels else 0 for label in x_hat_labels_50_95], x_hat_probs_50_95) if len(x_hat_labels_50_95) > 0 else 0.0,
                "iou_x": max([get_iou(box, box_) for box in boxes for box_ in x_boxes]) if len(x_boxes) > 0 else 0.0,
                "iou_x_hat": max([get_iou(box, box_) for box in boxes for box_ in x_hat_boxes]) if len(x_hat_boxes) > 0 else 0.0,
            }
            with open(outpath, mode="a") as f:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                if outpath.stat().st_size == 0:
                    writer.writeheader()
                writer.writerow(metrics)

        with open(fidkidpath, mode="a") as f:
            metrics = {
                "quality": quality,
                "fid": get_fid(real_features, fake_features),
                "kid": get_kid(real_features, fake_features),
            }
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if fidkidpath.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(metrics)
