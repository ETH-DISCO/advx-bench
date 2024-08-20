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
from torchvision.models import inception_v3
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize
from tqdm import tqdm

from models.utils import set_seed
from metrics.metrics import get_fid, get_kid, get_time_result


def get_imagenet_generator(size: int, seed: int = 41):
    subset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=True).take(size).shuffle(seed=seed)
    for elem in subset:
        yield elem["image_id"], elem["image"].convert("RGB"), elem["label"], elem["caption_enriched"]


def get_imagenet_label(idx: int) -> str:
    datapath = Path.cwd() / "data" / "imagenet_labels.json"
    data = json.loads(datapath.read_text())
    return data[str(idx)]



if __name__ == "__main__":
    set_seed(41)

    config = {
        "outpath": Path.cwd() / "data" / "eval" / "eval_cls.csv",
        "fidkidpath": Path.cwd() / "data" / "eval" / "eval_cls_fidkid.csv",
        "subset_size": 25,
    }
    config["outpath"].unlink(missing_ok=True)
    config["fidkidpath"].unlink(missing_ok=True)

    dataset = get_imagenet_generator(size=config["subset_size"])

    # fid/kid data
    inception = inception_v3(pretrained=True, transform_input=False).eval()
    inception_transform = Compose([Resize(299), CenterCrop(299), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    real_features = []
    fake_features = []

    for id, image, label_id, caption in tqdm(dataset, total=config["subset_size"]):
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

        imgnet_normalize = lambda x: (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        metrics = {
            # time
            "time_compression": time_compression,
            "time_decompression": time_decompression,
            
            # semantic similarity
            "latent_cosine_similarity": F.cosine_similarity(codec.encode(x)["latent"].view(1, -1), codec.encode(x_hat)["latent"].view(1, -1)).item(),
            "psnr": (20 * torch.log10(1.0 / torch.sqrt(torch.mean((x - x_hat) ** 2)))).item(),
            "ssim": structural_similarity(np.array(x.squeeze().permute(1, 2, 0).cpu().numpy()), np.array(x_hat.squeeze().permute(1, 2, 0).cpu().numpy()), multichannel=True, channel_axis=2, data_range=1.0),
            
            # accuracy
            "img_id": id,
            "ground_truth": label_id,
            "ground_truth_label": get_imagenet_label(label_id),
            "ground_truth_caption": caption,
            "x_preds_top5": json.dumps(dict(zip(resnet.predict_top_k(imgnet_normalize(x), 5), F.softmax(resnet.model(x), dim=1).squeeze().tolist()))),
            "x_hat_preds_top5": json.dumps(dict(zip(resnet.predict_top_k(imgnet_normalize(x_hat), 5), F.softmax(resnet.model(x_hat), dim=1).squeeze().tolist()))),
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
