import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets import load_dataset
from skimage.metrics import structural_similarity
from tqdm import tqdm

from metrics.metrics import get_fid, get_inception_features, get_kid
from models.utils import set_seed


def get_imagenet_generator(size: int, seed: int = 41):
    subset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=True).take(size).shuffle(seed=seed)  # type: ignore
    for elem in subset:
        yield elem["image_id"], elem["image"].convert("RGB"), elem["label"], elem["caption_enriched"]  # type: ignore


def get_imagenet_label(idx: int) -> str:
    datapath = Path.cwd() / "data" / "imagenet_labels.json"
    data = json.loads(datapath.read_text())
    return data[str(idx)]


def get_advx(x: torch.Tensor, config: dict) -> torch.Tensor:
    # apply
    return x


if __name__ == "__main__":
    set_seed(41)

    config = {
        "outpath": Path.cwd() / "data" / "eval" / "eval_cls.csv",
        "fidkidpath": Path.cwd() / "data" / "eval" / "eval_cls_fidkid.csv",
        "subset_size": 25,
        # advx config
        # ...
    }
    config["outpath"].unlink(missing_ok=True)
    config["fidkidpath"].unlink(missing_ok=True)

    dataset = get_imagenet_generator(size=config["subset_size"])

    x_features = []
    advx_features = []

    for id, image, label_id, caption in tqdm(dataset, total=config["subset_size"]):
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])

        x: torch.Tensor = transform(image).unsqueeze(0)
        advx: torch.Tensor = get_advx(x, config)

        x_features.append(get_inception_features(x))
        advx_features.append(get_inception_features(advx))

        imgnet_normalize = lambda x: (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    #     metrics = {
    #         # semantic similarity
    #         # "latent_cosine_similarity": F.cosine_similarity(codec.encode(x)["latent"].view(1, -1), codec.encode(x_hat)["latent"].view(1, -1)).item(), ---> write function to get latents
    #         "psnr": (20 * torch.log10(1.0 / torch.sqrt(torch.mean((x - x_hat) ** 2)))).item(),
    #         "ssim": structural_similarity(np.array(x.squeeze().permute(1, 2, 0).cpu().numpy()), np.array(x_hat.squeeze().permute(1, 2, 0).cpu().numpy()), multichannel=True, channel_axis=2, data_range=1.0),
    #         # accuracy
    #         "img_id": id,
    #         "ground_truth": label_id,
    #         "ground_truth_label": get_imagenet_label(label_id),
    #         "ground_truth_caption": caption,
    #         "x_preds_top5": json.dumps(dict(zip(resnet.predict_top_k(imgnet_normalize(x), 5), F.softmax(resnet.model(x), dim=1).squeeze().tolist()))),
    #         "x_hat_preds_top5": json.dumps(dict(zip(resnet.predict_top_k(imgnet_normalize(x_hat), 5), F.softmax(resnet.model(x_hat), dim=1).squeeze().tolist()))),
    #     }
    #     with open(outpath, mode="a") as f:
    #         writer = csv.DictWriter(f, fieldnames=metrics.keys())
    #         if outpath.stat().st_size == 0:
    #             writer.writeheader()
    #         writer.writerow(metrics)

    # with open(fidkidpath, mode="a") as f:
    #     metrics = {
    #         "quality": quality,
    #         "fid": get_fid(x_features, advx_features),
    #         "kid": get_kid(x_features, advx_features, config["subset_size"]),
    #     }
    #     writer = csv.DictWriter(f, fieldnames=metrics.keys())
    #     if fidkidpath.stat().st_size == 0:
    #         writer.writeheader()
    #     writer.writerow(metrics)
