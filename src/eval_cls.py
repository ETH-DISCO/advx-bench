import csv
import io
import json
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from compressai.zoo import bmshj2018_factorized
from datasets import load_dataset
from PIL import Image
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import polynomial_kernel
from torchvision.models import inception_v3
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize
from tqdm import tqdm


"""
codecs
"""


class CodecInterface(ABC):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> dict:
        pass

    @abstractmethod
    def decode(self, bneck_obj: dict) -> dict:
        pass

    @abstractmethod
    def compress(self, x: torch.Tensor) -> dict:
        pass

    @abstractmethod
    def decompress(self, compressed_data: dict) -> torch.Tensor:
        pass


class Bmshj2018Codec(CodecInterface):
    # .encode and .decode don't reduce the image size in bytes but translate the image to a latent space and back.
    # .compress and .decompress are the ones that return the size reduced image in bytes.

    def __init__(self, quality):
        kwargs = {
            "quality": quality,
            "metric": "mse",  # important for psnr
        }
        self.model = bmshj2018_factorized(pretrained=True, progress=True, **kwargs).eval()

    def encode(self, x):
        with torch.no_grad():
            y = self.model.g_a(x)
            y_q, y_likelihoods = self.model.entropy_bottleneck(y)
            num_pixels = x.size(2) * x.size(3)
            bpp = torch.sum(torch.log2(y_likelihoods)) / (-num_pixels)
        return {"latent": y_q, "likelihoods": y_likelihoods, "bpp": bpp.item()}

    def decode(self, bneck_obj):
        with torch.no_grad():
            y_q = bneck_obj["latent"]
            x_hat = self.model.g_s(y_q)
        return {"x_hat": x_hat}

    def compress(self, x):
        encoded = self.encode(x)
        compressed_latent = encoded["latent"]
        buffer = io.BytesIO()
        torch.save(compressed_latent, buffer)
        compressed_bytes = buffer.getvalue()
        return {"compressed": compressed_bytes, "bpp": encoded["bpp"]}

    def decompress(self, compressed_data):
        buffer = io.BytesIO(compressed_data["compressed"])
        latent = torch.load(buffer, weights_only=True)
        decoded = self.decode({"latent": latent})
        x_hat = decoded["x_hat"]
        return x_hat


"""
classifiers
"""


class ClassifierInterface(ABC):
    @abstractmethod
    def predict_top_k(self, image: torch.Tensor, k: int) -> list:
        pass


class Resnet50Classifier(ClassifierInterface):
    def __init__(self):
        self.model = timm.create_model("resnet50", pretrained=True).eval()

    def predict_top_k(self, image, k):
        with torch.no_grad():
            output = self.model(image)
        _, top_k = torch.topk(output, k)
        return top_k.squeeze().tolist()


"""
dataset
"""


class DatasetInterface(ABC):
    @abstractmethod
    def get_generator(self):
        pass


class ImagenetDataset(DatasetInterface):
    def __init__(self):
        self.dataset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=True)

    def get_generator(self, size):
        subset = self.dataset.take(size).shuffle(seed=41)
        for elem in subset:
            yield elem["image_id"], elem["image"].convert("RGB"), elem["label"], elem["caption_enriched"]


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


def get_fid(real_features, fake_features):
    # FID (Fréchet Inception Distance)
    real_features = np.array(real_features)
    fake_features = np.array(fake_features)
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def get_kid(real_features, fake_features, subset_size=1000):
    # KID (Kernel Inception Distance)
    real_features = np.array(real_features)
    fake_features = np.array(fake_features)
    n = min(real_features.shape[0], fake_features.shape[0], subset_size)
    real_subset = real_features[:n]
    fake_subset = fake_features[:n]
    mmds = []
    for i in range(n):
        mmd = polynomial_kernel(real_subset[i : i + 1], fake_subset).mean()
        mmd -= polynomial_kernel(real_subset[i : i + 1], real_subset).mean()
        mmd += polynomial_kernel(fake_subset[i : i + 1], fake_subset).mean()
        mmds.append(mmd)
    return np.mean(mmds)


def get_imagenet_label(idx: int) -> str:
    datapath = Path.cwd() / "data" / "imagenet_labels.json"
    data = json.loads(datapath.read_text())
    return data[str(idx)]


"""
main
"""


if __name__ == "__main__":
    set_seed(41)

    outpath = Path.cwd() / "eval" / "eval_cls.csv"
    fidkidpath = Path.cwd() / "eval" / "eval_cls_fidkid.csv"
    outpath.unlink(missing_ok=True)
    fidkidpath.unlink(missing_ok=True)

    for quality in range(1, 9):
        codec = Bmshj2018Codec(quality=quality)
        subset_size = 25
        dataset = ImagenetDataset().get_generator(size=subset_size)
        resnet = Resnet50Classifier()

        inception = inception_v3(pretrained=True, transform_input=False).eval()
        inception_transform = Compose([Resize(299), CenterCrop(299), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        real_features = []
        fake_features = []

        for id, image, label_id, caption in tqdm(dataset, total=subset_size):
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
