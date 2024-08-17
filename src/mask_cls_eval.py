import csv
import io
import random
import time
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Iterator

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from compressai.zoo import bmshj2018_factorized
from datasets import load_dataset
from PIL import Image
from skimage.metrics import structural_similarity
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
    def get_generator(self) -> Iterator[Image.Image]:
        pass


class ImagenetteDataset(DatasetInterface):
    def __init__(self):
        # smaller imagenet that doesn't need huggingface auth
        self.dataset = load_dataset("frgfm/imagenette", "full_size", split="validation")["image"]

    def get_generator(self):
        for image in self.dataset:
            yield image


class KodakDataset(DatasetInterface):
    def __init__(self):
        path = Path.cwd() / "kodak"
        self.dataset = [Image.open(img) for img in glob(str(path / "*.png"))]

    def get_generator(self):
        for image in self.dataset:
            yield image


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


"""
main
"""


if __name__ == "__main__":
    set_seed(41)

    outpath = Path.cwd() / "eval" / "eval_cls.csv"
    if outpath.exists():
        outpath.unlink()

    for quality in range(1, 9):
        codec = Bmshj2018Codec(quality=quality)
        dataset = KodakDataset().get_generator()

        for image in tqdm(dataset):
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

            # normalization for imagenette
            # x = (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            # x_hat = (x_hat - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

            img_normalize = lambda x: x
            resnet = Resnet50Classifier()
            x_preds = resnet.predict_top_k(x, k=5)
            x_hat_preds = resnet.predict_top_k(x_hat, k=5)
            metrics = {
                "quality": quality,
                "time_compression": time_compression,
                "time_decompression": time_decompression,
                "ssim": structural_similarity(np.array(x.squeeze().permute(1, 2, 0).cpu().numpy()), np.array(x_hat.squeeze().permute(1, 2, 0).cpu().numpy()), multichannel=True, channel_axis=2, data_range=1.0),
                "bpp": compressed["bpp"],
                "latent_cosine_similarity": F.cosine_similarity(codec.encode(x)["latent"].view(1, -1), codec.encode(x_hat)["latent"].view(1, -1)).item(),
                "psnr": (20 * torch.log10(1.0 / torch.sqrt(torch.mean((x - x_hat) ** 2)))).item(),
                "bitrate": torch.sum(torch.log2(codec.encode(x)["likelihoods"])).item() / (x.size(2) * x.size(3)),
                # comparing against original image, not ground truth
                "top_1_accuracy": int(x_preds[0] == x_hat_preds[0]),
                "top_5_accuracy": len(set(x_preds) & set(x_hat_preds)) / 5,
            }

            with open(outpath, mode="a") as f:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                if outpath.stat().st_size == 0:
                    writer.writeheader()
                writer.writerow(metrics)
