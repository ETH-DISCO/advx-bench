import time
from typing import Callable, Union

import numpy as np
import requests
import torch
from PIL import Image
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import polynomial_kernel
from torchvision.models import inception_v3
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import ViTFeatureExtractor, ViTModel

from utils import get_device


def get_time_result(func: Callable, *args):
    time_start = time.time()
    result = func(*args)
    time_elapsed = time.time() - time_start
    return time_elapsed, result


def get_inception_features(x: torch.Tensor) -> np.ndarray:
    inception = inception_v3(pretrained=True, transform_input=False).eval()
    inception_transform = Compose([Resize(299), CenterCrop(299), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    with torch.no_grad():
        feature = inception(inception_transform(x)).squeeze().cpu().numpy()
    return feature


def get_psnr(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    mse = torch.mean((x - x_hat) ** 2)
    return float(20 * torch.log10(1.0 / torch.sqrt(mse)))


def get_ssim(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    return structural_similarity(np.array(x.squeeze().permute(1, 2, 0).cpu().numpy()), np.array(x_hat.squeeze().permute(1, 2, 0).cpu().numpy()), multichannel=True, channel_axis=2, data_range=1.0)


def get_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    # fid = fréchet inception distance
    real_features = np.array(real_features)
    fake_features = np.array(fake_features)
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):  # type: ignore
        covmean = covmean.real  # type: ignore
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)  # type: ignore
    return fid


def get_kid(real_features: np.ndarray, fake_features: np.ndarray, subset_size: int) -> float:
    # kid = kernel inception distance
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
    return float(np.mean(mmds))


def get_iou(box1: list[float], box2: list[float]) -> float:
    # iou = intersection over union
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xi1, yi1 = max(x1, x1_), max(y1, y1_)
    xi2, yi2 = min(x2, x2_), min(y2, y2_)
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    return intersection / (box1_area + box2_area - intersection)


def get_cosine_similarity(x: Union[torch.Tensor, Image.Image], y: Union[torch.Tensor, Image.Image]) -> float:
    device = get_device()
    model_name = "google/vit-base-patch16-224"
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name).to(device)

    def process_input(input_data):
        if isinstance(input_data, Image.Image):
            return feature_extractor(images=input_data, return_tensors="pt")
        elif isinstance(input_data, torch.Tensor):
            if input_data.dim() == 3:
                input_data = input_data.unsqueeze(0)
            return feature_extractor(images=input_data, return_tensors="pt")
        else:
            raise TypeError(f"input should be PIL Image or torch.Tensor. got {type(input_data)}")

    inputs1 = process_input(x)
    inputs2 = process_input(y)

    inputs1 = {k: v.to(device) for k, v in inputs1.items()}
    inputs2 = {k: v.to(device) for k, v in inputs2.items()}

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    latents1 = outputs1.last_hidden_state[:, 0, :]  # use CLS token as image representation
    latents2 = outputs2.last_hidden_state[:, 0, :]

    cosine_sim = torch.nn.functional.cosine_similarity(latents1, latents2).item()
    return cosine_sim


"""
example usage
"""


if __name__ == "__main__":
    url1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
    url2 = "http://images.cocodataset.org/val2017/000000000285.jpg"

    image1 = Image.open(requests.get(url1, stream=True).raw).convert("RGB")
    image2 = Image.open(requests.get(url2, stream=True).raw).convert("RGB")

    image1_tensor = ToTensor()(image1).unsqueeze(0)
    image2_tensor = ToTensor()(image2).unsqueeze(0)

    print(f"Cosine similarity: {get_cosine_similarity(image1, image2)}")
    print(f"Cosine similarity: {get_cosine_similarity(image1_tensor, image2_tensor)}")
