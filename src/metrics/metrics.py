import time
from typing import Callable

import numpy as np
import torch
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import polynomial_kernel
from torchvision.models import inception_v3
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize


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
    print(type(feature))
    return feature


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


"""
example usage
"""

if __name__ == "__main__":
    real_features = np.random.rand(1000, 2048)
    fake_features = np.random.rand(1000, 2048)

    fid = get_fid(real_features, fake_features)
    print(f"fid: {fid}")

    kid = get_kid(real_features, fake_features, subset_size=100)
    print(f"kid: {kid}")

    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    iou = get_iou(box1, box2)
    print(f"iou: {iou}")
