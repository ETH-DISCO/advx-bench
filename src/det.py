import requests
import torch
from PIL import Image

from utils import get_device


def get_clip_predictions(img: Image.Image, labels: list[str]) -> dict[str, float]:
    pass

labels = ["quirky kittens on a couch", "chaotic remote controls", "a work of art"]

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img = Image.open(requests.get(url, stream=True).raw)
