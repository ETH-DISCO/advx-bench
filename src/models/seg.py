import requests
from PIL import Image


def get_clipseg_results(img: Image.Image, labels: list[str]):
    pass


def get_groundingsam_results(img: Image.Image, labels: list[str]):
    pass


def get_mask2former_results(img: Image.Image, labels: list[str]):
    pass


labels = ["cat", "remote control", "dog"]
threshold = 0.1

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img = Image.open(requests.get(url, stream=True).raw)
