import matplotlib.pyplot as plt
import numpy as np
import torch
import requests
from PIL import Image


def get_clipseg_results(img: Image.Image, labels: list[str]):
    pass


def get_groundingsam_results(img: Image.Image, labels: list[str]):
    pass


def get_mask2former_results(img: Image.Image, labels: list[str]):
    pass


labels = ["cat", "remote control"]

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img = Image.open(requests.get(url, stream=True).raw)




# fig, axs = plt.subplots(1, 4, figsize=(20, 5))
# axs[0].imshow(img)
# axs[0].axis("off")
# axs[0].set_title("Input Image")
# for i, mask in enumerate(masks):
#     axs[i + 1].imshow(mask.squeeze().cpu().numpy(), cmap="gray")
#     axs[i + 1].axis("off")
#     axs[i + 1].set_title(f"segmentation mask for '{labels[i]}'")
# plt.show()
