import torch
import requests
from PIL import Image
from utils import get_device


def get_clip_predictions(img: Image.Image, labels: list[str]) -> dict[str, float]:
    import clip

    device = get_device()
    model, preprocess = clip.load("ViT-L/14@336px", device=device)  # largest ViT

    text = clip.tokenize(labels).to(device)
    image = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return {label: prob for label, prob in zip(labels, probs[0])}



labels = ["quirky kittens on a couch", "chaotic remote controls", "a work of art"]

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)


print(get_clip_predictions(image, labels))
