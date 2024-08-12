import os

import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "true"

try:
    from .utils import get_device
except ImportError:
    from utils import get_device

"""
models
"""


def classify_metaclip(img: Image.Image, labels: list[str]) -> list[float]:
    # best model for cpu, gpu
    from transformers import AutoModel, AutoProcessor

    device = get_device()
    model_id = "facebook/metaclip-h14-fullcc2.5b"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)

    inputs = processor(text=labels, images=img, return_tensors="pt", padding=True)

    inputs["input_ids"] = inputs["input_ids"].to(device, dtype=torch.long)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        text_probs = logits_per_image.softmax(dim=-1)

    probs = text_probs[0].cpu().numpy().tolist()
    assert all(isinstance(prob, float) for prob in probs)
    return probs


def classify_clip(img: Image.Image, labels: list[str]) -> list[float]:
    # most adv robust model
    import clip

    device = get_device()
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    model.eval()

    text = clip.tokenize(labels).to(device)
    image = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    probs = probs[0].tolist()
    assert all(isinstance(prob, float) for prob in probs)
    return probs


def classify_opencoca(img: Image.Image, labels: list[str]) -> list[float]:
    import open_clip

    device = get_device()
    model, _, preprocess = open_clip.create_model_and_transforms("coca_ViT-L-14", pretrained="mscoco_finetuned_laion2b_s13b_b90k", device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("coca_ViT-L-14")

    image = preprocess(img).unsqueeze(0).to(device)
    text = tokenizer(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    probs = text_probs[0].cpu().numpy().tolist()
    assert all(isinstance(prob, float) for prob in probs)
    return probs


def classify_eva(img: Image.Image, labels: list[str]) -> list[float]:
    import open_clip

    device = get_device()
    model, _, preprocess = open_clip.create_model_and_transforms("EVA01-g-14", pretrained="laion400m_s11b_b41k", device=device)  # largest that can fit in memory
    model.eval()
    tokenizer = open_clip.get_tokenizer("EVA01-g-14")

    image = preprocess(img).unsqueeze(0).to(device)
    text = tokenizer(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    probs = text_probs[0].cpu().numpy().tolist()
    assert all(isinstance(prob, float) for prob in probs)
    return probs


"""
utils
"""


def plot_classification(img: Image.Image, labels: list[str], probs: list[float]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title("Image")

    labels_probs = zip(labels, probs)
    sorted_preds = sorted(labels_probs, key=lambda x: x[1], reverse=True)
    ax2.barh([label for label, _ in sorted_preds], [prob for _, prob in sorted_preds], color="skyblue", align="center")

    # invert x axis
    ax2.invert_xaxis()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_xlim(1, 0)
    ax2.set_xlabel("Probability")
    ax2.set_title("Predictions")

    plt.tight_layout()
    plt.show()


"""
example
"""

if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw)

    labels = ["quirky kittens on a couch", "chaotic remote controls", "a work of art"]

    probs = classify_metaclip(img, labels)
    probs = classify_clip(img, labels)
    probs = classify_opencoca(img, labels)
    probs = classify_eva(img, labels)

    plot_classification(img, labels, probs)
