import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image
from utils import get_device


def classify_clip(img: Image.Image, labels: list[str]) -> dict[str, float]:
    import clip

    device = get_device()
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    model.eval()

    text = clip.tokenize(labels).to(device)
    image = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return {label: prob for label, prob in zip(labels, probs[0])}


def classify_opencoca(img: Image.Image, labels: list[str]) -> dict[str, float]:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("coca_ViT-L-14", pretrained="mscoco_finetuned_laion2b_s13b_b90k")
    model.eval()
    tokenizer = open_clip.get_tokenizer("coca_ViT-L-14")

    image = preprocess(img).unsqueeze(0)
    text = tokenizer(labels)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return {label: prob.item() for label, prob in zip(labels, text_probs[0])}


def classify_eva(img: Image.Image, labels: list[str]) -> dict[str, float]:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("EVA01-g-14", pretrained="laion400m_s11b_b41k")  # largest that can fit in memory
    model.eval()
    tokenizer = open_clip.get_tokenizer("EVA01-g-14")

    image = preprocess(img).unsqueeze(0)
    text = tokenizer(labels)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return {label: prob.item() for label, prob in zip(labels, text_probs[0])}


def classify_gem(img: Image.Image, labels: list[str]) -> dict[str, float]:
    from transformers import AutoModel, AutoProcessor

    model_id = "facebook/metaclip-h14-fullcc2.5b"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    inputs = processor(text=labels, images=img, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        text_probs = logits_per_image.softmax(dim=-1)

    return {label: prob.item() for label, prob in zip(labels, text_probs[0])}


def plot_classification(predictions: dict[str, float]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title("Image")

    sorted_preds = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
    ax2.barh(list(sorted_preds.keys()), list(sorted_preds.values()), color="skyblue", align="center")
    # invert x axis
    ax2.invert_xaxis()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_xlim(1, 0)
    ax2.set_xlabel("Probability")
    ax2.set_title("Predictions")

    plt.tight_layout()
    plt.show()


labels = ["quirky kittens on a couch", "chaotic remote controls", "a work of art"]
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img = Image.open(requests.get(url, stream=True).raw)

preds = classify_clip(img, labels)
# preds = classify_opencoca(img, labels)
# preds = classify_eva(img, labels)
# preds = classify_gem(img, labels)

plot_classification(preds)
