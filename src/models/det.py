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


def detect_vit(img: Image.Image, labels: list[str], threshold: float) -> tuple[list[list[float]], list[float], list[str]]:
    # best model for cpu, gpu
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    device = get_device()
    model_id = "google/owlvit-large-patch14"
    processor = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id).to(device)

    texts = [labels]
    inputs = processor(text=texts, images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.Tensor([img.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)

    results[0]["boxes"] = [elem.cpu().tolist() for elem in results[0]["boxes"]]
    results[0]["scores"] = [elem.cpu().item() for elem in results[0]["scores"]]
    results[0]["labels"] = [labels[elem.cpu().item()] for elem in results[0]["labels"]]

    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    labels = results[0]["labels"]
    assert all(isinstance(label, str) for label in labels)
    assert all(isinstance(score, float) for score in scores)
    assert all(isinstance(box, list) and len(box) == 4 for box in boxes)
    assert all(all(isinstance(coord, float) for coord in box) for box in boxes)
    return boxes, scores, labels


def detect_groundingdino(img: Image.Image, labels: list[str], threshold: float) -> tuple[list[list[float]], list[float], list[str]]:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = get_device()
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    labels_str = ".".join(labels) + "."
    inputs = processor(images=img, text=labels_str, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=threshold, text_threshold=threshold, target_sizes=[img.size[::-1]])[0]
    results["scores"] = [elem.item() for elem in results["scores"]]
    results["boxes"] = [elem.tolist() for elem in results["boxes"]]

    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]
    assert all(isinstance(label, str) for label in labels)
    assert all(isinstance(score, float) for score in scores)
    assert all(isinstance(box, list) and len(box) == 4 for box in boxes)
    assert all(all(isinstance(coord, float) for coord in box) for box in boxes)
    return boxes, scores, labels


def detect_detr(img: Image.Image, threshold: float) -> tuple[list[list[float]], list[float], list[str]]:
    from transformers import AutoImageProcessor, DetrForObjectDetection

    model_id = "facebook/detr-resnet-101-dc5"  # largest model
    device = get_device()

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = DetrForObjectDetection.from_pretrained(model_id).to(device)

    inputs = image_processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([img.size[::-1]]).to(device)
    results = image_processor.post_process_object_detection(outputs, threshold, target_sizes=target_sizes)[0]

    results["boxes"] = [elem.cpu().tolist() for elem in results["boxes"]]
    results["scores"] = [elem.cpu().item() for elem in results["scores"]]
    results["labels"] = [model.config.id2label[elem.cpu().item()] for elem in results["labels"]]

    # model_labels = [model.config.id2label[elem.item()] for elem in results["labels"]]
    # results["labels"] = []
    # for ml in model_labels:
    #     for cl in labels:
    #         if cl.lower() in ml.lower():
    #             results["labels"].append(cl)
    #             break
    #     else:
    #         results["labels"].append("unknown")

    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]

    # Assertions for type checking
    assert all(isinstance(label, str) for label in labels)
    assert all(isinstance(score, float) for score in scores)
    assert all(isinstance(box, list) and len(box) == 4 for box in boxes)
    assert all(all(isinstance(coord, float) for coord in box) for box in boxes)

    return boxes, scores, labels


"""
utils
"""


def plot_detection(img: Image.Image, boxes: list[list[float]], scores: list[float], labels: list[str]):
    plt.imshow(img)
    for box, score, label in zip(boxes, scores, labels):
        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor="red", lw=2))
        plt.text(box[0], box[1], f"{label}: {score:.4f}", color="red")
    plt.axis("off")
    plt.show()


"""
example usage
"""


if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw)

    labels = ["cat", "remote control", "dog"]
    threshold = 0.1

    # boxes, scores, labels = detect_groundingdino(img, labels, threshold)
    # boxes, scores, labels = detect_vit(img, labels, threshold)
    boxes, scores, labels = detect_detr(img, threshold)

    plot_detection(img, boxes, scores, labels)
