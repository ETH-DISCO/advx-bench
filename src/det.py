import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image

from utils import get_device


def get_groundingdino_predictions(img: Image.Image, labels: list[str], threshold: float) -> tuple[list[list[float]], list[float], list[str]]:
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
    return boxes, scores, labels


def get_vit_predictions(img: Image.Image, labels: list[str], threshold: float) -> tuple[list[list[float]], list[float], list[str]]:
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    # model_id = "google/owlvit-large-patch14" # inference too slow
    model_id = "google/owlvit-base-patch32"
    processor = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id)

    texts = [labels]
    inputs = processor(text=texts, images=img, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.Tensor([img.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)

    results[0]["boxes"] = [elem.tolist() for elem in results[0]["boxes"]]
    results[0]["scores"] = [elem.item() for elem in results[0]["scores"]]
    results[0]["labels"] = [labels[elem] for elem in results[0]["labels"]]
    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
    return boxes, scores, labels


def plot_results(results: tuple[list[list[float]], list[float], list[str]]):
    boxes, scores, labels = results
    plt.imshow(img)
    for box, score, label in zip(boxes, scores, labels):
        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor="red", lw=2))
        plt.text(box[0], box[1], f"{label}: {score:.4f}", color="red")
    plt.axis("off")
    plt.show()


labels = ["cat", "remote control", "dog"]
threshold = 0.1

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img = Image.open(requests.get(url, stream=True).raw)


# results = get_groundingdino_predictions(img, labels, threshold)
results = get_vit_predictions(img, labels, threshold)
plot_results(results)
