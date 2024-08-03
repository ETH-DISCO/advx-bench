import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image

from utils import get_device


def get_clip_predictions(img: Image.Image, labels: list[str], threshold: float) -> list:
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
    return results


def plot_results(results: dict[str, float]):
    plt.imshow(img)
    for res in list(zip(results["labels"], results["scores"], results["boxes"])):
        label = res[0]
        score = res[1]
        box = res[2]

        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor="red", lw=2))
        plt.text(box[0], box[1], f"{label}: {score:.4f}", color="red")

    plt.axis("off")
    plt.show()


labels = ["cat", "remote control", "dog"]
threshold = 0.3

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img = Image.open(requests.get(url, stream=True).raw)

results = get_clip_predictions(img, labels, threshold)
plot_results(results)
