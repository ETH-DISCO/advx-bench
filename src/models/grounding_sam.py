import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from det import detect_groundingdino
from PIL import Image
from seg import refine_masks
from transformers import AutoModelForMaskGeneration, AutoProcessor
from utils import get_device


def plot(image: Image.Image, results):
    boxes, scores, labels, masks = results
    boxes = [[math.floor(val) for val in box] for box in boxes]

    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for label, score, (xmin, ymin, xmax, ymax), mask in zip(labels, scores, boxes, masks):
        color = np.random.randint(0, 256, size=3)

        # bounding box
        cv2.rectangle(image_cv2, (xmin, ymin), (xmax, ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f"{label}: {score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # mask
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    annotated_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.show()


def segment_groundeddino_sam(image: Image.Image, labels: list[str], threshold: float):
    boxes, scores, labels = detect_groundingdino(img, labels, threshold)

    # update to sam2 model
    device = get_device()
    segmenter_id = "facebook/sam-vit-base"
    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)
    inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = segmentator(**inputs)
    masks = processor.post_process_masks(masks=outputs.pred_masks, original_sizes=inputs.original_sizes, reshaped_input_sizes=inputs.reshaped_input_sizes)[0]

    masks = refine_masks(masks)

    return boxes, scores, labels, masks


labels = ["a cat", "a remote control"]
threshold = 0.3

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img = Image.open(requests.get(url, stream=True).raw)

results = segment_groundeddino_sam(img, labels, threshold)

plot(img, results)
