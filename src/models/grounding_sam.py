import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from det import detect_groundingdino
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor
from utils import get_device


def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract the vertices of the contour
        polygon = largest_contour.reshape(-1, 2).tolist()

        return polygon

    def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert a polygon to a segmentation mask.

        Args:
        - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
        - image_shape (tuple): Shape of the image (height, width) for the mask.

        Returns:
        - np.ndarray: Segmentation mask with the polygon filled.
        """
        # Create an empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Convert polygon to an array of points
        pts = np.array(polygon, dtype=np.int32)

        # Fill the polygon with white color (255)
        cv2.fillPoly(mask, [pts], color=(255,))

        return mask

    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


@dataclass
class DetectionResult:
    score: float
    label: str

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    mask: Optional[np.array] = None


def plot(image: Image.Image, results):
    boxes, scores, labels, masks = results
    boxes = [[math.floor(val) for val in box] for box in boxes]
    detection_results = []
    for box, score, label, mask in zip(boxes, scores, labels, masks):
        detection_results.append(DetectionResult(score=score, label=label, xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3], mask=mask))


    image_cv2 = np.array(image)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    for detection in detection_results:
        label = detection.label
        score = detection.score
        xmin, ymin, xmax, ymax = detection.xmin, detection.ymin, detection.xmax, detection.ymax
        mask = detection.mask

        color = np.random.randint(0, 256, size=3)

        # bounding box
        cv2.rectangle(image_cv2, (xmin, ymin), (xmax, ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f"{label}: {score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # mask
        if mask is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    annotated_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.show()


def segment_groundeddino_sam(image: Image.Image, labels: List[str], threshold: float, polygon_refinement: bool):
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

    masks = refine_masks(masks, polygon_refinement)


    return boxes, scores, labels, masks


labels = ["a cat", "a remote control"]
threshold = 0.3

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img = Image.open(requests.get(url, stream=True).raw)

results = segment_groundeddino_sam(img, labels, threshold, polygon_refinement=True)

plot(img, results)
