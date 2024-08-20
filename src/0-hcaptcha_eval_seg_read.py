import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file


def get_img(image: Image.Image, boxes: list[list[float]], scores: list[float], labels: list[str], masks: list[torch.Tensor]):
    def _refine_masks(masks: torch.BoolTensor) -> list[np.ndarray]:
        def mask_to_polygon(mask: np.ndarray) -> list[list[int]]:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            polygon = largest_contour.reshape(-1, 2).tolist()  # extract vertices of the contour
            return polygon

        def polygon_to_mask(polygon: list[tuple[int, int]], image_shape: tuple[int, int]) -> np.ndarray:
            # polygon = (x, y) coordinates of the vertices
            # image_shape = (height, width) of the mask

            mask = np.zeros(image_shape, dtype=np.uint8)
            pts = np.array(polygon, dtype=np.int32)  # point array
            cv2.fillPoly(mask, [pts], color=(255,))
            return mask

        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)
        masks = list(masks)

        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

        return masks

    boxes = [[math.floor(val) for val in box] for box in boxes]
    masks = _refine_masks(masks)

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
    return annotated_image


def load_and_decode_safetensor(file_path):
    data_dict = load_file(file_path)

    out = {}
    for key, tensor in data_dict.items():
        if key == "image":
            img = Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            out[key] = img
        elif key == "captions" or key == "detection_labels":

            def decode_int32_to_string(tensor):
                return "".join(chr(i) for i in tensor.tolist())

            decoded = decode_int32_to_string(tensor)
            out[key] = decoded.split("|")
        elif key == "classification_probs" or key == "detection_scores":
            out[key] = tensor.tolist()
        elif key == "detection_boxes":
            out[key] = tensor.tolist()
        elif key == "segmentation_masks":
            out[key] = tensor
    return out


def refine_masks(masks):
    if masks.dim() == 1:
        # if masks is 1D, reshape it to 4D
        num_masks = masks.size(0)
        masks = masks.view(num_masks, 1, 1, 1)
    elif masks.dim() == 3:
        # if masks is 3D, add a channel dimension
        masks = masks.unsqueeze(1)

    masks = masks.permute(0, 2, 3, 1)
    return masks


datapath = Path.cwd() / "data" / "hcaptcha" / "seg" / "eval"
assert datapath.exists()


for file in datapath.glob("*.safetensors"):
    print(f"Reading {file.stem}:")
    ret = load_and_decode_safetensor(file)
    ret["segmentation_masks"] = refine_masks(ret["segmentation_masks"])
    ann_img = get_img(ret["image"], ret["detection_boxes"], ret["detection_scores"], ret["detection_labels"], ret["segmentation_masks"])

    if not (datapath / file.stem).exists():
        Image.fromarray(ann_img).save(datapath / f"{file.stem}.png")
    else:
        print(f"skipping: {file.stem}")

    print("-" * 40)
