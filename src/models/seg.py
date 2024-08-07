import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from det import *
from generate import *
from PIL import Image

from utils import get_device

"""
models
"""


def segment_clipseg(img: Image.Image, labels: list[str]) -> list[torch.Tensor]:
    from transformers import AutoProcessor, CLIPSegForImageSegmentation

    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    inputs = processor(text=labels, images=[img] * len(labels), padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    masks = torch.sigmoid(logits)

    assert all(isinstance(mask, torch.Tensor) for mask in masks)
    assert all(mask.dtype == torch.float32 for mask in masks)
    return masks


def segment_sam1(image: Image.Image, query: list[list[float]]) -> list[torch.Tensor]:
    assert len(query) > 0
    assert all(len(box) == 4 for box in query)
    from transformers import AutoModelForMaskGeneration, AutoProcessor

    segmenter_id = "facebook/sam-vit-base"
    device = get_device(disable_mps=True)
    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)
    inputs = processor(images=image, input_boxes=[query], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = segmentator(**inputs)
    masks = processor.post_process_masks(masks=outputs.pred_masks, original_sizes=inputs.original_sizes, reshaped_input_sizes=inputs.reshaped_input_sizes)[0]

    assert all(isinstance(mask, torch.Tensor) for mask in masks)
    assert all(mask.dtype == torch.bool for mask in masks)
    return masks


"""
utils
"""


def plot_segmentation_prob(img: Image.Image, labels: list[str], masks: list[torch.Tensor]):
    plt.figure(figsize=(8, 5))
    plt.imshow(img)
    plt.axis("off")

    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(labels)))

    threshold = 0.5  # threshold above which to consider the mask value as 'true'
    for mask, color in zip(masks, colors):
        np_mask = mask.squeeze().cpu().numpy()
        mask_resized = np.array(Image.fromarray(np_mask).resize(img.size, Image.BILINEAR))  # resize to fit the original image

        color_mask = np.zeros((*mask_resized.shape, 4))
        color_mask[..., :3] = color[:3]  # use the first 3 values (RGB) from the color tuple
        color_mask[..., 3] = mask_resized * threshold

        plt.imshow(color_mask, alpha=0.9)
        plt.contour(mask_resized, levels=[threshold], colors=[color[:3]], alpha=0.9)

    legend_elems = [plt.Line2D([0], [0], color=color, lw=4, label=label) for label, color in zip(labels, colors)]
    plt.legend(handles=legend_elems, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()


def plot_segmentation_detection(image: Image.Image, boxes: list[list[float]], scores: list[float], labels: list[str], masks: list[torch.Tensor]):
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
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.show()


"""
example
"""


def demo():
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # img = Image.open(requests.get(url, stream=True).raw)
    # threshold = 0.3
    # labels = ["cat", "remote control"]
    # masks = segment_clipseg(img, labels)
    # plot_segmentation_prob(img, labels, masks)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw)
    threshold = 0.1
    labels = ["cat", "remote control"]
    boxes, scores, labels = detect_groundingdino(img, labels, threshold)
    masks = segment_sam1(img, query=boxes)
    plot_segmentation_detection(img, boxes, scores, labels, masks)

    # img = Image.open(Path(__file__).parent.parent.parent / "data" / "kodak" / "kodim14.png")
    # threshold = 0.9
    # boxes, scores, labels = detect_detr(img, threshold)
    # masks = segment_sam1(img, query=boxes)
    # plot_segmentation_detection(img, boxes, scores, labels, masks)


def pipeline():
    GREEN = "\033[92m"
    RESET = "\033[0m"

    # 1. generate an image
    print(f"{GREEN}Generating an image...{RESET}")
    img = gen_stable_diffusion("an illustration of a cat, a dog, and a flamingo having a picnic")
    img.show()

    # 2. caption it
    print(f"{GREEN}Captioning the image...{RESET}")
    from caption import caption_vqa
    text_query = caption_vqa(img)
    print("text_query:", text_query)

    # 3. classify, detect, segment it
    print(f"{GREEN}Classifying the image...{RESET}")
    from cls import classify_clip, plot_classification
    labels, probs = classify_clip(img, text_query)
    plot_classification(img, labels, probs)

    print(f"{GREEN}Detecting objects in the image...{RESET}")
    threshold = 0.1
    boxes, scores, labels = detect_groundingdino(img, text_query, threshold)

    print(f"{GREEN}Segmenting objects in the image...{RESET}")
    masks = segment_sam1(img, query=boxes)
    plot_segmentation_detection(img, boxes, scores, labels, masks)


if __name__ == "__main__":
    demo()
    # pipeline()
