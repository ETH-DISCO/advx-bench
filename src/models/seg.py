import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from utils import get_random_color

"""
models
"""


def segment_clipseg(img: Image.Image, labels: list[str]) -> list[tuple[str, torch.Tensor]]:
    from transformers import AutoProcessor, CLIPSegForImageSegmentation

    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    inputs = processor(text=labels, images=[img] * len(labels), padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    masks = torch.sigmoid(logits)

    return list(zip(labels, masks))


def segment_groundingsam(img: Image.Image, labels: list[str]) -> list[tuple[str, torch.Tensor]]:
    pass


"""
utils
"""


def plot_segmentation(img: Image.Image, results: list[tuple[str, torch.Tensor]]):
    plt.imshow(img)
    plt.axis("off")

    colors = get_random_color(len(results))
    for (label, mask), color in zip(results, colors):
        mask_np = mask.squeeze().numpy()

        # resize the mask to match the original image dimensions
        mask_resized = np.array(Image.fromarray(mask_np).resize(img.size, Image.BILINEAR))

        color_mask = np.zeros((*mask_resized.shape, 4))
        color_mask[..., :3] = plt.cm.colors.to_rgb(color)
        color_mask[..., 3] = mask_resized * 0.5
        plt.imshow(color_mask)
        plt.contour(mask_resized, levels=[0.5], colors=[color], alpha=0.8)

    legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=label) for (label, mask), color in zip(results, colors)]
    plt.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()


def refine_masks(masks: torch.BoolTensor) -> list[np.ndarray]:
    # make mask prettier by adding contours and filling it in

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


# labels = ["cat", "remote control"]

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# img = Image.open(requests.get(url, stream=True).raw)

# results = segment_clipseg(img, labels)
# plot_segmentation(img, results)
