import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from utils import get_random_color


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

    legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=label) for (label, _), color in zip(results, colors)]
    plt.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()


# labels = ["cat", "remote control"]

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# img = Image.open(requests.get(url, stream=True).raw)

# results = segment_clipseg(img, labels)
# plot_segmentation(img, results)
