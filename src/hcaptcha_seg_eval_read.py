from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file

from models.seg import plot_segmentation_detection


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
    plot_segmentation_detection(ret["image"], ret["detection_boxes"], ret["detection_scores"], ret["detection_labels"], ret["segmentation_masks"])
    print("-" * 40)
