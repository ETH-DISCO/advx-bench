import requests
import torch
from PIL import Image

from utils import get_device


def encode_vit_mae(img: Image.Image) -> torch.Tensor:
    # best model for cpu, gpu
    from transformers import ViTImageProcessor, ViTMAEModel

    device = get_device()
    model_id = "facebook/vit-mae-base"
    model = ViTMAEModel.from_pretrained(model_id, attn_implementation="sdpa").to(device)
    processor = ViTImageProcessor.from_pretrained(model_id)

    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)  # avg pooling over sequence length

    return embedding


"""
example usage
"""


if __name__ == "__main__":
    img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw).convert("RGB")
    embedding = encode_vit_mae(img)
    print(embedding.shape)
