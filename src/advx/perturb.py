import json
from pathlib import Path

import clip
import requests
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from utils import get_device

"""
attacks
"""


def get_fgsm_resnet_imagenet(image: Image.Image, target: int, epsilon: float, debug: bool = False) -> Image.Image:
    model = models.resnet18(pretrained=True)
    model.eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    def fgsm_attack(image, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    input_batch.requires_grad = True

    output = model(input_batch)
    target = torch.tensor([target])
    loss = torch.nn.functional.cross_entropy(output, target)

    model.zero_grad()
    loss.backward()

    data_grad = input_batch.grad.data
    perturbed_data = fgsm_attack(input_batch, epsilon, data_grad)

    if debug:
        original_top5 = torch.nn.functional.softmax(output, dim=1).topk(5)
        perturbed_output = model(perturbed_data)
        perturbed_top5 = torch.nn.functional.softmax(perturbed_output, dim=1).topk(5)

        original_label_preds: dict = {}
        for i in range(5):
            indices = original_top5.indices.squeeze(0).tolist()
            labels = [get_imagenet_label(idx) for idx in indices]
            original_label_preds[labels[i]] = original_top5.values.squeeze(0)[i].item()
        print("perturbed top-5 predictions:", perturbed_top5)

        perturbed_label_preds: dict = {}
        for i in range(5):
            indices = perturbed_top5.indices.squeeze(0).tolist()
            labels = [get_imagenet_label(idx) for idx in indices]
            perturbed_label_preds[labels[i]] = perturbed_top5.values.squeeze(0)[i].item()
        print("original top-5 predictions:", original_label_preds)

    return transforms.ToPILImage()(perturbed_data.squeeze(0))


def get_fgsm_clipvit_imagenet(image: Image.Image, target_idx: int, labels: list, epsilon: float, debug: bool = False) -> Image.Image:
    device = get_device(disable_mps=True)
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    model.eval()

    # enable gradients for model parameters
    for param in model.parameters():
        param.requires_grad = True

    input_tensor = preprocess(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True  # enable gradients for input tensor
    text_inputs = clip.tokenize(labels).to(device)

    def fgsm_attack(image, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    with torch.enable_grad():
        model.zero_grad()

        image_features = model.encode_image(input_tensor)
        text_features = model.encode_text(text_inputs)

        logits_per_image = image_features @ text_features.T
        logits_per_text = logits_per_image.T

        loss = -logits_per_image[0, target_idx]  # maximize the target class score

        try:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        except RuntimeError as e:
            print(f"Error during backward pass: {e}")
            print(f"Loss requires grad: {loss.requires_grad}")
            print(f"Input tensor requires grad: {input_tensor.requires_grad}")
            raise

    data_grad = input_tensor.grad.data

    perturbed_data = fgsm_attack(input_tensor, epsilon, data_grad)

    if debug:
        with torch.no_grad():
            image_features = model.encode_image(input_tensor)
            text_features = model.encode_text(text_inputs)
            logits_per_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            original_label_preds = [(labels[i], logits_per_image[0, i].item()) for i in range(len(labels))]
            print("original label predictions:", original_label_preds)

            perturbed_features = model.encode_image(perturbed_data)
            perturbed_logits = (100.0 * perturbed_features @ text_features.T).softmax(dim=-1)
            perturbed_label_preds = [(labels[i], perturbed_logits[0, i].item()) for i in range(len(labels))]
            print("perturbed label predictions:", perturbed_label_preds)

    return transforms.ToPILImage()(perturbed_data.squeeze(0))


"""
utils
"""


def get_imagenet_label(idx: int) -> str:
    datapath = Path.cwd() / "data" / "imagenet_labels.json"
    data = json.loads(datapath.read_text())
    return data[str(idx)]


"""
example usage
"""


if __name__ == "__main__":
    image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw).convert("RGB")

    # target = 281 # tabby cat in ImageNet
    # epsilon = 0.001
    # perturbed = get_fgsm_resnet_imagenet(image, target, epsilon, debug=True)

    target_idx = 0
    labels = ["a photo of a dog", "a photo of a bird", "a photo of a cat"]
    epsilon = 0.8  # larger epsilon for CLIP-ViT because it's more robust
    perturbed = get_fgsm_clipvit_imagenet(image, target_idx, labels, epsilon, debug=True)

    perturbed.show()
