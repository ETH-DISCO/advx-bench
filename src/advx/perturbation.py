import requests
# from PIL import Image
# from utils import get_device
# import torch

# def classify_clip(img: Image.Image, labels: list[str]) -> list[float]:
#     import clip

#     device = get_device()
#     model, preprocess = clip.load("ViT-L/14@336px", device=device)
#     model.eval()

#     text = clip.tokenize(labels).to(device)
#     image = preprocess(img).unsqueeze(0).to(device)

#     with torch.no_grad():
#         logits_per_image, logits_per_text = model(image, text)
#         probs = logits_per_image.softmax(dim=-1).cpu().numpy()

#     probs = probs[0].tolist()
#     assert all(isinstance(prob, float) for prob in probs)
#     return probs


# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

import torch
import torch.nn.functional as F
import clip
from PIL import Image

def generate_adversarial_example(img: Image.Image, labels: list[str], target_label: int, epsilon: float = 0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    model.eval()

    # Ensure labels are properly tokenized and converted to the correct dtype
    text = clip.tokenize(labels).to(device)
    
    # Preprocess and convert image to the correct dtype
    image = preprocess(img).unsqueeze(0).to(device).to(torch.float32)
    image.requires_grad = True

    # Forward pass
    logits_per_image, _ = model(image, text)
    
    # Ensure target_label is a long tensor
    target = torch.tensor([target_label], device=device, dtype=torch.long)
    
    # Calculate loss
    loss = F.cross_entropy(logits_per_image, target)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Generate adversarial example
    data_grad = image.grad.data
    perturbed_image = image + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image.squeeze(0).detach().cpu()

# Usage example
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
labels = ["a photo of a cat", "a photo of a dog"]
target_label = 1  # Targeting "dog"
epsilon = 0.01

adversarial_image = generate_adversarial_example(image, labels, target_label, epsilon)

