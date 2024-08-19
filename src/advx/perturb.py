import json
import matplotlib.pyplot as plt
import torch
import clip
from PIL import Image
import requests
from io import BytesIO
import requests
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import CLIPModel, CLIPProcessor
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path

"""
attacks
"""

def get_fgsm_resnet_imagenet(image: Image.Image, target: int, epsilon: float, debug: bool = True) -> Image.Image:
    model = models.resnet18(pretrained=True)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
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
        print("Original top 5 predictions:")
        for i in range(5):
            indices = original_top5.indices.squeeze(0).tolist()
            labels = [get_imagenet_label(idx) for idx in indices]
            print(f"\t{labels[i]}: {original_top5.values.squeeze(0)[i]:.2%}")
        print("Perturbed top 5 predictions:")
        for i in range(5):
            indices = perturbed_top5.indices.squeeze(0).tolist()
            labels = [get_imagenet_label(idx) for idx in indices]
            print(f"\t{labels[i]}: {perturbed_top5.values.squeeze(0)[i]:.2%}")
    
    return transforms.ToPILImage()(perturbed_data.squeeze(0))


"""
plot
"""

def get_imagenet_label(idx: int) -> str:
    datapath = Path.cwd() / "data" / "imagenet_labels.json"
    data = json.loads(datapath.read_text())
    return data[str(idx)]

# # Display original and perturbed images
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(tensor_to_pil(input_batch))
# plt.title("Original Image")
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(tensor_to_pil(perturbed_data))
# plt.title("Perturbed Image")
# plt.axis('off')

# plt.show()







# # Load CLIP model and processor
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-L/14@336px", device=device)
# model.eval()

# # Load image
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# response = requests.get(url)
# image = Image.open(BytesIO(response.content)).convert("RGB")

# # Preprocess image
# input_tensor = preprocess(image).unsqueeze(0).to(device)

# text_inputs = clip.tokenize(["a photo of a cat", "a photo of a dog"]).to(device)

# def fgsm_attack(image, epsilon, data_grad):
#     sign_data_grad = data_grad.sign()
#     perturbed_image = image + epsilon * sign_data_grad
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
#     return perturbed_image

# # epsilon = 0.07
# epsilon = 1 # because clip is adversarially robust it requires a higher epsilon value
# input_tensor.requires_grad = True

# # Forward pass
# image_features = model.encode_image(input_tensor)
# text_features = model.encode_text(text_inputs)

# # Calculate similarity
# logits_per_image = image_features @ text_features.T
# logits_per_text = logits_per_image.T

# # We want to maximize the probability of "cat"
# loss = -logits_per_image[0][0]

# # Backward pass
# loss.backward()

# # Collect datagrad
# data_grad = input_tensor.grad.data

# # Call FGSM Attack
# perturbed_data = fgsm_attack(input_tensor, epsilon, data_grad)

# # import matplotlib.pyplot as plt
# # import torchvision.transforms as transforms

# # def tensor_to_pil(tensor):
# #     return transforms.ToPILImage()(tensor.squeeze(0).cpu())

# # plt.figure(figsize=(10, 5))
# # plt.subplot(1, 2, 1)
# # plt.imshow(tensor_to_pil(input_tensor))
# # plt.title("Original Image")
# # plt.axis('off')

# # plt.subplot(1, 2, 2)
# # plt.imshow(tensor_to_pil(perturbed_data))
# # plt.title("Perturbed Image")
# # plt.axis('off')

# # plt.show()

# # Print probabilities
# with torch.no_grad():
#     image_features = model.encode_image(input_tensor)
#     text_features = model.encode_text(text_inputs)
#     logits_per_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#     print("Original image probabilities:", logits_per_image)

#     perturbed_features = model.encode_image(perturbed_data)
#     perturbed_logits = (100.0 * perturbed_features @ text_features.T).softmax(dim=-1)
#     print("Perturbed image probabilities:", perturbed_logits)


if __name__ == "__main__":
    image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw).convert("RGB")

    target = 281 # tabby cat in ImageNet
    epsilon = 0.02
    perturbed = get_fgsm_resnet_imagenet(image, target, epsilon)

    # Display original and perturbed images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(perturbed)
    plt.title("Perturbed Image")
    plt.axis('off')

    plt.show()

