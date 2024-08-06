import requests
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

# Load model and processor
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Load images
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_target = Image.open(requests.get(url, stream=True).raw)
query_url = "http://images.cocodataset.org/val2017/000000524280.jpg"
query_image = Image.open(requests.get(query_url, stream=True).raw)

# Prepare inputs
inputs = processor(images=image_target, query_images=query_image, return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model.image_guided_detection(**inputs)

# Process outputs
target_sizes = torch.Tensor([image_target.size[::-1]])
results = processor.post_process_image_guided_detection(outputs=outputs, target_sizes=target_sizes)

# plot
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.imshow(image_target)
plt.axis("off")
