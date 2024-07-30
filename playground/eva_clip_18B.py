"""
don't run, too large
this completely fried my laptop
"""

import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor, CLIPTokenizer

image_path = "../data/kodak/kodim.01.png"
model_name_or_path = "BAAI/EVA-CLIP-8B"  # or /path/to/local/EVA-CLIP-8B
image_size = 224

processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True).to(device).eval()

image = Image.open(image_path)
captions = ["a diagram", "a dog", "a cat"]
tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path)
input_ids = tokenizer(captions, return_tensors="pt", padding=True).input_ids.to("cuda")
input_pixels = processor(images=image, return_tensors="pt", padding=True).pixel_values.to("cuda")

with torch.no_grad():
    image_features = model.encode_image(input_pixels)
    text_features = model.encode_text(input_ids)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

label_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print(f"Label probs: {label_probs}")
