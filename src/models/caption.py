import requests
from PIL import Image

"""
models
"""

# def caption_blip(img: Image.Image) -> str:
#     from transformers import BlipProcessor, BlipForConditionalGeneration

#     model_id = "moranyanuka/blip-image-captioning-large-mocha"
#     processor = BlipProcessor.from_pretrained(model_id)
#     model = BlipForConditionalGeneration.from_pretrained(model_id)
#     inputs = processor(img, return_tensors="pt")
#     out = model.generate(**inputs)
#     label: str = processor.decode(out[0], skip_special_tokens=True)

#     # todo: convert to list of nouns and adjectives

#     return label

"""
example
"""

if __name__ == "__main__":
    labels = ["quirky kittens on a couch", "chaotic remote controls", "a work of art"]
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # print(caption_blip(image))
