import requests
from PIL import Image

"""
models
"""

def caption_blip(img: Image.Image) -> str:
    from transformers import BlipProcessor, BlipForConditionalGeneration

    model_id = "moranyanuka/blip-image-captioning-large-mocha"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    inputs = processor(img, return_tensors="pt")
    out = model.generate(**inputs)
    label: str = processor.decode(out[0], skip_special_tokens=True)

    # todo: convert to list of nouns and adjectives

    return label

"""
example
"""

if __name__ == "__main__":
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    print(caption_blip(image))
