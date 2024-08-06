import requests
from PIL import Image

"""
models
"""

def caption_blip(img: Image.Image) -> str:
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained("moranyanuka/blip-image-captioning-large-mocha")
    model = BlipForConditionalGeneration.from_pretrained("moranyanuka/blip-image-captioning-large-mocha")
    inputs = processor(img, return_tensors="pt")
    out = model.generate(**inputs)
    label = processor.decode(out[0], skip_special_tokens=True)
    assert isinstance(label, str)
    return label

"""
example
"""

if __name__ == "__main__":
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    print(caption_blip(raw_image))
