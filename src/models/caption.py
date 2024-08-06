import requests
from PIL import Image

"""
models
"""


def caption_vqa(img: Image.Image) -> str:
    from transformers import BlipForQuestionAnswering, BlipProcessor

    model_id = "Salesforce/blip-vqa-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(model_id)

    question = "how many dogs are in the picture?"
    inputs = processor(img, question, return_tensors="pt")

    out = model.generate(**inputs)
    res = processor.decode(out[0], skip_special_tokens=True)

    assert isinstance(res, str)
    return res


def caption_gpt2(img: Image.Image) -> str:
    from transformers import pipeline

    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    res = image_to_text(img)[0]["generated_text"]

    assert isinstance(res, str)
    return res


"""
example
"""

if __name__ == "__main__":
    labels = ["quirky kittens on a couch", "chaotic remote controls", "a work of art"]
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    print(caption_gpt2(img))
    print(caption_vqa(img))
