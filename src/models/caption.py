import requests
from PIL import Image

"""
models
"""


def caption_vqa(img: Image.Image) -> list[str]:
    from transformers import BlipForQuestionAnswering, BlipProcessor

    model_id = "Salesforce/blip-vqa-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(model_id)

    question = "what is in the image?"
    inputs = processor(img, question, return_tensors="pt")

    out = model.generate(**inputs)
    res = processor.decode(out[0], skip_special_tokens=True)

    assert isinstance(res, str)
    return get_noun_chunks(res)


def caption_gpt2(img: Image.Image) -> list[str]:
    from transformers import pipeline

    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    res = image_to_text(img)[0]["generated_text"]

    assert isinstance(res, str)
    return get_noun_chunks(res)


"""
utils
"""


def get_noun_chunks(sentence: str) -> list[str]:
    # $ python -m spacy download en_core_web_sm
    import spacy

    nlp = spacy.load("en_core_web_sm")
    sentence = "a photo of a cute little kitten laying on a blanket next to a dog laying on a bed"
    doc = nlp(sentence)
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    return noun_chunks


"""
example
"""

if __name__ == "__main__":
    labels = ["quirky kittens on a couch", "chaotic remote controls", "a work of art"]
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    print(caption_gpt2(img))
    print(caption_vqa(img))
