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
    # python -m spacy download en_core_web_sm
    import spacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    return noun_chunks


"""
example
"""

if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    text_query = caption_gpt2(img)
    print(text_query)

    text_query = caption_vqa(img)
    print(text_query)
