import torch
import requests
from PIL import Image
from utils import get_device


def get_clip_predictions(img: Image.Image, labels: list[str]) -> dict[str, float]:
    import clip

    device = get_device()
    model, preprocess = clip.load("ViT-L/14@336px", device=device)  # largest ViT
    model.eval()

    text = clip.tokenize(labels).to(device)
    image = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return {label: prob for label, prob in zip(labels, probs[0])}


def get_open_coca_predictions(img: Image.Image, labels: list[str]) -> dict[str, float]:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms('coca_ViT-L-14', pretrained='mscoco_finetuned_laion2b_s13b_b90k')
    model.eval()
    tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')

    image = preprocess(image).unsqueeze(0)
    text = tokenizer(labels)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return {label: prob for label, prob in zip(labels, probs[0])}


# TODO: eva

# TODO: gem


labels = ["quirky kittens on a couch", "chaotic remote controls", "a work of art"]

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)


print(get_clip_predictions(image, labels))
print(get_open_coca_predictions(image, labels))
