from transformers import pipeline

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

res = image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png")[0]["generated_text"]

print(res)
