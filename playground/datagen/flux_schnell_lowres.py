# pip install git+https://github.com/huggingface/diffusers.git
# pip install sentencepiece
# pip install --upgrade transformers diffusers torch
# pip uninstall torch torchvision
# pip install torch torchvision

import torch
from diffusers import FluxPipeline

model_id = "black-forest-labs/FLUX.1-schnell"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cpu")

prompt = "A cat holding a sign that says hello world"
seed = 42
image = pipe(
    prompt,
    output_type="pil",
    num_inference_steps=4,
    generator=torch.Generator("cpu").manual_seed(seed)
).images[0]
image.save("flux-schnell.png")
