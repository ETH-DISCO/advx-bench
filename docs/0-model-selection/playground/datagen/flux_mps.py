import os

import torch
from diffusers import FluxPipeline

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
torch.mps.empty_cache()

ckpt_id = "black-forest-labs/FLUX.1-schnell"
prompt = [
    "an astronaut riding a horse",
]
height, width = 256, 256

# denoising
pipe = FluxPipeline.from_pretrained(
    ckpt_id,
    torch_dtype=torch.bfloat16,
)
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
pipe.to("mps")

image = pipe(
    prompt,
    num_inference_steps=1,
    guidance_scale=0.0,
    height=height,
    width=width,
).images[0]

import matplotlib.pyplot as plt

plt.imshow(image)
plt.show()
