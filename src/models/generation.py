import os

import torch
from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "true"

try:
    from .utils import get_device
except ImportError:
    from utils import get_device


"""
models
"""


def gen_fluxv1(prompt: str) -> Image.Image:
    # best model for gpu
    assert torch.cuda.is_available(), "GPU not available"
    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
    # pipe.enable_model_cpu_offload() # uncomment if you have a weak GPU

    images = pipe(prompt, height=1024, width=1024, guidance_scale=3.5, num_inference_steps=50, max_sequence_length=512, generator=torch.Generator("cpu").manual_seed(0))
    img = images.images[0]
    assert isinstance(img, Image.Image)
    return img


def gen_stable_diffusion(prompt: str) -> Image.Image:
    # best model for cpu
    from diffusers import DiffusionPipeline

    device = get_device()
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)
    images = pipe(prompt=prompt).images
    img = images[0]
    assert isinstance(img, Image.Image)
    return img


def gen_stable_diffusion_light(prompt: str) -> Image.Image:
    import torch
    from diffusers import StableDiffusionPipeline

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(get_device())
    image = pipe(prompt).images[0]
    return image


"""
example usage
"""


if __name__ == "__main__":
    prompt = "a photo of an astronaut riding a horse on mars"
    image = gen_stable_diffusion(prompt)
    image.show()
