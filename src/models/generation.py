from PIL import Image

try:
    from .utils import get_device
except ImportError:
    from utils import get_device

"""
models
"""


def gen_stable_diffusion(prompt: str) -> Image.Image:
    import torch
    from diffusers import StableDiffusionPipeline

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(get_device())

    image = pipe(prompt).images[0]
    return image


"""
example
"""

if __name__ == "__main__":
    prompt = "a photo of an astronaut riding a horse on mars"
    image = gen_stable_diffusion(prompt)
    image.show()
