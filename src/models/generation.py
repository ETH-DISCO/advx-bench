from PIL import Image

try:
    from .utils import get_device
except ImportError:
    from utils import get_device

"""
models
"""


def gen_stable_diffusion(prompt: str) -> Image.Image:
    # best model, while waiting for flux-v1 to be ported to huggingface
    import torch
    from diffusers import StableDiffusionPipeline

    device = get_device()
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    image = pipe(prompt).images[0].cpu().detach().numpy()
    return image


"""
example
"""

if __name__ == "__main__":
    prompt = "a photo of an astronaut riding a horse on mars"
    image = gen_stable_diffusion(prompt)
    image.show()
