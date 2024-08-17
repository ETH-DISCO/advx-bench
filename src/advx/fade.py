import numpy as np
import requests
from PIL import Image


def get_rounded_corners(img: Image.Image, fraction: float = 0.30) -> Image.Image:
    fraction = 0.5

    width, height = img.size
    center_radius = min(width, height) * fraction
    mask = Image.new("L", (width, height), 0)

    for y in range(height):
        for x in range(width):
            distance = np.sqrt((x - width // 2) ** 2 + (y - height // 2) ** 2)
            alpha = max(0, 255 - int((distance / center_radius) * 255))
            mask.putpixel((x, y), alpha)

    img_with_transparency = Image.new("RGBA", img.size)
    for y in range(height):
        for x in range(width):
            img_with_transparency.putpixel((x, y), img.getpixel((x, y))[:-1] + (mask.getpixel((x, y)),))

    return img_with_transparency


if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGBA")
    
    img = get_rounded_corners(img)
    img.show()
