import numpy as np
import requests
from PIL import Image


def get_rounded_corners(
    img: Image.Image,
    fraction: float = 0.3,
) -> Image.Image:
    width, height = img.size
    center_radius = min(width, height) * fraction
    mask = Image.new("L", (width, height), 255)  # 100% opaque

    for y in range(height):
        for x in range(width):
            distance = np.sqrt((x - width // 2) ** 2 + (y - height // 2) ** 2)
            if distance > center_radius:
                # calculate alpha only for pixels outside the circle
                alpha = int(255 * (1 - (distance - center_radius) / (min(width, height) / 2 - center_radius)))
                mask.putpixel((x, y), alpha)

    img_with_transparency = Image.new("RGBA", img.size)
    for y in range(height):
        for x in range(width):
            img_with_transparency.putpixel((x, y), img.getpixel((x, y))[:-1] + (mask.getpixel((x, y)),))

    return img_with_transparency


def place_within(
    background: Image.Image,
    inner: Image.Image,
    inner_ratio: float = 0.25,
    inner_position: tuple[int, int] = (-1, -1),
):
    target_area = inner_ratio * background.size[0] * background.size[1]
    aspect_ratio = inner.size[0] / inner.size[1]

    new_height = int(np.sqrt(target_area / aspect_ratio))
    new_width = int(aspect_ratio * new_height)

    inner_resized = inner.resize((new_width, new_height), Image.LANCZOS)

    paste_position = (inner_position[0] if inner_position[0] >= 0 else (background.size[0] - new_width) // 2, inner_position[1] if inner_position[1] >= 0 else (background.size[1] - new_height) // 2)
    result = background.copy()
    result.paste(inner_resized, paste_position, inner_resized)
    return result


if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGBA")

    background = img
    inner1 = get_rounded_corners(Image.new("RGBA", (100, 100), (255, 0, 0, 255)))
    inner2 = get_rounded_corners(Image.new("RGBA", (100, 100), (0, 255, 0, 255)))
    inner3 = get_rounded_corners(img)

    result = place_within(background, inner1, inner_position=(0, 0))
    result = place_within(result, inner2, inner_position=(200, 300))
    result = place_within(result, inner3, inner_position=(300, 0))
    result.show()
