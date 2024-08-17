import matplotlib.pyplot as plt
import math
from pathlib import Path
from typing import Optional
import cairo
from PIL import Image

def gen_circle_image(
    width: int = 1000,
    height: int = 1000,
    row_count: int = 3,
    ring_count: int = 12,
    max_radius: Optional[int] = None,
) -> Image.Image:
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    context = cairo.Context(surface)
    context.set_source_rgba(0, 0, 0, 0)

    max_radius = width / 2 / row_count if max_radius is None else max_radius

    def draw_concentric_circles(x, y, ring_count):
        for i in range(ring_count):
            radius = max_radius * (i + 1) / ring_count
            context.arc(x, y, radius, 0, 2 * math.pi)

            color_ratio = i / (ring_count - 1)
            if color_ratio < 0.10:
                # red
                rgb_color = (1, 0, 0)
            elif color_ratio < 0.5:
                # gray
                rgb_color = (1, 1, 0)
            elif color_ratio < 0.75:
                # yellow
                rgb_color = (0.5, 0.5, 0.5)
            else:
                # blue
                rgb_color = (0, 0, 1)

            context.set_source_rgb(*rgb_color)
            context.set_line_width(1.5)
            context.stroke()

    for row in range(row_count):
        for col in range(row_count):
            x = (col + 0.5) * width / row_count
            y = (row + 0.5) * width / row_count
            draw_concentric_circles(x, y, ring_count)

    # surface.write_to_png(Path("circles.png"))
    return Image.frombuffer("RGBA", (width, height), surface.get_data(), "raw", "BGRA", 0, 1)

if __name__ == "__main__":
    img = gen_circle_image()
    plt.imshow(img)
    plt.axis("off")
    plt.show()
