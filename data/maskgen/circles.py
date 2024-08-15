import math
from pathlib import Path

import cairo

WIDTH, HEIGHT = 1000, 1000

surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
context = cairo.Context(surface)
context.set_source_rgba(0, 0, 0, 0)


def draw_concentric_circles(x, y, max_radius, num_rings):
    for i in range(num_rings):
        radius = max_radius * (i + 1) / num_rings
        context.arc(x, y, radius, 0, 2 * math.pi)

        color_ratio = i / (num_rings - 1)
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


count = 3
num_rings = 12
max_radius = WIDTH / 2 / count

for row in range(count):
    for col in range(count):
        x = (col + 0.5) * WIDTH / count
        y = (row + 0.5) * WIDTH / count
        draw_concentric_circles(x, y, max_radius, num_rings)

surface.write_to_png(Path("data/maskgen/circles.png"))
