from pathlib import Path

import cairo

WIDTH, HEIGHT = 1000, 1000

surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
context = cairo.Context(surface)
context.set_source_rgba(0, 0, 0, 0)


def draw_concentric_squares(x, y, size, num_squares):
    step = size / num_squares
    for i in range(num_squares):
        if i == 0:
            context.set_source_rgb(0, 0, 1)
        elif i == num_squares - 1:
            context.set_source_rgb(1, 0, 0)
        else:
            brown = 0.6 - (i / num_squares) * 0.4
            context.set_source_rgb(brown, brown * 0.7, 0)

        width = size - i * step
        height = size - i * step
        context.rectangle(x + (size - width) / 2, y + (size - height) / 2, width, height)
        context.stroke()


grid_width = 3
num_squares = 10

for row in range(grid_width):
    for col in range(grid_width):
        x = col * (WIDTH // grid_width)
        y = row * (HEIGHT // grid_width)
        draw_concentric_squares(x, y, WIDTH // grid_width, num_squares)

surface.write_to_png(Path("data/maskgen/square.png"))
