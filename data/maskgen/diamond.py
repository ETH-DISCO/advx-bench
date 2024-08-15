from pathlib import Path

import cairo

WIDTH, HEIGHT = 1000, 1000

surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
context = cairo.Context(surface)
context.set_source_rgba(0, 0, 0, 0)


def draw_diamond(x, y, size, color):
    context.set_source_rgba(*color)
    context.move_to(x, y - size / 2)
    context.line_to(x + size / 2, y)
    context.line_to(x, y + size / 2)
    context.line_to(x - size / 2, y)
    context.close_path()
    context.stroke()


def get_color(i, max_i):
    if i == max_i:
        return (0, 0, 1, 1)  # blue for outermost diamond
    elif i == 1:
        return (1, 0, 0, 1)  # red for innermost diamond
    else:
        t = (i - 1) / (max_i - 1)  # gradient between blue and red
        return (0, 0.5 * (1 - t), 0, 1)


diamond_size = 300
num_diamonds = 10

for row in range(-1, HEIGHT // int(diamond_size / 2) + 2):
    for col in range(-1, WIDTH // diamond_size + 2):
        center_x = col * diamond_size
        center_y = row * int(diamond_size / 2)

        # row offset
        if row % 2 == 1:
            center_x += diamond_size / 2

        for i in range(num_diamonds, 0, -1):
            size = diamond_size * i / num_diamonds
            color = get_color(i, num_diamonds)
            draw_diamond(center_x, center_y, size, color)


surface.write_to_png(Path("data/maskgen/diamond.png"))
