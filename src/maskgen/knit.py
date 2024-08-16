from pathlib import Path

import cairo

WIDTH, HEIGHT = 1000, 1000

surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
context = cairo.Context(surface)
context.set_source_rgba(0, 0, 0, 0)


def draw_knit(x, y, size, color):
    context.set_source_rgb(*[int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)])
    context.move_to(x, y - size)
    context.line_to(x + size, y)
    context.line_to(x, y + size)
    context.line_to(x - size, y)
    context.close_path()
    context.stroke()


STEP = 35
KNIT_COLORS = ["#0000FF", "#008000", "#804000", "#FF0000"]  # Blue, Green, Brown, Red

for x in range(0, WIDTH + STEP, STEP):
    for y in range(0, HEIGHT + STEP, STEP):
        for i, color in enumerate(KNIT_COLORS):
            size = STEP - (i * STEP / 4)
            draw_knit(x, y, size, color)

surface.write_to_png(Path("src/maskgen/knit.png"))
