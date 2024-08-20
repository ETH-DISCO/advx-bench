import random
import string

import cairo
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_perlin_background(
    width=1000,
    height=700,
    pixel_size=7,
    colors=[(0.8, 0.7, 0.6), (0.7, 0.8, 0.6), (0.6, 0.7, 0.8), (0.8, 0.6, 0.7), (0.7, 0.6, 0.8)],
):
    import noise

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgba(0, 0, 0, 0)

    # random offset
    x_offset = random.uniform(0, 1000)
    y_offset = random.uniform(0, 1000)

    def combined_noise(x, y):
        perlin = noise.pnoise2((x + x_offset) / 100.0, (y + y_offset) / 100.0, octaves=8, persistence=0.5)
        simplex = noise.snoise2((x + x_offset) / 80.0, (y + y_offset) / 80.0, octaves=6)
        return (perlin + simplex) / 2

    perlin_noise = np.fromfunction(np.vectorize(combined_noise), (height, width))

    # normalize noise values to [0, 1]
    perlin_noise = (perlin_noise - perlin_noise.min()) / (perlin_noise.max() - perlin_noise.min())

    for y in range(0, height, pixel_size):
        for x in range(0, width, pixel_size):
            noise_value = perlin_noise[y // pixel_size][x // pixel_size]
            color_idx = int(noise_value * (len(colors) - 1))
            ctx.set_source_rgb(*colors[color_idx])
            ctx.rectangle(x, y, pixel_size, pixel_size)
            ctx.fill()

    img = Image.frombuffer("RGBA", (width, height), surface.get_data(), "raw", "BGRA", 0, 1)
    return img


def get_zigzag_background(
    width=1000,
    height=700,
    pixel_size=7,
    num_colors=5,
):
    from scipy.ndimage import gaussian_filter

    def generate_color_map(num_colors):
        return np.random.rand(num_colors, 3)

    def generate_noise(width, height, scale=100):
        x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        noise = np.sin(x * scale) * np.sin(y * scale)
        noise += np.sin(x * scale * 2) * np.sin(y * scale * 2) * 0.5
        noise += np.sin(x * scale * 4) * np.sin(y * scale * 4) * 0.25
        return (noise - noise.min()) / (noise.max() - noise.min())

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    context = cairo.Context(surface)

    color_map = generate_color_map(num_colors)
    noise = generate_noise(width, height)
    smoothed_noise = gaussian_filter(noise, sigma=3)

    thresholds = np.linspace(0, 1, num_colors + 1)
    color_indices = np.digitize(smoothed_noise, thresholds[1:-1]) - 1

    for x in range(0, width, pixel_size):
        for y in range(0, height, pixel_size):
            color_index = color_indices[y // pixel_size, x // pixel_size]
            color = color_map[color_index]
            context.set_source_rgba(color[0], color[1], color[2], 1)
            context.rectangle(x, y, pixel_size, pixel_size)
            context.fill()

    img = Image.frombuffer("RGBA", (width, height), surface.get_data(), "raw", "BGRA", 0, 1)

    border_noise = generate_noise(width, height, scale=200)
    border_mask = (border_noise > 0.7).astype(float)
    border_mask = gaussian_filter(border_mask, sigma=1)

    img_array = np.array(img)
    sand_color = np.array([0.8, 0.7, 0.6, 1])
    img_array = img_array * (1 - border_mask[:, :, np.newaxis]) + sand_color * 255 * border_mask[:, :, np.newaxis]
    img = Image.fromarray(img_array.astype("uint8"))

    small_image = img.resize((width // pixel_size, height // pixel_size), Image.NEAREST)
    img = small_image.resize((width, height), Image.NEAREST)
    return img


def get_gradient_background(
    width=1000,
    height=700,
    num_colors=3,
):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)

    colors = [(random.random(), random.random(), random.random()) for _ in range(num_colors)]
    pat = cairo.LinearGradient(0.0, 0.0, 0.0, height)
    num_colors = len(colors)

    for i, color in enumerate(colors):
        pat.add_color_stop_rgb(i / (num_colors - 1), *color)

    ctx.rectangle(0, 0, width, height)
    ctx.set_source(pat)
    ctx.fill()

    img = Image.frombuffer("RGBA", (width, height), surface.get_data(), "raw", "BGRA", 0, 1)
    return img


def get_random_background(
    width=1000,
    height=700,
    num_colors=3,
    num_letters=500,
):
    import matplotlib

    matplotlib.use("Agg")  # matplotlib can't render fonts

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)

    colors = [(random.random(), random.random(), random.random()) for _ in range(num_colors)]
    pat = cairo.LinearGradient(0.0, 0.0, 0.0, height)
    num_colors = len(colors)

    for i, color in enumerate(colors):
        pat.add_color_stop_rgb(i / (num_colors - 1), *color)

    ctx.rectangle(0, 0, width, height)
    ctx.set_source(pat)
    ctx.fill()

    for _ in range(num_letters):
        letter = random.choice(string.ascii_letters + string.digits + string.punctuation)
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        font_size = random.uniform(10, 50)
        color = (random.random(), random.random(), random.random())

        ctx.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(font_size)
        ctx.move_to(x, y)
        ctx.set_source_rgb(*color)
        ctx.show_text(letter)

    img = Image.frombuffer("RGBA", (width, height), surface.get_data(), "raw", "BGRA", 0, 1)
    return img


"""
example usage
"""


if __name__ == "__main__":
    img = get_perlin_background()
    # img = get_zigzag_background()
    # img = get_gradient_background()
    # img = get_random_background()

    plt.imshow(img)
    plt.axis("off")
    plt.show()

    # img.save("background.png")
