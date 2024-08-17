import math
import random
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend instead of the default
import matplotlib.pyplot as plt
import cairo
import numpy as np

WIDTH, HEIGHT = 1000, 1000

surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
context = cairo.Context(surface)
context.set_source_rgba(0, 0, 0, 0)
context.paint()

words = ["banana", "apple", "orange", "grape", "pear", "donkey", "elephant", "giraffe", "hippopotamus", "kangaroo"]

num_words = 15
font_range = (20, 100)

for i in range(num_words):
    x = random.randint(0, WIDTH)
    y = random.randint(0, HEIGHT)
    context.set_font_size(random.randint(*font_range))

    word = random.choice(words)
    orientation = random.choice(["horizontal", "vertical", "flipped"])

    context.save()
    context.translate(x, y)

    if orientation == "vertical":
        context.rotate(-math.pi / 2)
    elif orientation == "flipped":
        context.rotate(random.uniform(0, 2 * math.pi))
        context.scale(-1 if random.random() > 0.5 else 1, -1 if random.random() > 0.5 else 1)

    grayshade = random.random()
    context.set_source_rgb(grayshade, grayshade, grayshade)

    context.move_to(0, 0)
    context.show_text(word)

    context.restore()

# Convert the Cairo surface to a numpy array
buf = surface.get_data()
data = np.ndarray(shape=(HEIGHT, WIDTH, 4), dtype=np.uint8, buffer=buf)

# Create a new figure and display the image
plt.figure(figsize=(10, 10))
plt.imshow(data)
plt.axis('off')
