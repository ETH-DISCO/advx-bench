import matplotlib.pyplot as plt
import numpy as np
import torch
import requests
import random
from PIL import Image


"""
plot utils
"""

def get_random_color(num_colors: int) -> list[str]:
    named_css_colors = [
        "aliceblue",
        "antiquewhite",
        "aqua",
        "aquamarine",
        "azure",
        "beige",
        "bisque",
        "black",
        "blanchedalmond",
        "blue",
        "blueviolet",
        "brown",
        "burlywood",
        "cadetblue",
        "chartreuse",
        "chocolate",
        "coral",
        "cornflowerblue",
        "cornsilk",
        "crimson",
        "cyan",
        "darkblue",
        "darkcyan",
        "darkgoldenrod",
        "darkgray",
        "darkgreen",
        "darkgrey",
        "darkkhaki",
        "darkmagenta",
        "darkolivegreen",
        "darkorange",
        "darkorchid",
        "darkred",
        "darksalmon",
        "darkseagreen",
        "darkslateblue",
        "darkslategray",
        "darkslategrey",
        "darkturquoise",
        "darkviolet",
        "deeppink",
        "deepskyblue",
        "dimgray",
        "dimgrey",
        "dodgerblue",
        "firebrick",
        "floralwhite",
        "forestgreen",
        "fuchsia",
        "gainsboro",
        "ghostwhite",
        "gold",
        "goldenrod",
        "gray",
        "green",
        "greenyellow",
        "grey",
        "honeydew",
        "hotpink",
        "indianred",
        "indigo",
        "ivory",
        "khaki",
        "lavender",
        "lavenderblush",
        "lawngreen",
        "lemonchiffon",
        "lightblue",
        "lightcoral",
        "lightcyan",
        "lightgoldenrodyellow",
        "lightgray",
        "lightgreen",
        "lightgrey",
        "lightpink",
        "lightsalmon",
        "lightseagreen",
        "lightskyblue",
        "lightslategray",
        "lightslategrey",
        "lightsteelblue",
        "lightyellow",
        "lime",
        "limegreen",
        "linen",
        "magenta",
        "maroon",
        "mediumaquamarine",
        "mediumblue",
        "mediumorchid",
        "mediumpurple",
        "mediumseagreen",
        "mediumslateblue",
        "mediumspringgreen",
        "mediumturquoise",
        "mediumvioletred",
        "midnightblue",
        "mintcream",
        "mistyrose",
        "moccasin",
        "navajowhite",
        "navy",
        "oldlace",
        "olive",
        "olivedrab",
        "orange",
        "orangered",
        "orchid",
        "palegoldenrod",
        "palegreen",
        "paleturquoise",
        "palevioletred",
        "papayawhip",
        "peachpuff",
        "peru",
        "pink",
        "plum",
        "powderblue",
        "purple",
        "rebeccapurple",
        "red",
        "rosybrown",
        "royalblue",
        "saddlebrown",
        "salmon",
        "sandybrown",
        "seagreen",
        "seashell",
        "sienna",
        "silver",
        "skyblue",
        "slateblue",
        "slategray",
        "slategrey",
        "snow",
        "springgreen",
        "steelblue",
        "tan",
        "teal",
        "thistle",
        "tomato",
        "turquoise",
        "violet",
        "wheat",
        "white",
        "whitesmoke",
        "yellow",
        "yellowgreen",
    ]
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))



def plot_results(img: Image.Image, results: list[tuple[str, torch.Tensor]]):
    plt.imshow(img)
    plt.axis('off')

    colors = get_random_color(len(results))
    for (label, mask), color in zip(results, colors):
        mask_np = mask.squeeze().numpy()
        
        # resize the mask to match the original image dimensions
        mask_resized = np.array(Image.fromarray(mask_np).resize(img.size, Image.BILINEAR))
        
        color_mask = np.zeros((*mask_resized.shape, 4))
        color_mask[..., :3] = plt.cm.colors.to_rgb(color)
        color_mask[..., 3] = mask_resized * 0.5
        plt.imshow(color_mask)
        plt.contour(mask_resized, levels=[0.5], colors=[color], alpha=0.8)

    legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=label)
                         for (label, _), color in zip(results, colors)]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()


"""
models
"""

def get_clipseg_results(img: Image.Image, labels: list[str]) -> list[tuple[str, torch.Tensor]]:
    from transformers import AutoProcessor, CLIPSegForImageSegmentation

    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")


    inputs = processor(text=labels, images=[img] * len(labels), padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    masks = torch.sigmoid(logits)

    return list(zip(labels, masks))


def get_groundingsam_results(img: Image.Image, labels: list[str]):
    pass


labels = ["cat", "remote control"]

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img = Image.open(requests.get(url, stream=True).raw)

results = get_clipseg_results(img, labels)
plot_results(img, results)
