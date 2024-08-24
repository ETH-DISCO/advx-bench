# waiting for sam2 to be ported: https://github.com/huggingface/transformers/pull/32394

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import Sam2ImagePredictor

predictor = Sam2ImagePredictor.from_pretrained("facebook/sam2-hiera-large-hf")

img = np.random.rand(1080, 920, 3).astype(np.float32)
# prompt input
points = np.array([[100, 100]], dtype=np.float32)  # Dummy point
labels = np.array([1], np.int32)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(img)
    output = predictor.predict(
        point_coords=points,
        point_labels=labels,
    )
    masks, scores, low_res_masks = output.to_tuple()

    # plot the masks
    for mask in masks:
        plt.imshow(mask.cpu().numpy())
        plt.show()
