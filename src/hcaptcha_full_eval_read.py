from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file


def load_and_decode_safetensor(file_path):
    data_dict = load_file(file_path)
    for key, tensor in data_dict.items():
        print(f"\n{key}:")

        if key == "image":
            # convert image tensor to PIL Image and display
            img = Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            img.show()
            print(f"Image shape: {tensor.shape}")

        elif key == "captions" or key == "detection_labels":
            # decode int32 tensors to strings
            def decode_int32_to_string(tensor):
                return "".join(chr(i) for i in tensor.tolist())

            decoded = decode_int32_to_string(tensor)
            print(decoded.split("|"))

        elif key == "classification_probs" or key == "detection_scores":
            print(tensor.tolist())

        elif key == "detection_boxes":
            print(tensor.tolist())

        elif key == "segmentation_masks":
            print(f"Segmentation masks shape: {tensor.shape}")
            # you might want to visualize these masks, but it depends on how you want to display them
        else:
            print(tensor)


datapath = Path.cwd() / "data" / "hcaptcha" / "seg" / "eval"
assert datapath.exists()

for file in datapath.glob("*.safetensors"):
    print(f"Data from {file.stem}:")
    data_dict = load_file(file)
    for key, value in data_dict.items():
        print(f"{key}: {value.shape if hasattr(value, 'shape') else value}")
        print(value)

    load_and_decode_safetensor(file)

    print("-" * 20)
