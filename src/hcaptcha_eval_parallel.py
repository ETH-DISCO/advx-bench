from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from models.caption import caption_blip
from models.cls import classify_metaclip
from models.det import detect_vit
from models.seg import segment_sam1


def process_image(args):
    file, categoryname, outputpath = args
    filename = file.stem

    img = Image.open(file)
    threshold = 0.1

    print("filename:", filename, "- stage 0")
    captions: list[str] = caption_blip(img)
    print("filename:", filename, "- stage 1")
    probs: list[float] = classify_metaclip(img, captions)
    print("filename:", filename, "- stage 2")
    detection: tuple[list[list[float]], list[float], list[str]] = detect_vit(img, captions, threshold)
    print("filename:", filename, "- stage 3")
    boxes, scores, labels = detection
    masks: torch.Tensor = segment_sam1(img, boxes)
    print("filename:", filename, "- stage 4")

    def ensure_contiguous(tensor):
        if len(tensor) == 0:
            return torch.empty(0)
        elif tensor.is_contiguous():
            return tensor
        else:
            return tensor.contiguous()

    img_tensor = ensure_contiguous(torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0)
    data_dict = {
        "image": img_tensor,
        "captions": ensure_contiguous(torch.tensor([ord(c) for c in "|".join(captions)], dtype=torch.int32)),
        "classification_probs": ensure_contiguous(torch.tensor(probs, dtype=torch.float32)),
        "detection_boxes": ensure_contiguous(torch.tensor(boxes, dtype=torch.float32)),
        "detection_scores": ensure_contiguous(torch.tensor(scores, dtype=torch.float32)),
        "detection_labels": ensure_contiguous(torch.tensor([ord(c) for c in "|".join(labels)], dtype=torch.int32)),
        "segmentation_masks": ensure_contiguous(masks),
    }
    save_file(data_dict, outputpath / f"{categoryname}_{filename}.safetensors")
    print(f"processed: `{categoryname}_{filename}.safetensors`")


def main():
    datapath = Path.cwd() / "data" / "hcaptcha"
    outputpath = Path.cwd() / "data" / "hcaptcha-eval"
    assert datapath.exists()
    subdirs = [x for x in datapath.iterdir() if x.is_dir()]

    work_items = []
    for subdir in subdirs:
        categoryname = subdir.name
        for file in subdir.glob("*.png"):

            # check if equivalent file exists in outputpath
            if (outputpath / f"{categoryname}_{file.stem}.safetensors").exists():
                print(f"skipping: `{categoryname}_{file.stem}.safetensors`")
                continue
            
            work_items.append((file, categoryname, outputpath))

    with Pool(processes=cpu_count()) as pool:
        pool.map(process_image, work_items)


if __name__ == "__main__":
    main()
