"""
check progress:

$ ls -al ./data/hcaptcha/cls/eval | wc -l
$ ls -al ./data/hcaptcha/cls/data | wc -l
"""

import json
import os
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from models.caption import caption_blip
from models.cls import classify_clip, classify_eva, classify_metaclip

datapath = Path.cwd() / "data" / "hcaptcha" / "cls" / "data"
outputpath = Path.cwd() / "data" / "hcaptcha" / "cls" / "eval"
outputpath.mkdir(parents=True, exist_ok=True)
assert datapath.exists()
assert outputpath.exists()

datafiles = list(datapath.glob("*.png"))
random.shuffle(datafiles)

for file in tqdm(datafiles):
    if (outputpath / f"{file.stem}.json").exists():
        print(f"skipping: `{file.stem}.json`")
        continue

    img: Image = Image.open(file)
    captions: list[str] = caption_blip(img)

    out = {
        "captions": captions,
        "metaclip": classify_metaclip(img, captions),
        "clip": classify_clip(img, captions),
        "eva": classify_eva(img, captions),
    }

    with open(outputpath / f"{file.stem}.json", "w") as f:
        json.dump(out, f, indent=4)

    # os.system(f"git add . && git commit -m 'autocommit' && git push")
