import os
import glob
from pathlib import Path

datapath = Path.cwd().parent / 'data' / 'hcaptcha'
subdirs = [x for x in datapath.iterdir() if x.is_dir()]

all_files = [
    f for subdir in subdirs for f in
    glob.glob(str(subdir / '*.png'))
]

print(f"Found {len(all_files)} files in {len(subdirs)} subdirectories.")
