from pathlib import Path

from safetensors import safe_open
from tqdm import tqdm

def read_safetensors(file):
    tensors = {}
    with safe_open(file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


datapath = Path.cwd() / "data" / "hcaptcha-eval"
assert datapath.exists()
files = [x for x in datapath.iterdir() if x.is_file()]

for file in tqdm(files):
    tensors = read_safetensors(file)
    for key, tensor in tensors.items():
        print(f"{key}: {tensor}")
    break

