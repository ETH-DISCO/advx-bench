import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from models.utils import get_device

model = AutoModel.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device=get_device())

tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True)
model.eval()

import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
question = "What is in the image?"
msgs = [{"role": "user", "content": question}]

res = model.chat(
    image=image,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7,
    # system_prompt=''
)
print(res)


# res = model.chat(
#     image=image,
#     msgs=msgs,
#     tokenizer=tokenizer,
#     sampling=True,
#     temperature=0.7,
#     stream=True
# )

# generated_text = ""
# for new_text in res:
#     generated_text += new_text
#     print(new_text, flush=True, end='')
