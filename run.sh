python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

pip install git+https://github.com/openai/CLIP.git
pip install open-clip-torch
python -m spacy download en_core_web_sm
pip install clip diffusers matplotlib numpy opencv_python opencv_python_headless Pillow Requests spacy torch transformers accelerate


make monitor filepath="./src/3-train_cls_robustified_clip_vit.py"

# nohup .venv/bin/python3 src/XYZ.py >> "monitor-process.log" 2>&1 &
