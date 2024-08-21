pip install git+https://github.com/openai/CLIP.git
pip install open-clip-torch
python -m spacy download en_core_web_sm
pip install clip diffusers matplotlib numpy opencv_python opencv_python_headless Pillow Requests spacy torch transformers accelerate

# run
python3 -m venv .venv
source .venv/bin/activate
nohup $PWD/.venv/bin/python3 $PWD/src/XYZ.py > output.log 2>&1 &
nohup $PWD/.venv/bin/python3 $PWD/src/2-eval_cls_mask.py > output.log 2>&1 &

# check progress
ps aux | grep advx-bench
htop
