python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

pip install git+https://github.com/openai/CLIP.git
pip install open-clip-torch
python -m spacy download en_core_web_sm
pip install clip diffusers matplotlib numpy opencv_python opencv_python_headless Pillow Requests spacy torch transformers accelerate


export OPENAI_API_KEY="sk-xxxxxx"

nohup $PWD/.venv/bin/python3 $PWD/src/3-eval_cls_perturb.py > output.log 2>&1 &

watch cat output.log
htop
