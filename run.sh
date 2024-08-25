export OPENAI_API_KEY="sk-xxxxxx"


python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

pip install git+https://github.com/openai/CLIP.git
pip install open-clip-torch
python -m spacy download en_core_web_sm
pip install clip diffusers matplotlib numpy opencv_python opencv_python_headless Pillow Requests spacy torch transformers accelerate

# stay alive
python_file="./src/4-eval_cls_perturb.py"
chmod +x run-monitor.sh
nohup ./run-monitor.sh "$python_file" > run-monitor.log 2>&1 & echo $! > "run-monitor.pid"

# manual eval
# nohup $PWD/.venv/bin/python3 "$python_file" > run.log 2>&1 & echo $! > run.pid

watch -n 0.1 "tail -n 100 run.log"
pgrep -f "4-eval"
