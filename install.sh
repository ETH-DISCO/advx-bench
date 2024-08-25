export OPENAI_API_KEY="sk-xxxxxx"


python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

pip install git+https://github.com/openai/CLIP.git
pip install open-clip-torch
python -m spacy download en_core_web_sm
pip install clip diffusers matplotlib numpy opencv_python opencv_python_headless Pillow Requests spacy torch transformers accelerate


python_file="./src/4-eval_cls_perturb.py"
pid_file="eval.pid"

# initial start
nohup $PWD/.venv/bin/python3 "$python_file" > output.log 2>&1 & echo $! > "$pid_file"

# endless loop to monitor and restart the process
while true; do
    if ! ps -p $(cat "$pid_file") > /dev/null; then
        echo "Process died, restarting..."
        nohup $PWD/.venv/bin/python3 "$python_file" > output.log 2>&1 & echo $! > "$pid_file"
    fi
    sleep 5 # check every 5 seconds
done


watch -n 0.1 "tail -n 100 output.log"
