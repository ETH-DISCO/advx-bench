python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

pip install git+https://github.com/openai/CLIP.git
pip install open-clip-torch
python -m spacy download en_core_web_sm
pip install clip diffusers matplotlib numpy opencv_python opencv_python_headless Pillow Requests spacy torch transformers accelerate


# ---

# filepath="$PWD/src/XYZ.py"

# # nohup .venv/bin/python3 "$filepath" >> "monitor-process.log" 2>&1 &
# monitor() {
#     while true; do
#         if ! pgrep -f "$filepath" > /dev/null; then
#             echo "$(date): process died, restarting..." >> monitor.log 
#             rm -rf "monitor-process.log"
#             rm -rf "monitor-process.pid"
#             .venv/bin/python3 "$filepath" >> "monitor-process.log" 2>&1 &
#             echo $! > "monitor-process.pid"
#         fi
#         sleep 5
#     done
# }
# monitor >> "monitor.log" 2>&1 &
# echo $! > "monitor.pid"
# echo "$(date): started" >> "monitor.log"

# # watch
# watch -n 0.1 "tail -n 100 monitor-process.log"
# while true; do clear; tail -n 100 monitor-process.log; sleep 0.1; done
# pgrep -f "$filepath"
# nvtop
# htop

# # kill
# kill $(cat "monitor-process.pid")
# rm -f monitor-process.log
# rm -f monitor-process.pid
# kill $(cat "monitor.pid")
# rm -f monitor.log
# rm -f monitor.pid


# ---

make monitor filepath="./src/3-eval_cls_robustified_clip_vit.py"
