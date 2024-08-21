# run nohup
source .venv/bin/activate
nohup $PWD/.venv/bin/python3 $PWD/src/2-eval_cls_mask.py > output.log 2>&1 &

# check process
ps aux | grep advx-bench
htop
