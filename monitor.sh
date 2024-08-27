filepath="$PWD/src/3-eval_cls_robustified_clip_vit.py"

monitor() {
    while true; do
        if ! pgrep -f "$filepath" > /dev/null; then
            echo "$(date): process died, restarting..." >> monitor.log
            rm -rf "monitor-process.log"
            rm -rf "monitor-process.pid"
            python3 "$filepath" >> "monitor-process.log" 2>&1 &
            echo $! > "monitor-process.pid"
        fi
        sleep 5
    done
}
monitor >> "monitor.log" 2>&1 &
echo $! > "monitor.pid"
echo "$(date): started" >> "monitor.log"

# watch
watch -n 0.1 "tail -n 100 monitor-process.log"
while true; do clear; tail -n 100 monitor-process.log; sleep 0.1; done
pgrep -f "$filepath"
nvtop
htop

# kill
kill $(cat "monitor-process.pid")
rm -f monitor-process.log
rm -f monitor-process.pid
kill $(cat "monitor.pid")
rm -f monitor.log
rm -f monitor.pid
