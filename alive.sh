#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_python_file>"
    exit 1
fi

python_file="$1"
pid_file="process.pid"

while true; do
    if [ ! -f "$pid_file" ] || ! ps -p $(cat "$pid_file") > /dev/null; then
        echo "starting/restarting process..."
        nohup $PWD/.venv/bin/python3 "$python_file" > output.log 2>&1 & echo $! > "$pid_file"
    fi
    sleep 60
done
