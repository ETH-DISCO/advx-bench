#!/bin/bash

# Usage: ./alive.sh <python_file>

python_file="$1"
pid_file="script.pid"

# start process
$PWD/.venv/bin/python3 "$python_file" > run.log 2>&1 & echo $! > run.pid

# restart after process dies
while true; do
    if ! ps -p $(cat "$pid_file") > /dev/null; then
        echo "process died, restarting..."
        $PWD/.venv/bin/python3 "$python_file" > run.log 2>&1 & echo $! > run.pid
    fi
    sleep 5 # check every 5 seconds
done
