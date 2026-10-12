#!/usr/bin/env bash
# Kill team tmux sessions bao-trainning-2..10 (never touches bao-trainning-1 = PM)
for n in $(seq 2 10); do tmux kill-session -t "bao-trainning-$n" 2>/dev/null && echo "killed bao-trainning-$n"; done
