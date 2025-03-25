#!/bin/bash

SESSION_NAME="open_loop_rosbag"

tmux new-session -d -s $SESSION_NAME

tmux split-window -v

tmux select-pane -t $SESSION_NAME:0.0
tmux send-keys -t $SESSION_NAME:0.0 'roscore' C-m

sleep 3

tmux select-pane -t $SESSION_NAME:0.1
tmux send-keys -t $SESSION_NAME:0.1 'rosbag play /rosbags/10.bag' C-m

tmux select-pane -t $SESSION_NAME:0.2
tmux send-keys -t $SESSION_NAME:0.2 'rosrun rviz rviz -d /configs/config.rviz' C-m

tmux select-pane -t $SESSION_NAME:0.3
tmux send-keys -t $SESSION_NAME:0.3 'conda activate priest && python3 open_loop_bag_ros1.py' C-m

tmux attach -t $SESSION_NAME