# Spoter-SL


## Run model 

    
    python train.py --epoch 1 --experiment_name DGI305-AEC --lr 0.01 --num_rows 64 --hidden_dim 108 --num_heads 9 --num_layers_1 6 --num_layers_2 6 --dim_feedforward 256

## Run model in background

https://medium.com/swlh/introduction-to-process-handling-in-cmd-and-using-terminal-multiplexers-for-uninterrupted-bfd1bf2c16c2

To exit a tmux session, press control and ‘B’ together. Then, press ‘D’ to detach yourself from the session. You can also create sessions with a specific name using the following command:

    tmux new -s '<name>'
  
You can list all running tmux sessions using ‘tmux ls’. You can attach your window to a specific session using its name in the following command:

    tmux a -t ‘<name>’
  
You can also kill a particular session using the following command:

    tmux kill-session -t ‘<name>’


# Run model

    
    sh run_model_server.sh