# Spoter-SL


## Run model 

    hidden_dim = POINTS*2 = 54*2 = 108


    python train.py --epoch 200 --experiment_name 305-aec --lr 0.0001 --num_rows 64 --num_heads 9 --num_layers_1 9 --num_layers_2 9 --dim_feedforward 128 --device 1,2

## Run model in background

https://medium.com/swlh/introduction-to-process-handling-in-cmd-and-using-terminal-multiplexers-for-uninterrupted-bfd1bf2c16c2


To exit a tmux session, press control and ‘B’ together. Then, press ‘D’ to detach yourself from the session. You can also create sessions with a specific name using the following command:

    tmux new -s '<name>'
    

    tmux new -s session_01
  
You can list all running tmux sessions using ‘tmux ls’. You can attach your window to a specific session using its name in the following command:

    tmux a -t ‘<name>’
    tmux a -t session_01

You can also kill a particular session using the following command:

    tmux kill-session -t ‘<name>’
    tmux kill-session -t session_01


# Run model with sweep

    wandb login
    cd /ruta/a/Spoter-SL  # Cambia esto a la ruta real de tu proyecto
    wandb init

    wandb sweep config.yaml
    wandb agent ml_projects/SLR_2023/zdf5fe5t

# Check gpu memory

    watch -n 1 nvidia-smi

