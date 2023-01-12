import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "0,1"

import json

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

def main():
    multigpu = int(os.getenv("MULTI_GPU")) == 1 if os.getenv("MULTI_GPU") else 0

    if multigpu:
        print("Using multpgpu")
        n_cuda = os.getenv("CUDA_VISIBLE_DEVICES") if os.getenv("CUDA_VISIBLE_DEVICES") else 0
        print(f"Ncuda = {n_cuda}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
    else:
        print("Using single gpu")
        n_cuda = os.getenv('N_CUDA')if os.getenv('N_CUDA') else "0"
        print(f"Ncuda = {n_cuda}")

        device = torch.device("cuda:" + (n_cuda) if torch.cuda.is_available() else "cpu")
        print(device)

    if torch.cuda.is_available():
        print(f"Training in {torch.cuda.get_device_name(0)}" )  
        print(f"Current cuda device {torch.cuda.current_device()}")
        print(f"Number of devices {torch.cuda.device_count()}")
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    else:
        print("Training in CPU")

if __name__ == '__main__':
    main()