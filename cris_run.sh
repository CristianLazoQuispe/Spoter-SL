'''
# AEC-29
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=0 nohup python train.py --wandb --epoch 15000 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 1 --dim_feedforward 32 --device 0 --seed 1 --dataset AEC --keypoints_model mediapipe > log_AEC_29_00005_mediapipe.txt &
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=0 nohup python train.py --wandb --epoch 15000 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 1 --dim_feedforward 32 --device 0 --seed 1 --dataset AEC --keypoints_model wholepose > log_AEC_29_00005_wholepose.txt &
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=0 nohup python train.py --wandb --epoch 15000 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 1 --dim_feedforward 32 --device 0 --seed 1 --dataset AEC --keypoints_model openpose > log_AEC_29_00005_openpose.txt &

# AEC-71
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=1 nohup python train.py --wandb --epoch 15000 --learning_rate 0.00005 --keypoints_number 71 --num_coder_layers 1 --dim_feedforward 8 --device 0 --seed 1 --dataset AEC --keypoints_model mediapipe > log_AEC_71_00005-mediapipe.txt & 
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=1 nohup python train.py --wandb --epoch 15000 --learning_rate 0.00005 --keypoints_number 71 --num_coder_layers 1 --dim_feedforward 8 --device 0 --seed 1 --dataset AEC --keypoints_model wholepose > log_AEC_71_00005-wholepose.txt &
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=1 nohup python train.py --wandb --epoch 15000 --learning_rate 0.00005 --keypoints_number 71 --num_coder_layers 1 --dim_feedforward 8 --device 0 --seed 1 --dataset AEC --keypoints_model openpose > log_AEC_71_00005-openpose.txt &


#PUCP-29
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=0 nohup python train.py --wandb --epoch 20000 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 2 --dim_feedforward 32 --device 0 --seed 1 --dataset PUCP_PSL_DGI156 --keypoints_model mediapipe > ../logs_spoter/log_PUCP_29_00005-mediapipe.txt & 
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=0 nohup python train.py --wandb --epoch 20000 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 2 --dim_feedforward 32 --device 0 --seed 1 --dataset PUCP_PSL_DGI156 --keypoints_model wholepose > ../logs_spoter/log_PUCP_29_00005-wholepose.txt & 
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=0 nohup python train.py --wandb --epoch 20000 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 2 --dim_feedforward 32 --device 0 --seed 1 --dataset PUCP_PSL_DGI156 --keypoints_model openpose  > ../logs_spoter/log_PUCP_29_00005-openpose.txt & 

#PUCP-71
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=1 nohup python train.py --wandb --epoch 20000 --learning_rate 0.00005 --keypoints_number 71 --num_coder_layers 1 --dim_feedforward 32 --device 0 --seed 1 --dataset PUCP_PSL_DGI156 --keypoints_model mediapipe > ../logs_spoter/log_PUCP_71_00005-mediapipe.txt & 
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=1 nohup python train.py --wandb --epoch 20000 --learning_rate 0.00005 --keypoints_number 71 --num_coder_layers 1 --dim_feedforward 32 --device 0 --seed 1 --dataset PUCP_PSL_DGI156 --keypoints_model wholepose > ../logs_spoter/log_PUCP_71_00005-wholepose.txt & 
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=1 nohup python train.py --wandb --epoch 20000 --learning_rate 0.00005 --keypoints_number 71 --num_coder_layers 1 --dim_feedforward 32 --device 0 --seed 1 --dataset PUCP_PSL_DGI156 --keypoints_model openpose  > ../logs_spoter/log_PUCP_71_00005-openpose.txt & 

###MULTI_GPU=1 CUDA_VISIBLE_DEVICES=0 python train.py --wandb --epoch 20 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 2 --dim_feedforward 32 --device 0 --seed 1 --dataset PUCP_PSL_DGI156 --keypoints_model mediapipe > ../logs_spoter/prueba_log_PUCP_29_00005-mediapipe.txt & 
'''

#WLASL-29
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=0 nohup python train.py --wandb --epoch 5000 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 12 --dim_feedforward 32 --device 0 --seed 1 --dataset WLASL --keypoints_model mediapipe > ../logs_spoter/log_WLASL_29_00005-mediapipe.txt & 
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=0 nohup python train.py --wandb --epoch 5000 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 12 --dim_feedforward 32 --device 0 --seed 1 --dataset WLASL --keypoints_model wholepose > ../logs_spoter/log_WLASL_29_00005-wholepose.txt & 
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=0 nohup python train.py --wandb --epoch 5000 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 12 --dim_feedforward 32 --device 0 --seed 1 --dataset WLASL --keypoints_model openpose  > ../logs_spoter/log_WLASL_29_00005-openpose.txt & 

#WLASL-71
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=1 nohup python train.py --wandb --epoch 5000 --learning_rate 0.00005 --keypoints_number 71 --num_coder_layers 6 --dim_feedforward 32 --device 0 --seed 1 --dataset WLASL --keypoints_model mediapipe > ../logs_spoter/log_WLASL_71_00005-mediapipe.txt & 
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=1 nohup python train.py --wandb --epoch 5000 --learning_rate 0.00005 --keypoints_number 71 --num_coder_layers 6 --dim_feedforward 32 --device 0 --seed 1 --dataset WLASL --keypoints_model wholepose > ../logs_spoter/log_WLASL_71_00005-wholepose.txt & 
MULTI_GPU=1 CUDA_VISIBLE_DEVICES=1 nohup python train.py --wandb --epoch 5000 --learning_rate 0.00005 --keypoints_number 71 --num_coder_layers 6 --dim_feedforward 32 --device 0 --seed 1 --dataset WLASL --keypoints_model openpose  > ../logs_spoter/log_WLASL_71_00005-openpose.txt & 
