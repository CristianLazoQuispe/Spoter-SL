'''

############################
python train.py --experiment_name cris_openpose_AEC_lr_0.00025 --training_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 50 --keypoints_model openpose --lr 0.00025


python train.py --experiment_name cris_openpose_AEC_lr_0.0005_exp --training_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 50 --keypoints_model openpose --lr 0.0005


python train.py --experiment_name cris_openpose_AEC_lr_0.001 --training_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 50 --keypoints_model openpose --lr 0.001


python train.py --experiment_name cris_openpose_AEC_lr_0.002 --training_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 50 --keypoints_model openpose --lr 0.002

python train.py --experiment_name cris_openpose_AEC_lr_0.003 --training_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 50 --keypoints_model openpose --lr 0.003


######################################

python train.py --experiment_name cris_openpose_AEC_lr_0.0005 --training_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 200 --keypoints_model openpose --lr 0.0005

python train.py --experiment_name cris_mediapipe_AEC_lr_0.0005 --training_set_path ../../joe/ConnectingPoints/split/AEC--mediapipe-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/AEC--mediapipe-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/AEC--mediapipe-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 200 --keypoints_model mediapipe --lr 0.0005


python train.py --experiment_name cris_wholepose_AEC_lr_0.0005 --training_set_path ../../joe/ConnectingPoints/split/AEC--wholepose-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/AEC--wholepose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/AEC--wholepose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 200 --keypoints_model wholepose --lr 0.0005
'''

#############################
python train.py --experiment_name tunning/PUCP/cris_mediapipe_PUCP_lr_0.00025 --training_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5 --hidden_dim 142 --num_classes 36 --epochs 30 --keypoints_model mediapipe --lr 0.00025

python train.py --experiment_name tunning/PUCP/cris_mediapipe_PUCP_lr_0.0005 --training_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5 --hidden_dim 142 --num_classes 36 --epochs 30 --keypoints_model mediapipe --lr 0.0005

python train.py --experiment_name tunning/PUCP/cris_mediapipe_PUCP_lr_0.001 --training_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5 --hidden_dim 142 --num_classes 36 --epochs 30 --keypoints_model mediapipe --lr 0.001

python train.py --experiment_name tunning/PUCP/cris_mediapipe_PUCP_lr_0.002 --training_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5 --hidden_dim 142 --num_classes 36 --epochs 30 --keypoints_model mediapipe --lr 0.002


#python train.py --experiment_name cris_mediapipe_PUCP_lr_0.003 --training_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5 --hidden_dim 142 --num_classes 36 --epochs 30 --keypoints_model mediapipe --lr 0.003

##############################
python train.py --experiment_name tunning/WLASL/cris_wholepose_WLASL_lr_0.00025 --training_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --hidden_dim 142 --num_classes 101 --epochs 30 --keypoints_model wholepose --lr 0.00025

python train.py --experiment_name tunning/WLASL/cris_wholepose_WLASL_lr_0.0005 --training_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --hidden_dim 142 --num_classes 101 --epochs 30 --keypoints_model wholepose --lr 0.0005

python train.py --experiment_name tunning/WLASL/cris_wholepose_WLASL_lr_0.001 --training_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --hidden_dim 142 --num_classes 101 --epochs 30 --keypoints_model wholepose --lr 0.001

python train.py --experiment_name tunning/WLASL/cris_wholepose_WLASL_lr_0.002 --training_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --hidden_dim 142 --num_classes 101 --epochs 30 --keypoints_model wholepose --lr 0.002

#python train.py --experiment_name cris_wholepose_WLASL_lr_0.003 --training_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --hidden_dim 142 --num_classes 101 --epochs 30 --keypoints_model wholepose --lr 0.003

