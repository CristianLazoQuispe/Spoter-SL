
import os
import json
import argparse
import random
import logging
import torch

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from torchvision import transforms
from pathlib import Path

from Src.datasets.utils_split import __balance_val_split, __split_of_train_sequence, __log_class_statistics

from Src.datasets.SpoterDataset import SpoterDataset
from Src.datasets.SpoterDataLoader import SpoterDataLoader

from Src.datasets.drawing import drawing

from Src.spoter.spoter_model import SPOTER
from Src.spoter.spoter_model1 import SPOTER1
from Src.spoter.spoter_model2 import SPOTER2
from Src.spoter.spoter_model3 import SPOTER3
from Src.spoter.spoter_model4 import SPOTER4
from Src.spoter.utils import train_epoch, evaluate, generate_csv_result, generate_csv_accuracy
from Src.spoter.gaussian_noise import GaussianNoise
from Src.spoter.gpu import configurar_cuda_visible
from Src.spoter.optimizer import dynamic_weight_decay
import wandb
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
import time

from Src.spoter import save_artifact
import atexit

import gc
import argparse
import hashlib
import random
import string

run = None
model_save_folder_path = None
top_val_f1_weighted = 0
top_val_f1_weighted_before = 0

def finish_process():
    global run,model_save_folder_path
    global top_val_f1_weighted,top_val_f1_weighted_before
    """
    function to finish wandb if there is an error in the code or force stop
    """
    print("Finishing process")
    if top_val_f1_weighted!= top_val_f1_weighted_before:
        print("Sending artifact to wandb!")
        artifact = wandb.Artifact(f'best-model_{run.id}.pth', type='model')
        artifact.add_file(model_save_folder_path + "/checkpoint_best_model.pth")
        run.log_artifact(artifact)
    print("Closing wandb.. ")
    wandb.finish()
    print("Wandb closed")
    print("Cleaning memory.. ")
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleaned")

# Modifica el collate_fn para rellenar secuencias
def custom_collate_fn(batch):
    data, labels, video_names = zip(*batch)
    data = pad_sequence(data, batch_first=True)
    labels = torch.cat(labels)
    return data, labels, video_names




from dotenv import load_dotenv
import os
load_dotenv()

os.environ["WANDB_API_KEY"] =  os.getenv("WANDB_API_KEY")
PROJECT_WANDB =  os.getenv("PROJECT_WANDB")
ENTITY        =  os.getenv("ENTITY")

def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="305-aec-sw",
                        help="Name of the experiment after which the logs and plots will be named")

    parser.add_argument("--models_random_name", type=str, default="",
                        help="Name of the experiment after which the logs and plots will be named")

    parser.add_argument("--num_classes", type=int, default=38, help="Number of classes to be recognized by the model")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")

    # Data
    parser.add_argument("--training_set_path", type=str, default="../ConnectingPoints/split/DGI305-AEC--38--incremental--mediapipe-Train.hdf5", help="Path to the training dataset CSV file")
    parser.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset CSV file")
    parser.add_argument("--experimental_train_split", type=float, default=None,
                        help="Determines how big a portion of the training set should be employed (intended for the "
                             "gradually enlarging training set experiment from the paper)")

    parser.add_argument("--validation_set", type=str, choices=["from-file", "split-from-train", "none"],
                        default="from-file", help="Type of validation set construction. See README for further rederence")
    parser.add_argument("--validation_set_size", type=float,
                        help="Proportion of the training set to be split as validation set, if 'validation_size' is set"
                             " to 'split-from-train'")
    parser.add_argument("--validation_set_path", type=str, default="../ConnectingPoints/split/DGI305-AEC--38--incremental--mediapipe-Val.hdf5", help="Path to the validation dataset CSV file")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=20000, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the model training")
    parser.add_argument("--log_freq", type=int, default=1,
                        help="Log frequency (frequency of printing all the training info)")

    ## trainable parameters
    #num_classes, num_rows=64,hidden_dim=108, num_heads=9, num_layers_1=6, num_layers_2=6, dim_feedforward=256)
    
    parser.add_argument("--sweep", type=int, default=0, help="")
    parser.add_argument("--num_rows", type=int, default=64, help="")
    parser.add_argument("--use_spoter2", type=int, default=0, help="")
    parser.add_argument("--norm_first", type=int, default=0, help="")
    parser.add_argument("--not_requires_grad_n_layers", type=int, default=1, help="")
    

    parser.add_argument("--hidden_dim", type=int, default=108, help="")
    parser.add_argument("--num_heads", type=int, default=9, help="")
    parser.add_argument("--num_layers_1", type=int, default=6, help="")
    parser.add_argument("--num_layers_2", type=int, default=6, help="")
    parser.add_argument("--dim_feedforward_encoder", type=int, default=64, help="")
    parser.add_argument("--dim_feedforward_decoder", type=int, default=256, help="")

    parser.add_argument("--early_stopping_patience", type=int, default=200, help="")
    parser.add_argument("--max_acc_difference", type=float, default=0.35, help="")

    # Checkpointing
    parser.add_argument("--save_checkpoints", type=bool, default=True,
                        help="Determines whether to save weights checkpoints")

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="", help="Factor for the steplr plateu scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=50,help="Patience for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_factor", type=float, default=0.99, help="Factor for the ReduceLROnPlateau scheduler")

    parser.add_argument("--weight_decay_dynamic", type=int, default=0,help="Patience for the ReduceLROnPlateau scheduler")
    parser.add_argument("--weight_decay_patience", type=int, default=1,help="Patience for the ReduceLROnPlateau scheduler")
    parser.add_argument("--weight_decay_max", type=float, default=0.05,help="Patience for the ReduceLROnPlateau scheduler")
    parser.add_argument("--weight_decay_min", type=float, default=0.000005,help="Patience for the ReduceLROnPlateau scheduler")
    parser.add_argument("--weight_decay_kp", type=float, default=0.0001,help="Patience for the ReduceLROnPlateau scheduler")
    parser.add_argument("--weight_decay_ki", type=float, default=0.0,help="Patience for the ReduceLROnPlateau scheduler")
    parser.add_argument("--weight_decay_kd", type=float, default=0.0,help="Patience for the ReduceLROnPlateau scheduler")
    parser.add_argument("--weight_decay_setpoint", type=float, default=0.5,help="Patience for the ReduceLROnPlateau scheduler")

    # Gaussian noise normalization
    parser.add_argument("--gaussian_mean", type=float, default=0.0, help="Mean parameter for Gaussian noise layer")
    parser.add_argument("--gaussian_std", type=float, default=0.001,
                        help="Standard deviation parameter for Gaussian noise layer")

    # Visualization
    parser.add_argument("--plot_stats", type=bool, default=True,
                        help="Determines whether continuous statistics should be plotted at the end")
    parser.add_argument("--plot_lr", type=bool, default=True,
                        help="Determines whether the LR should be plotted at the end")

    parser.add_argument("--device", type=str, default='0',
                    help="Determines which Nvidia device will use (just one number)")
    # To continue training the data
    parser.add_argument("--resume", type=int, default=1,help="path to retrieve the model for continue training")
    parser.add_argument("--transfer_learning", type=str, default="",help="path to retrieve the model for transfer learning")
    parser.add_argument("--augmentation", type=int, default=0,
                        help="Augmentations")
    parser.add_argument("--factor_aug", type=int, default=2,
                        help="factor para multiplicar los datos de augmentation")
    parser.add_argument("--batch_name", type=str, default="",
                        help=" | mean_1:calcula backward en cada batch | mean_2: calcula backward en cada instancia")    
    parser.add_argument("--batch_size", type=int, default=64,help="batch_size ")
    parser.add_argument("--num_workers", type=int, default=8,help="num_workers ")
                        
    parser.add_argument("--loss_weighted_factor", type=int, default=1,
                        help="Loss crossentropy weighted ")
    parser.add_argument("--label_smoothing", type=float, default=0,
                        help="Loss crossentropy weighted ")
    parser.add_argument("--optimizer", type=str, default='sgd',
                        help="Loss crossentropy weighted ")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="Loss crossentropy weighted ")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout weighted ")                                                
    parser.add_argument("--data_fold", type=int, default=5,help="")
    parser.add_argument("--data_seed", type=int, default=42,help="")
                                                
    parser.add_argument("--use_wandb", type=int, default=1,help="")
                           


    return parser

# TO MODIFY THE LEARNING RATE
def lr_lambda(current_step, optim):

    #lr_rate = 0.0003
    lr_rate = 0.00005
    '''
    if current_step <= 30:
        lr_rate = current_step/30000  # Función lineal
    else:
        lr_rate = (0.00003/current_step) ** 0.5  # Función de raíz cuadrada inversa
    '''

    print(f'[{current_step}], Lr_rate: {lr_rate}')
    optim.param_groups[0]['lr'] = lr_rate

    return optim

def get_df_stats(data_set,stats,num_classes):
    stats = {data_set.inv_dict_labels_dataset[k]:v for k,v in stats.items() if k < num_classes}
    df_stats = pd.DataFrame(stats.items(), columns=['gloss', 'n_success_n_total'])
    df_stats[['n_success', 'n_total']] = pd.DataFrame(df_stats['n_success_n_total'].tolist(), index=df_stats.index)
    df_stats.drop(columns=['n_success_n_total'], inplace=True)
    df_stats['gloss_acc'] = df_stats['n_success'] / df_stats['n_total']
    df_stats = df_stats.sort_values(by='gloss')
    return df_stats

import random
import string

def generate_string(longitud):
    caracteres = string.ascii_letters + string.digits
    cadena = ''.join(random.choice(caracteres) for i in range(longitud))
    
    return cadena

def train(args):
    global run,model_save_folder_path
    global top_val_f1_weighted,top_val_f1_weighted_before


    if args.validation_set == "from-file":
        if args.validation_set_path == "":
            args.validation_set_path = args.training_set_path.replace("Train","Val")

    if args.models_random_name == '':
        #args.models_random_name = generate_string(10)

        key_parts = []
        for k, v in vars(args).items():
            key_parts.append(f"{k}_{v}")

        key = "".join(key_parts) 
        # Crear hash final
        hash_object = hashlib.md5(key.encode())
        models_random_name = hash_object.hexdigest()[:8]

        print(models_random_name) 
        args.models_random_name =  models_random_name

    args.models_random_name += "-"+str(args.hidden_dim)+"-"+str(args.num_heads)+"-"+ str(args.num_layers_1)+"-"
    args.models_random_name += str(args.num_layers_2)+"-"+str(args.dim_feedforward_encoder)+"-"+str(args.dim_feedforward_decoder)

    # Initialize all the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Set the output format to print into the console and save into LOG file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("Results/logs/"+args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + ".log")
        ]
    )

    #args.experiment_name = "_".join([args.experiment_name.split('--')[0], 
    #                                 f"lr-{args.lr}",
    #                                 f"Nclass-{args.num_classes}"]) 



    # Set device to CUDA only if applicable
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    
    drawer = drawing(w = 256,h = 256,path_points= 'Data/points_54.csv')

    # DATA LOADER
        # Training set
    transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
    

    # Ensure that the path for checkpointing and for images both exist
    Path("Results/").mkdir(parents=True, exist_ok=True)
    
    

    Path("Results/checkpoints/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    Path("Results/checkpoints/" + args.experiment_name + "/"+args.training_set_path.split("/")[-1].split(".")[0]+"/").mkdir(parents=True, exist_ok=True)
    Path("Results/checkpoints/" + args.experiment_name + "/"+args.training_set_path.split("/")[-1].split(".")[0]+"/"+args.models_random_name+"/").mkdir(parents=True, exist_ok=True)
    Path("Results/images/metrics/").mkdir(parents=True, exist_ok=True)
    Path("Results/images/histograms/").mkdir(parents=True, exist_ok=True)
    Path("Results/images/keypoints/").mkdir(parents=True, exist_ok=True)

    epoch_start = 0

    checkpoint = None
    # RETRIEVE TRAINING
    if args.resume:
        model_save_folder_path = os.path.join("Results/checkpoints/" + args.experiment_name,
        args.training_set_path.split("/")[-1].split(".")[0],
        args.models_random_name)

        path_model = model_save_folder_path+'/checkpoint_model.pth'
        print("path_model:",path_model)
        if args.use_spoter2 ==1:
            print("USING SPOTER Version 2 + encoder + decoder con multiatten and atten")
            slrt_model = SPOTER1(num_classes=args.num_classes, num_rows=args.num_rows,
                                hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                                num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                                dim_feedforward_encoder=args.dim_feedforward_encoder,
                                dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout,
                                norm_first = bool(args.norm_first),
                                not_requires_grad_n_layers = bool(args.not_requires_grad_n_layers))

        elif args.use_spoter2 ==2:
            print("USING SPOTER Version 3 + solo encoder")
            slrt_model = SPOTER2(num_classes=args.num_classes, num_rows=args.num_rows,
                                hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                                num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                                dim_feedforward_encoder=args.dim_feedforward_encoder,
                                dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout,
                                norm_first = bool(args.norm_first),
                                not_requires_grad_n_layers = bool(args.not_requires_grad_n_layers))

        elif args.use_spoter2 ==3:
            print("USING SPOTER Version 4 + solo encoder + mlp")
            slrt_model = SPOTER2(num_classes=args.num_classes, num_rows=args.num_rows,
                                hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                                num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                                dim_feedforward_encoder=args.dim_feedforward_encoder,
                                dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout,
                                norm_first = bool(args.norm_first),
                                not_requires_grad_n_layers = bool(args.not_requires_grad_n_layers),
                                has_mlp=True)

        elif args.use_spoter2 ==4:
            print("USING SPOTER Version 5 + solo encoder ResiDual + mlp")
            slrt_model = SPOTER3(num_classes=args.num_classes, num_rows=args.num_rows,
                                hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                                num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                                dim_feedforward_encoder=args.dim_feedforward_encoder,
                                dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout,
                                norm_first = bool(args.norm_first),
                                not_requires_grad_n_layers = bool(args.not_requires_grad_n_layers),
                                has_mlp=True)

        elif args.use_spoter2 ==5:
            print("USING SPOTER Version 6 +  encoder ResiDual + decoder ResiDual")
            slrt_model = SPOTER4(num_classes=args.num_classes, num_rows=args.num_rows,
                                hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                                num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                                dim_feedforward_encoder=args.dim_feedforward_encoder,
                                dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout,
                                norm_first = bool(args.norm_first),
                                not_requires_grad_n_layers = bool(args.not_requires_grad_n_layers),
                                )
        else:
            slrt_model = SPOTER(num_classes=args.num_classes, num_rows=args.num_rows,
                                hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                                num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                                dim_feedforward_encoder=args.dim_feedforward_encoder,
                                dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout)

        if os.path.exists(path_model):
            print("RESUME MODEL : Load weights")
            checkpoint = torch.load(path_model)
            if args.use_wandb:
                print("wandb id:",checkpoint["wandb"])
            #print("lr      :",checkpoint["lr"])
            print("epoch   :",checkpoint["epoch"])
            slrt_model.load_state_dict(checkpoint['model_state_dict'])
            slrt_model.to("cuda")
            args.lr = 0.000099#checkpoint['lr']

        if args.optimizer == 'adam':
            sgd_optimizer = optim.Adam(slrt_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            sgd_optimizer = optim.SGD(slrt_model.parameters(), lr=args.lr)

        if os.path.exists(path_model):
            print("RESUME MODEL : Load optimizer")
            sgd_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch']

    # TRANSFER LEARNING
    elif args.transfer_learning:
        slrt_model = SPOTER(num_classes=100, num_rows=args.num_rows,
                            hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                            num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                            dim_feedforward=args.dim_feedforward)
        checkpoint = torch.load(args.transfer_learning)
        slrt_model.load_state_dict(checkpoint['model_state_dict'])

        # freeze all model layer
        for param in slrt_model.parameters():
            param.requires_grad = False

        slrt_model.linear_class = nn.Linear(slrt_model.linear_class.in_features, args.num_classes)

        # unfreeze last model layer
        for param in slrt_model.linear_class.parameters():
            param.requires_grad = True

        sgd_optimizer = optim.SGD(slrt_model.linear_class.parameters(), lr=args.lr)

    # Normal scenario
    else:


        if args.use_spoter2:
            print("USING SPOTER Version 2")
            slrt_model = SPOTER2(num_classes=args.num_classes, num_rows=args.num_rows,
                                hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                                num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                                dim_feedforward_encoder=args.dim_feedforward_encoder,
                                dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout,
                                norm_first = bool(args.norm_first),
                                not_requires_grad_n_layers = bool(args.not_requires_grad_n_layers))

        else:
            slrt_model = SPOTER(num_classes=args.num_classes, num_rows=args.num_rows,
                                hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                                num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                                dim_feedforward_encoder=args.dim_feedforward_encoder,
                                dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout)

        if args.optimizer == 'adam':
            sgd_optimizer = optim.Adam(slrt_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            #sgd_optimizer = optim.AdamW(slrt_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        else:
            sgd_optimizer = optim.SGD(slrt_model.parameters(), lr=args.lr)

    # Construct the model
    

    if args.use_wandb:
        if (args.resume and checkpoint is not None):
            print("RESUME MUST")
            print("RESUME MUST")
            print("RESUME MUST")
        # MARK: TRAINING PREPARATION AND MODULES
        run = wandb.init(project=PROJECT_WANDB, 
                        entity=ENTITY,
                        config=args, 
                        name=args.experiment_name, 
                        job_type="model-training",
                        save_code=True,
                        #settings=wandb.Settings(start_method="fork"),
                        resume="must" if (args.resume and checkpoint is not None)  else None,
                        id=checkpoint["wandb"] if (args.resume and checkpoint is not None)  else None,
                        tags=["paper"])

    if args.augmentation:
        train_set = SpoterDataset(args.training_set_path, transform=transform, has_augmentation=True,keypoints_model='mediapipe',factor=args.factor_aug)
    else:
        train_set = SpoterDataset(args.training_set_path, transform=transform, has_augmentation=False,keypoints_model='mediapipe',factor=args.factor_aug)

    # Validation set
    if args.validation_set == "from-file":
        print("NO AUGMENTATION UN VALIDATION")
        val_set    = SpoterDataset(args.validation_set_path, has_augmentation=False,keypoints_model='mediapipe')
        val_loader = SpoterDataLoader(val_set, shuffle=True, generator=g, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    elif args.validation_set == "split-from-train":
        train_set, val_set = __balance_val_split(train_set, 0.2)

        val_set.transform = None
        val_set.augmentations = False
        val_loader = SpoterDataLoader(val_set, shuffle=True, generator=g, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    else:
        val_loader = None

    # Testing set
    if args.testing_set_path:
        eval_set = SpoterDataset(args.testing_set_path, has_augmentation=False,keypoints_model='mediapipe')
        eval_loader = SpoterDataLoader(eval_set, shuffle=True, generator=g, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    else:
        eval_loader = None

    # Final training set refinements
    if args.experimental_train_split:
        train_set = __split_of_train_sequence(train_set, args.experimental_train_split)


    train_loader = SpoterDataLoader(train_set, shuffle=True, generator=g, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    

    # Construct the other modules
    
    lr_scheduler = None

    if args.scheduler == 'steplr':
        lr_scheduler = optim.lr_scheduler.StepLR(sgd_optimizer, step_size=1, gamma=0.9995)
    if args.scheduler == 'plateu':
        #args.
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, mode='max', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=False,threshold=0.0001, threshold_mode='rel',cooldown=0, min_lr=0.00001, eps=1e-08)
        #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, mode='max', factor=0.1, patience=5, verbose=True)

        if args.weight_decay_dynamic:
            wd_scheduler = dynamic_weight_decay(weight_decay_patience = args.weight_decay_patience,
                weight_decay_max = args.weight_decay_max,
                weight_decay_min = args.weight_decay_min,
                kp = args.weight_decay_kp,
                ki = args.weight_decay_ki,
                kd = args.weight_decay_kd,
                setpoint = args.weight_decay_setpoint)

            



    # MARK: DATA
    #artifact_name = config[args.experiment_name]
    #print("artifact_name : ", artifact_name)
    #model_artifact = wandb.Artifact(artifact_name, type='model')

    print("#"*50)
    print("#"*30)
    print("#"*10)
    total_params = sum(p.numel() for p in slrt_model.parameters())
    trainable_params = sum(p.numel() for p in slrt_model.parameters() if p.requires_grad)
    ratio = trainable_params / total_params
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Trainable Parameters Ratio: {ratio:.4f}")
    print("#"*10)
    print("#"*30)
    print("#"*50)


    if args.use_wandb:
        # Log the parameters to wandb
        wandb.config.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_parameters_ratio": ratio
        })

        config = wandb.config
        #wandb.watch_called = False

    
    # MARK: TRAINING
    train_acc, val_acc = 0, 0
    losses, train_accs, val_accs, val_accs_top5 = [], [], [], []
    lr_progress = []
    top_train_acc, top_val_acc = 0, 0
    top_val_f1_weighted = 0
    top_val_f1_weighted_before = 0
    checkpoint_index = 0

    if args.experimental_train_split:
        print("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")
        logging.info("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")

    else:
        print("Starting " + args.experiment_name + "...\n\n")
        logging.info("Starting " + args.experiment_name + "...\n\n")

    slrt_model.train(True)
    slrt_model.to(device)


    best_val_max_acc = -1#float('inf')
    epochs_distance_without_improvement = 0
    epochs_without_improvement = 0


    print("*"*50)
    print("args.augmentation:",args.augmentation)
    if args.augmentation:
        print("AUGMENTATION IS USED")

    
    # LABEL SMOOTHING IN CRITERION
    if args.loss_weighted_factor!=0:
        # CLASS WEIGHT
        print("train_set.factors",train_set.factors)
        factors = [train_set.factors[i]**args.loss_weighted_factor for i in range(args.num_classes)]
        min_factor = min(factors) #sum es indiferente porque lo normaliza 
        factors = [value/min_factor for value in factors]

        name_factors = {train_set.inv_dict_labels_dataset[i]:value for i, value in enumerate(factors)}
        print("name_factors:")
        print(json.dumps(name_factors, indent=4))
        class_weight = torch.FloatTensor(factors).to(device)
        print("\\\\\\"*20)
        print("class_weight:",class_weight)
        cel_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=class_weight)
    else:
        cel_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)#, weight=class_weight)
    #cel_criterion = nn.CrossEntropyLoss()
    
    previous_val_loss = 0
    previous_val_acc  = 0 

    print("training_set_path   :",args.training_set_path)
    print("validation_set_path :",args.validation_set_path)

    amp = True
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    step = None
    if args.use_wandb:
        step = run.summary.get("_step")
    
    step = 0 if step is None else step+1

    for epoch in range(epoch_start, args.epochs):

        #sgd_optimizer = lr_lambda(epoch, sgd_optimizer)
        start_time = time.time()

        current_lr = sgd_optimizer.param_groups[0]["lr"]
        current_weight_decay = sgd_optimizer.param_groups[0]['weight_decay']
        train_loss,train_stats,train_labels_original,train_labels_predicted,list_depth_map_train,list_label_name_train = train_epoch(slrt_model, train_loader, 
        cel_criterion, sgd_optimizer,device,epoch=epoch,args=args,grad_scaler=grad_scaler)
        
        train_acc   = f1_score(train_labels_original, train_labels_predicted, average='micro',zero_division=0)

        losses.append(train_loss)
        train_accs.append(train_acc)
        # Obtener la tasa de aprendizaje actual


        if val_loader:
            slrt_model.train(False)
            val_loss, val_acc_top5, val_stats,val_labels_original,val_labels_predicted,list_depth_map_val,list_label_name_val = evaluate(slrt_model, val_loader, cel_criterion, device,epoch=epoch,args=args)
            slrt_model.train(True)

            val_acc           = f1_score(val_labels_original, val_labels_predicted, average='micro',zero_division=0)
            train_f1_weighted = f1_score(train_labels_original, train_labels_predicted, average='weighted',zero_division=0)
            val_f1_weighted   = f1_score(val_labels_original, val_labels_predicted, average='weighted',zero_division=0)


            val_accs.append(val_acc)
            val_accs_top5.append(val_acc_top5)

            # GUARDANDO EL MEJOR MODELO
            if val_f1_weighted > best_val_max_acc:
                best_val_max_acc = val_f1_weighted
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            # Check for early stopping based on the difference between training and validation loss
            acc_difference = np.abs(train_f1_weighted - val_f1_weighted)

            if acc_difference > args.max_acc_difference:
                epochs_distance_without_improvement+=1
            else:
                epochs_distance_without_improvement=0
                
            if epochs_distance_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping! Training and validation acc difference exceeded threshold.")
                break  # Exit the training loop

            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping! No improvement for {args.early_stopping_patience} consecutive epochs.")
                break  # Exit the training loop

            
        df_train_stats = get_df_stats(train_set,train_stats,args.num_classes).add_prefix('train_')
        df_val_stats   = get_df_stats(val_set,val_stats,args.num_classes).add_prefix('val_')

        df_merged = pd.merge(df_train_stats, df_val_stats, how='inner', left_on='train_gloss', right_on='val_gloss')
        # Puedes eliminar la columna redundante después de la fusión
        df_merged.drop('val_gloss', axis=1, inplace=True)
        # Renombra las columnas para que solo quede el nombre "gloss"
        df_merged.rename(columns={'train_gloss': 'gloss'}, inplace=True)

        

        total_time = time.time()-start_time


                
        if val_loss is not None:
            previous_val_loss = val_loss
        if val_acc is not None:
            previous_val_acc = val_acc
        if val_loader:
            current_weight_decay = sgd_optimizer.param_groups[0]['weight_decay']
            log_values = {
                'current_lr':current_lr,
                'current_weight_decay':current_weight_decay,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': previous_val_acc,
                'val_loss':previous_val_loss,
                'val_best_acc': top_val_acc,
                'val_top5_acc': val_acc_top5,
                'epoch': epoch,
                'train_f1_weighted':train_f1_weighted,
                'val_f1_weighted':val_f1_weighted,
                'total_time':total_time
            }

        if args.scheduler == 'steplr':
            lr_scheduler.step()

        if args.scheduler == 'plateu' and val_acc>0:
            if val_loss is not None:
                lr_scheduler.step(val_acc)
                # Actualizar weight decay
                if args.weight_decay_dynamic:
                    weight_decay = sgd_optimizer.param_groups[0]['weight_decay']
                    weight_decay = wd_scheduler.step(train_loss, val_loss, weight_decay)
                    sgd_optimizer.param_groups[0]['weight_decay'] = weight_decay


        """
        if val_loader:
            for _, row in df_train_stats.iterrows():
                gloss_name = row['train_gloss']
                accuracy_metric_name = f'train_acc_{gloss_name}'
                log_values[accuracy_metric_name] = row['train_gloss_acc']
            for _, row in df_val_stats.iterrows():
                gloss_name = row['val_gloss']
                accuracy_metric_name = f'val_acc_{gloss_name}'
                log_values[accuracy_metric_name] = row['val_gloss_acc']
        """


            
        # Save checkpoints if they are best in the current subset
        if args.save_checkpoints:
            
            #model_save_folder_path = os.path.join("Results/checkpoints/" + args.experiment_name,args.models_random_name,args.training_set_path.split("/")[-1])
            model_save_folder_path = os.path.join("Results/checkpoints/" + args.experiment_name,
            args.training_set_path.split("/")[-1].split(".")[0],
            args.models_random_name)
            

            if args.use_wandb:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': slrt_model.state_dict(),
                    'optimizer_state_dict': sgd_optimizer.state_dict(),
                    'loss': train_loss,
                    'current_lr':current_lr,
                    "lr_scheduler": lr_scheduler.state_dict(),

                    'current_weight_decay':current_weight_decay,

                    "wandb": save_artifact.WandBID(wandb.run.id).state_dict(),
                    "epoch": save_artifact.Epoch(epoch).state_dict(),
                    "metric_val_acc": save_artifact.Metric(previous_val_acc).state_dict()
                }, model_save_folder_path + "/checkpoint_model.pth")
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': slrt_model.state_dict(),
                    'optimizer_state_dict': sgd_optimizer.state_dict(),
                    'loss': train_loss,

                    'current_lr':current_lr,
                    "lr_scheduler": lr_scheduler.state_dict(),

                    'current_weight_decay':current_weight_decay,

                }, model_save_folder_path + "/checkpoint_model.pth")

            if val_acc > top_val_acc:
                top_val_acc = val_acc

            if val_f1_weighted > top_val_f1_weighted:
                top_val_f1_weighted = val_f1_weighted

                print("Saving best model!")
                print(model_save_folder_path)

                if args.use_wandb:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': slrt_model.state_dict(),
                        'optimizer_state_dict': sgd_optimizer.state_dict(),
                        'loss': train_loss,

                        'current_lr':current_lr,
                        "lr_scheduler": lr_scheduler.state_dict(),

                        'current_weight_decay':current_weight_decay,

                        "wandb": save_artifact.WandBID(wandb.run.id).state_dict(),
                        "epoch": save_artifact.Epoch(epoch).state_dict(),
                        "metric_val_acc": save_artifact.Metric(top_val_acc).state_dict()

                    }, model_save_folder_path + "/checkpoint_best_model.pth")
                else:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': slrt_model.state_dict(),
                        'optimizer_state_dict': sgd_optimizer.state_dict(),
                        'loss': train_loss,

                        'current_lr':current_lr,
                        "lr_scheduler": lr_scheduler.state_dict(),

                        'current_weight_decay':current_weight_decay,

                    }, model_save_folder_path + "/checkpoint_best_model.pth")
                
                checkpoint_index += 1

        if epoch%1000 == 0:
            list_images_train,filename_train = drawer.get_video_frames_25_glosses_batch(list_depth_map_train,list_label_name_train,suffix='train',save_gif=True)
            print("sending gif")
            #wandb.log({"train_video": wandb.Video(filename_train, fps=1, format="mp4")})

            list_images_val,filename_val   = drawer.get_video_frames_25_glosses_batch(list_depth_map_val,list_label_name_val,suffix='val',save_gif=True)
            print("sending gif")
            if args.use_wandb:
                if val_loader:
                    wandb.log({"gloss_train_video": wandb.Video(filename_train, format="gif")})
                    wandb.log({"gloss_val_video": wandb.Video(filename_val, format="gif")})
                    #wandb.log({"val_video": wandb.Video(filename_val, fps=1,format="mp4")})

        if epoch%100 == 0:

            if args.use_wandb:
                if val_loader:
                    #log_values['Train_table_stats']   =  wandb.Table(dataframe=df_train_stats)
                    #log_values['Val_table_stats']     =  wandb.Table(dataframe=df_val_stats)
                    log_values['Compare_table_stats'] =  wandb.Table(dataframe=df_merged)


            if top_val_f1_weighted!= top_val_f1_weighted_before:
                top_val_f1_weighted_before = top_val_f1_weighted
                if args.use_wandb:
                    print("Sending artifact best model to wandb!")
                    artifact = wandb.Artifact(f'best-model_{run.id}.pth', type='model')
                    artifact.add_file(model_save_folder_path + "/checkpoint_best_model.pth")
                    run.log_artifact(artifact)

        if val_loader:
            if args.use_wandb:
                wandb.log(log_values, step=step)
                step+=1

        lr_progress.append(sgd_optimizer.param_groups[0]["lr"])

    # MARK: TESTING

    print("\nTesting checkpointed models starting...\n")
    logging.info("\nTesting checkpointed models starting...\n")

    top_result, top_result_name = 0, ""

    # PLOT 0: Performance (loss, accuracies) chart plotting
    if args.plot_stats:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(losses) + 1), losses, c="#D64436", label="Training loss")
        ax.plot(range(1, len(train_accs) + 1), train_accs, c="#00B09B", label="Training accuracy")

        if val_loader:
            ax.plot(range(1, len(val_accs) + 1), val_accs, c="#E0A938", label="Validation accuracy")
            ax.plot(range(1, len(val_accs_top5) + 1), val_accs_top5, c="#F2B09A", label="val_Top5_acc")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
        ax.grid()
        fig.savefig("Results/images/metrics/" + args.experiment_name + "_loss.png")

    # PLOT 1: Learning rate progress
    if args.plot_lr:
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, len(lr_progress) + 1), lr_progress, label="LR")
        ax1.set(xlabel="Epoch", ylabel="LR", title="")
        ax1.grid()

        fig1.savefig("Results/images/metrics/" + args.experiment_name + "_lr.png")

    print("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    logging.info("\nAny desired statistics have been plotted.\nThe experiment is finished.")


if __name__ == '__main__':
    
    atexit.register(finish_process)
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    
    
    
    @configurar_cuda_visible([int(value) for value in args.device.split(",")])
    def main_process(): 
        train(args)
    main_process()