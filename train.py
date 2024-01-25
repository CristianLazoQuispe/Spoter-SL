
import os
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
from torch.utils.data import DataLoader
from pathlib import Path

from Src.datasets.utils_split import __balance_val_split, __split_of_train_sequence, __log_class_statistics
from Src.datasets.Spoter_dataloader import LSP_Dataset
from Src.datasets.Spoter_dataloader_aug import AugmentedDataLoader
from Src.spoter.spoter_model import SPOTER
from Src.spoter.utils import train_epoch, evaluate, generate_csv_result, generate_csv_accuracy
from Src.spoter.gaussian_noise import GaussianNoise
from Src.spoter.gpu import configurar_cuda_visible
import wandb
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
import time

# Modifica el collate_fn para rellenar secuencias
def custom_collate_fn(batch):
    data, labels, video_names = zip(*batch)
    data = pad_sequence(data, batch_first=True)
    labels = torch.cat(labels)
    return data, labels, video_names



PROJECT_WANDB = "SLR_2023"#Spoter-as-orignal"
ENTITY = "ml_projects" #c-vasquezr

from dotenv import load_dotenv
import os
load_dotenv()

os.environ["WANDB_API_KEY"] =  os.getenv("WANDB_API_KEY")

def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="305-aec-sw",
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
    parser.add_argument("--hidden_dim", type=int, default=108, help="")
    parser.add_argument("--num_heads", type=int, default=9, help="")
    parser.add_argument("--num_layers_1", type=int, default=6, help="")
    parser.add_argument("--num_layers_2", type=int, default=6, help="")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="")

    parser.add_argument("--early_stopping_patience", type=int, default=200, help="")
    parser.add_argument("--max_acc_difference", type=float, default=0.35, help="")

    # Checkpointing
    parser.add_argument("--save_checkpoints", type=bool, default=True,
                        help="Determines whether to save weights checkpoints")

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="", help="Factor for the steplr plateu scheduler")
    parser.add_argument("--scheduler_factor", type=int, default=0.5, help="Factor for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=10,
                        help="Patience for the ReduceLROnPlateau scheduler")

    # Gaussian noise normalization
    parser.add_argument("--gaussian_mean", type=int, default=0, help="Mean parameter for Gaussian noise layer")
    parser.add_argument("--gaussian_std", type=int, default=0.00001,
                        help="Standard deviation parameter for Gaussian noise layer")

    # Visualization
    parser.add_argument("--plot_stats", type=bool, default=True,
                        help="Determines whether continuous statistics should be plotted at the end")
    parser.add_argument("--plot_lr", type=bool, default=True,
                        help="Determines whether the LR should be plotted at the end")

    parser.add_argument("--device", type=str, default='0',
                    help="Determines which Nvidia device will use (just one number)")
    # To continue training the data
    parser.add_argument("--continue_training", type=str, default="",help="path to retrieve the model for continue training")
    parser.add_argument("--transfer_learning", type=str, default="",help="path to retrieve the model for transfer learning")
    parser.add_argument("--augmentation", type=int, default=0,
                        help="Augmentations")
    parser.add_argument("--factor_aug", type=int, default=2,
                        help="factor para multiplicar los datos de augmentation")
    parser.add_argument("--batch_mean", type=int, default=0,
                        help="batch_mean flag")    
    parser.add_argument("--batch_size", type=int, default=0,
                        help="batch_size ")
    parser.add_argument("--is_weighted", type=int, default=0,
                        help="Loss crossentropy weighted ")
    parser.add_argument("--is_weighted_squared", type=int, default=0,
                        help="Loss crossentropy weighted ")
    parser.add_argument("--weighted_squared", type=int, default=1,
                        help="Loss crossentropy weighted ")
    parser.add_argument("--label_smoothing", type=float, default=0,
                        help="Loss crossentropy weighted ")
                                                
                           


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

def train(args):

    # MARK: TRAINING PREPARATION AND MODULES
    run = wandb.init(project=PROJECT_WANDB, 
                     entity=ENTITY,
                     config=args, 
                     name=args.experiment_name, 
                     job_type="model-training",
                     tags=["paper"])


    
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

    

    # DATA LOADER
        # Training set
    transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
    
    if args.augmentation:
        train_set = LSP_Dataset(args.training_set_path, transform=transform, have_aumentation=False,has_normalization=False, keypoints_model='mediapipe',factor=args.factor_aug)
    else:
        train_set = LSP_Dataset(args.training_set_path, transform=transform, have_aumentation=True, has_normalization=True,keypoints_model='mediapipe')

    # Validation set
    if args.validation_set == "from-file":
        val_set = LSP_Dataset(args.validation_set_path, keypoints_model='mediapipe', have_aumentation=False,has_normalization=True)
        
        val_loader = DataLoader(val_set, shuffle=True, generator=g)

    elif args.validation_set == "split-from-train":
        train_set, val_set = __balance_val_split(train_set, 0.2)

        val_set.transform = None
        val_set.augmentations = False
        val_loader = DataLoader(val_set, shuffle=True, generator=g)
    else:
        val_loader = None

    # Testing set
    if args.testing_set_path:
        eval_set = LSP_Dataset(args.testing_set_path, keypoints_model='mediapipe', have_aumentation=False,has_normalization=True)
        eval_loader = DataLoader(eval_set, shuffle=True, generator=g)
    else:
        eval_loader = None

    # Final training set refinements
    if args.experimental_train_split:
        train_set = __split_of_train_sequence(train_set, args.experimental_train_split)


    if args.augmentation:
        train_loader = AugmentedDataLoader(train_set, shuffle=True, generator=g)
    else:
        train_loader = DataLoader(train_set, shuffle=True, generator=g)
    # Crea un nuevo DataLoader con el collate_fn personalizado
    #train_loader = DataLoader(train_set, shuffle=True, generator=g, collate_fn=custom_collate_fn, batch_size=64)


    # RETRIEVE TRAINING
    if args.continue_training:

        slrt_model = SPOTER(num_classes=args.num_classes, num_rows=args.num_rows,
                            hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                            num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                            dim_feedforward=args.dim_feedforward)
        checkpoint = torch.load(args.continue_training)
        slrt_model.load_state_dict(checkpoint['model_state_dict'])

        sgd_optimizer = optim.SGD(slrt_model.parameters(), lr=args.lr)

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
        slrt_model = SPOTER(num_classes=args.num_classes, num_rows=args.num_rows,
                            hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                            num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                            dim_feedforward=args.dim_feedforward)
        
        sgd_optimizer = optim.SGD(slrt_model.parameters(), lr=args.lr)
    # Construct the model
        

    # Construct the other modules
    
    
    # LABEL SMOOTHING IN CRITERION
    if args.is_weighted:
        if args.is_weighted_squared:
            # CLASS WEIGHT
            print("train_set.factors",train_set.factors)
            factors = [train_set.factors[i]**args.weighted_squared for i in range(args.num_classes)]
            min_factor = min(factors)
            factors = [value/min_factor for value in factors]
            class_weight = torch.FloatTensor(factors).to(device)
            print("\\\\\\"*20)
            print("class_weight:",class_weight)
            cel_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=class_weight)    

        else:
            # CLASS WEIGHT
            class_weight = torch.FloatTensor([train_set.factors[i] for i in range(args.num_classes)]).to(device)
            print("\\\\\\"*20)
            print("class_weight:",class_weight)
            cel_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=class_weight)    
    else:
        cel_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)#, weight=class_weight)
    #cel_criterion = nn.CrossEntropyLoss()
    
    
    epoch_start = 0

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, factor=args.scheduler_factor, patience=args.scheduler_patience)
    #scheduler = optim.lr_scheduler.LambdaLR(sgd_optimizer, lr_lambda=lr_lambda)
    lr_scheduler = None

    if args.scheduler == 'steplr':
        lr_scheduler = optim.lr_scheduler.StepLR(sgd_optimizer, step_size=1, gamma=0.9995)
    if args.scheduler == 'plateu':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, mode='min', factor=0.99, patience=100, verbose=True)


    # Ensure that the path for checkpointing and for images both exist
    Path("Results/").mkdir(parents=True, exist_ok=True)
    Path("Results/checkpoints/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    Path("Results/images/metrics/").mkdir(parents=True, exist_ok=True)
    Path("Results/images/histograms/").mkdir(parents=True, exist_ok=True)
    Path("Results/images/keypoints/").mkdir(parents=True, exist_ok=True)

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



    # Log the parameters to wandb
    wandb.config.update({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_parameters_ratio": ratio
    })

    config = wandb.config
    wandb.watch_called = False

    
    # MARK: TRAINING
    train_acc, val_acc = 0, 0
    losses, train_accs, val_accs, val_accs_top5 = [], [], [], []
    lr_progress = []
    top_train_acc, top_val_acc = 0, 0
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
    for epoch in range(epoch_start, args.epochs):

        #sgd_optimizer = lr_lambda(epoch, sgd_optimizer)
        start_time = time.time()

        current_lr = sgd_optimizer.param_groups[0]["lr"]

        train_loss, _, _, train_acc,train_stats,train_labels_original,train_labels_predicted = train_epoch(slrt_model, train_loader, cel_criterion, sgd_optimizer,device,epoch=epoch,args=args)
        losses.append(train_loss.item())
        train_accs.append(train_acc)
        # Obtener la tasa de aprendizaje actual


        if val_loader:
            slrt_model.train(False)
            val_loss, _, _, val_acc, val_acc_top5, val_stats,val_labels_original,val_labels_predicted = evaluate(slrt_model, val_loader, cel_criterion, device,epoch=epoch,args=args)
            slrt_model.train(True)
            val_accs.append(val_acc)
            val_accs_top5.append(val_acc_top5)

            if val_acc > best_val_max_acc:
                best_val_max_acc = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            # Check for early stopping based on the difference between training and validation loss
            acc_difference = np.abs(train_acc - best_val_max_acc)

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

        
        train_f1_micro = f1_score(train_labels_original, train_labels_predicted, average='micro')
        val_f1_micro   = f1_score(val_labels_original, val_labels_predicted, average='micro')

        train_f1_weighted = f1_score(train_labels_original, train_labels_predicted, average='weighted')
        val_f1_weighted   = f1_score(val_labels_original, val_labels_predicted, average='weighted')

        total_time = time.time()-start_time


        if args.scheduler == 'steplr':
            lr_scheduler.step()
        if args.scheduler == 'plateu':
            lr_scheduler.step(val_loss)
            
        if val_loader:
            log_values = {
                'current_lr':current_lr,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_loss':val_loss,
                'val_best_acc': top_val_acc,
                'val_top5_acc': val_acc_top5,
                'epoch': epoch,
                'train_f1_micro':train_f1_micro,
                'val_f1_micro':val_f1_micro,
                'train_f1_weighted':train_f1_weighted,
                'val_f1_weighted':val_f1_weighted,
                'total_time':total_time
            }

        if val_loader:
            for _, row in df_train_stats.iterrows():
                gloss_name = row['train_gloss']
                accuracy_metric_name = f'train_acc_{gloss_name}'
                log_values[accuracy_metric_name] = row['train_gloss_acc']
            for _, row in df_val_stats.iterrows():
                gloss_name = row['val_gloss']
                accuracy_metric_name = f'val_acc_{gloss_name}'
                log_values[accuracy_metric_name] = row['val_gloss_acc']



            
        # Save checkpoints if they are best in the current subset
        if args.save_checkpoints:
            if val_acc > top_val_acc:

                if val_loader:

                    log_values['Train_table_stats']  = wandb.Table(dataframe=df_train_stats)
                    log_values['Val_table_stats'] =  wandb.Table(dataframe=df_val_stats)
                    log_values['Compare_table_stats'] =  wandb.Table(dataframe=df_merged)


                top_val_acc = val_acc
                model_save_folder_path = "Results/checkpoints/" + args.experiment_name

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': slrt_model.state_dict(),
                    'optimizer_state_dict': sgd_optimizer.state_dict(),
                    'loss': train_loss
                }, model_save_folder_path + "/checkpoint_best_model.pth")
                
                generate_csv_result(run, slrt_model, val_loader, model_save_folder_path, val_set.inv_dict_labels_dataset, device)
                generate_csv_accuracy(df_train_stats, model_save_folder_path,name='/train_accuracy.csv')
                generate_csv_accuracy(df_val_stats, model_save_folder_path,name='/evaluate_accuracy.csv')
                
                artifact = wandb.Artifact(f'best-model_{run.id}.pth', type='model')
                artifact.add_file(model_save_folder_path + "/checkpoint_best_model.pth")
                run.log_artifact(artifact)
                wandb.save(model_save_folder_path + "/checkpoint_best_model.pth")

                checkpoint_index += 1

        if val_loader:
            wandb.log(log_values)

        if epoch % args.log_freq == 0:
            print("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item()) + " acc: " + str(train_acc))
            logging.info("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item()) + " acc: " + str(train_acc))

            if val_loader:
                print("[" + str(epoch + 1) + "] VALIDATION  loss: " + str(val_loss.item()) + "acc: " + str(val_acc) + " top-5(acc): " + str(val_acc_top5))
                logging.info("[" + str(epoch + 1) + "] VALIDATION  loss: " + str(val_loss.item()) + "acc: " + str(val_acc) + " top-5(acc): " + str(val_acc_top5))

            print("")
            logging.info("")

        # Reset the top accuracies on static subsets
        #if epoch % 10 == 0:
        #    top_train_acc, top_val_acc, val_acc_top5 = 0, 0, 0
        #    checkpoint_index += 1

        lr_progress.append(sgd_optimizer.param_groups[0]["lr"])

    # MARK: TESTING

    print("\nTesting checkpointed models starting...\n")
    logging.info("\nTesting checkpointed models starting...\n")

    top_result, top_result_name = 0, ""

    if eval_loader:
        for i in range(checkpoint_index):
            for checkpoint_id in ["v"]: #["t", "v"]:
                # tested_model = VisionTransformer(dim=2, mlp_dim=108, num_classes=100, depth=12, heads=8)
                tested_model = torch.load("Results/checkpoints/" + args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i) + ".pth")
                tested_model.train(False)
                _, _, eval_acc = evaluate(tested_model, eval_loader, device, print_stats=True)

                if eval_acc > top_result:
                    top_result = eval_acc
                    top_result_name = args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i)

                print("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))
                logging.info("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))

        print("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
        logging.info("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")


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
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    
    
    
    @configurar_cuda_visible([int(value) for value in args.device.split(",")])
    def main_process(): 
        train(args)
    main_process()