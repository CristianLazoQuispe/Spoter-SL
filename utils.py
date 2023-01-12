
from collections import Counter
import random
import os
import argparse
import json


import numpy as np
import torch
from torchvision import transforms
from spoter.gaussian_noise import GaussianNoise
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

from Src.Lsp_dataset import LSP_Dataset



def __balance_val_split(dataset, val_split=0.):
    targets = np.array(dataset.targets)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,
        stratify=targets
    )

    train_dataset = Subset(dataset, indices=train_indices)
    val_dataset = Subset(dataset, indices=val_indices)

    return train_dataset, val_dataset


def __split_of_train_sequence(subset: Subset, train_split=1.0):
    if train_split == 1:
        return subset

    targets = np.array([subset.dataset.targets[i] for i in subset.indices])  # type: ignore
    train_indices, _ = train_test_split(
        np.arange(targets.shape[0]),
        test_size=1 - train_split,
        stratify=targets
    )

    train_dataset = Subset(subset.dataset, indices=[subset.indices[i] for i in train_indices])

    return train_dataset


def __log_class_statistics(subset: Subset):
    train_classes = [subset.dataset.targets[i] for i in subset.indices]  # type: ignore
    print(dict(Counter(train_classes)))


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Is cuda available?")
    if torch.cuda.is_available():
        print("Cuda available")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_dataset_by_kpm(
    config,
    keypoints_model
):

    name_train_file = f"{config['dataset']}--{keypoints_model}-Train.hdf5"
    name_test_file = f"{config['dataset']}--{keypoints_model}-Val.hdf5"
    name_val_file = f"{config['dataset']}--{keypoints_model}-Val.hdf5"
    training_set_path = os.path.join(config['dataset_path'], name_train_file)
    testing_set_path = os.path.join(config['dataset_path'], name_test_file)
    validation_set_path = os.path.join(config['dataset_path'], name_val_file)

    g = torch.Generator()
    transform = transforms.Compose([GaussianNoise(config['gaussian_mean'], config['gaussian_std'])])
    train_set = LSP_Dataset(training_set_path,
                            keypoints_model, 
                            transform=transform, 
                            augmentations=False,
                            keypoints_number=config['keypoints_number']
                            )

    print('train_set',len(train_set.data))
    print('train_set',train_set.data[0].shape)

    print("Training dict encoder"+ "\n" +str(train_set.dict_labels_dataset)+ "\n")

    print("Training inv dict decoder"+ "\n" +str(train_set.inv_dict_labels_dataset)+ "\n")


    # Validation set
    if config['validation_set'] == "from-file":
        val_set = LSP_Dataset(validation_set_path, keypoints_model,
                            dict_labels_dataset=train_set.dict_labels_dataset,
                            inv_dict_labels_dataset = train_set.inv_dict_labels_dataset,keypoints_number = config['keypoints_number'])
        val_loader = DataLoader(val_set, shuffle=True, generator=g)

    elif config['validation_set'] == "split-from-train":
        train_set, val_set = __balance_val_split(train_set, 0.2)

        val_set.transform = None
        val_set.augmentations = False
        val_loader = DataLoader(val_set, shuffle=True, generator=g)

    else:
        val_loader = None

    # Testing set
    if testing_set_path:
        #eval_set = CzechSLRDataset(testing_set_path)
        eval_set = LSP_Dataset(testing_set_path,keypoints_model,
                            dict_labels_dataset=train_set.dict_labels_dataset,
                            inv_dict_labels_dataset = train_set.inv_dict_labels_dataset,keypoints_number = config['keypoints_number'])
        eval_loader = DataLoader(eval_set, shuffle=True, generator=g)

    else:
        eval_loader = None

    # Final training set refinements
    if config['experimental_train_split']:
        train_set = __split_of_train_sequence(train_set, config['experimental_train_split'])

    train_loader = DataLoader(train_set, shuffle=True, generator=g)
    
    print('train_loader',len(train_loader))

    if config['experimental_train_split']:
        print("Starting " + config['weights_trained'] + "_" + str(config['experimental_train_split']).replace(".", "") + "...\n\n")
    else:
        print("Starting " + config['weights_trained'] + "...\n\n")

    return train_loader, val_loader, eval_loader, train_set.dict_labels_dataset, train_set.inv_dict_labels_dataset



def get_dataset(
    config,
    use_wandb
):
    #config = type("configuration", (object,), config_json) if use_wandb else config_json

    train_loader, val_loader, eval_loader, dict_labels_dataset, inv_dict_labels_dataset = get_dataset_by_kpm(
                                                                                            config,
                                                                                            config['keypoints_model']
                                                                                        )

    return train_loader, val_loader, eval_loader, dict_labels_dataset, inv_dict_labels_dataset


def get_datasets_by_dsname(
    config_json,
    use_wandb
):
    #config = config_json
    config = type("configuration", (object,), config_json) if use_wandb else config_json

    dict_train_loader = {}
    dict_val_loader = {}
    dict_eval_loader = {}
    dict_dict_labels_dataset = {}
    dict_inv_dict_labels_dataset = {}

    for keypoints_model in ["openpose", "wholepose", "mediapipe"]:
        train_loader, val_loader, eval_loader, dict_labels_dataset, inv_dict_labels_dataset = get_dataset_by_kpm(
                                                                                                config,
                                                                                                keypoints_model
                                                                                            )
        dict_train_loader[keypoints_model] = train_loader
        dict_val_loader[keypoints_model] = val_loader
        dict_eval_loader[keypoints_model] = eval_loader
        dict_dict_labels_dataset[keypoints_model] = dict_labels_dataset
        dict_inv_dict_labels_dataset[keypoints_model] = inv_dict_labels_dataset

    return dict_train_loader, dict_val_loader, dict_eval_loader, dict_dict_labels_dataset, dict_inv_dict_labels_dataset



def parse_arguments_automated():
    ap = argparse.ArgumentParser()

    #WANDB ARGUMENTS
    ap.add_argument('-w', '--wandb', default=False, action='store_true',
                    help="use weights and biases")
    ap.add_argument('-n', '--no-wandb', dest='wandb', action='store_false',
                    help="not use weights and biases")
    ap.add_argument('-e', '--experimentation', default=False, action='store_true',
                    help="train several experiments for each pose estimation library for some dataset")
    ap.add_argument('-l', '--num_logs', required=False, default=5, type=int,
                    help="if experimentation, num of logs of the experiment. Required in that case")
    ap.add_argument('-r', '--exp_name', required=False, type=str, default=None,
                    help="name of the execution to save")
    ap.add_argument('-t', '--exp_notes', required=False, type=str, default=None,
                    help="notes of the execution to save")

    ap.add_argument('-ep', '--epoch', required=False, type=int, default=None,
                    help="number of epoch")
    ap.add_argument('-lr', '--learning_rate', required=False, type=float, default=None,
                    help="learning rate for transformer model")
    ap.add_argument('-kn', '--keypoints_number', required=False, type=int, default=None,
                    help="number of keypoints taken from the csv of points position")
    ap.add_argument('-ncl', '--num_coder_layers', required=False, type=int, default=None,
                    help="learning rate for transformer model")
    ap.add_argument('-df', '--dim_feedforward', required=False, type=int, default=None,
                    help="number of feedfoward dimention of the transformer model")
    ap.add_argument('-dv', '--device', required=False, type=str, nargs='+', default=None,
                    help="select just one of the GPU('s)")
    ap.add_argument('-s', '--seed', required=False, type=int, default=None,
                    help="seed for the model")
    ap.add_argument('-ds', '--dataset', required=False, type=str, default=None,
                    help="name of the dataset")
    ap.add_argument('-km', '--keypoints_model', required=False, type=str, default=None,
                    help="name of the keypoint estimation model")
                         

    args = ap.parse_args()

    return args


def parse_configuration(config_file):
    """Loads config file if a string was passed
        and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        with open(config_file, 'r') as json_file:
            return json.load(json_file)
    else:
        return config_file


def configure_model(config_file, use_wandb, arg):

    config_file = parse_configuration(config_file)

    config = dict(
        #hidden_dim = config_file["hparams"]["hidden_dim"],
        #num_classes = config_file["hparams"]["num_classes"],
        epochs = arg.epoch if arg.epoch else config_file["hparams"]["epochs"],
        num_backups = config_file["hparams"]["num_backups"],
        keypoints_model = arg.keypoints_model if arg.keypoints_model else config_file["hparams"]["keypoints_model"],
        lr = arg.learning_rate if arg.learning_rate else config_file["hparams"]["lr"],
        keypoints_number = arg.keypoints_number if arg.keypoints_number else config_file["hparams"]["keypoints_number"],

        nhead = config_file["hparams"]["nhead"],
        num_encoder_layers = arg.num_coder_layers if arg.num_coder_layers else config_file["hparams"]["num_encoder_layers"],
        num_decoder_layers = arg.num_coder_layers if arg.num_coder_layers else config_file["hparams"]["num_decoder_layers"],
        dim_feedforward = arg.dim_feedforward if arg.dim_feedforward else config_file["hparams"]["dim_feedforward"],

        experimental_train_split = config_file["hparams"]["experimental_train_split"],
        validation_set = config_file["hparams"]["validation_set"],
        validation_set_size = config_file["hparams"]["validation_set_size"],
        log_freq = config_file["hparams"]["log_freq"],
        save_checkpoints = config_file["hparams"]["save_checkpoints"],
        scheduler_factor = config_file["hparams"]["scheduler_factor"],
        scheduler_patience = config_file["hparams"]["scheduler_patience"],
        gaussian_mean = config_file["hparams"]["gaussian_mean"],
        gaussian_std = config_file["hparams"]["gaussian_std"],
        plot_stats = config_file["hparams"]["plot_stats"],
        plot_lr = config_file["hparams"]["plot_lr"],

        #training_set_path = config_file["data"]["training_set_path"],
        #validation_set_path = config_file["data"]["validation_set_path"],
        #testing_set_path = config_file["data"]["testing_set_path"],

        n_seed = arg.seed if arg.seed else config_file["seed"],
        device = arg.device if arg.device else config_file["device"],
        dataset_path = config_file["dataset_path"],
        weights_trained = config_file["weights_trained"],
        save_weights_path = config_file["save_weights_path"],
        dataset = arg.dataset if arg.dataset else config_file["dataset"]
    )

   
    print(config['keypoints_model'])
    return config