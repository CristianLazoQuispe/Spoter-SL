import os
from datetime import datetime
import sys

import wandb

from spoter.training_spoter import TrainingSpoter

class ExperimenterSpoter():
    def __init__(self,
                config, 
                use_wandb, 
                num_logs,
                dict_train_loader,
                dict_val_loader,
                dict_eval_loader,
                dict_dict_labels_dataset,
                dict_inv_dict_labels_dataset
                ):
        self.config = type("configuration", (object,), config) if use_wandb else config
        self.config_json = config
        self.use_wandb = use_wandb
        self.num_logs = num_logs

        self.dict_train_loader = dict_train_loader
        self.dict_val_loader = dict_val_loader
        self.dict_eval_loader = dict_eval_loader
        self.dict_dict_labels_dataset = dict_dict_labels_dataset
        self.dict_inv_dict_labels_dataset = dict_inv_dict_labels_dataset


    def train_experiments(self, 
                        entity,
                        project_wandb,
                        exp_name, 
                        exp_notes
                        ):

        now = datetime.now()
        now_txt = str(now).replace(":","-").replace(".", "-").replace(" ", "_")
        name_dataset = self.config.dataset
        keypoints_number = self.config.keypoints_number
        exp_name = f"Exp_{name_dataset}_{keypoints_number}_{now_txt}" if exp_name is None else exp_name

        path_save_experiments = os.path.join(self.config.save_weights_path, exp_name)
        try:
            os.mkdir(path_save_experiments)
        except OSError:
            pass

        for num_log in range(self.num_logs):
            log_name = f"log n. {num_log+1}"
            name_default = f"{exp_name} - {log_name}"

            path_save_weights = os.path.join(path_save_experiments, log_name)
            try:
                os.mkdir(path_save_weights)
            except OSError:
                pass

            if self.use_wandb:
                run = wandb.init(reinit=True, 
                                project=project_wandb,
                                entity=entity,
                                config=self.config_json,
                                name=name_default,
                                notes=exp_notes)
                self.config = run.config
                wandb.watch_called = False
            else:
                run = None
            
            spoter_trainer = TrainingSpoter(config=self.config, use_wandb=self.use_wandb,
                                            path_save_weights=path_save_weights
                                            )
            spoter_trainer.train_experiment(
                self.dict_train_loader,
                self.dict_val_loader,
                self.dict_eval_loader,
                self.dict_dict_labels_dataset,
                self.dict_inv_dict_labels_dataset  
            )