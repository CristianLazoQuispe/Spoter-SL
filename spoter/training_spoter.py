import os
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from spoter.spoter_model import SPOTER
from spoter.utils import train_epoch, evaluate, my_evaluate, evaluate_top_k, get_metrics_epoch_zero

HIDDEN_DIM = {
    "29": 58,
    "51": 102,
    "71": 142
}

NUM_CLASSES = {
    "AEC": 28,
    "PUCP_PSL_DGI156": 29,
    "WLASL": 86,
    "AEC-DGI156-DGI305":72
}


def get_cuda_device():
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

    return multigpu, n_cuda, device

class TrainingSpoter():
    def __init__(
        self,
        config,
        path_save_weights,
        use_wandb=True,
        run = None,
        
    ):
        print("Starting training ...")
        self.use_wandb = use_wandb
        self.config = config
        self.run = run
        
        self.artifact_name = self.config['dataset']+'-'+str(self.config['keypoints_number'])+'-'+str(self.config['keypoints_model'])+'-'+str(self.run.id)+'-'+'model' 
        print("artifact_name : ",self.artifact_name)    
        self.model_artifact = wandb.Artifact(self.artifact_name, type='model')

        #n_cuda = os.getenv('N_CUDA') if os.getenv('N_CUDA') else str(*config['device'])
        #print(f"Ncuda = {n_cuda}")
        #self.device = torch.device("cuda:" + (n_cuda) if torch.cuda.is_available() else "cpu")
        multigpu, n_cuda, device = get_cuda_device()
        self.device = device
        
        self.path_save_weights = path_save_weights
        


    def save_weights(self, model, path_sub, keypoints_model, use_wandb=True):
        name_file = 'spoter-sl.pth' if keypoints_model=="" else f'spoter-sl-{keypoints_model}.pth'
        print("torch.save : ",os.path.join(path_sub, name_file))
        torch.save(model.state_dict(), os.path.join(path_sub, name_file))

        if use_wandb:
            wandb.save(os.path.join(path_sub, '*.pth'),
                    base_path='/'.join(path_sub.split('/')[:-2]))
            
            #if self.run !=None:
            #    print("log artifact wandb  : ",os.path.join(path_sub, name_file))
            #    self.model_artifact_test = wandb.Artifact(self.artifact_name+'_test', type='model')

            #    self.model_artifact_test.add_file(os.path.join(path_sub, name_file))
            #    self.run.log_artifact(self.model_artifact_test)




    def save_dict_labels_dataset(
        self,
        path_save_weights,
        dict_labels_dataset,
        inv_dict_labels_dataset,
        keypoints_model
    ):
        #TO-DO: Save Encoders
        name_encoder = f"dict_labels_dataset_{self.config['dataset']}_{keypoints_model}.json"
        name_inv_encoder = f"inv_dict_labels_dataset_{self.config['dataset']}_{keypoints_model}.json"
        path_encoder = os.path.join(path_save_weights, name_encoder)
        path_inv_encoder = os.path.join(path_save_weights, name_inv_encoder)
        
        with open(path_encoder, 'w') as f:
            json.dump(dict_labels_dataset, f)
        with open(path_inv_encoder, 'w') as f:
            json.dump(inv_dict_labels_dataset, f)

        if self.use_wandb:
            wandb.save(path_encoder)
            wandb.save(path_inv_encoder)


    def train_epoch_metrics(
        self,
        slrt_model,
        train_loader,
        val_loader,
        eval_loader,
        cel_criterion,
        sgd_optimizer,
        epoch,
        max_eval_acc,
        max_eval_acc_top5,
        keypoints_model=""
    ):
        slrt_model.train(True)
        train_loss, _, _, train_acc = train_epoch(slrt_model, train_loader, cel_criterion, sgd_optimizer, self.device)
        metrics_log = {"train_loss" if keypoints_model=="" else f"train_loss-{keypoints_model}": train_loss,
                        "train_acc" if keypoints_model=="" else f"train_acc-{keypoints_model}": train_acc
                    }
        print(f"Training epoch {keypoints_model}:")
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch +
                                                        1, self.config['epochs'], train_loss))
        print('Epoch [{}/{}], train_acc: {:.4f}'.format(epoch +
                                                        1, self.config['epochs'], train_acc))
        
        if val_loader:
            slrt_model.train(False)
            loss, _, _, val_acc = evaluate(slrt_model, val_loader, cel_criterion, self.device)
            slrt_model.train(True)
            metrics_log["val_acc" if keypoints_model=="" else f"val_acc-{keypoints_model}"] = val_acc

            print('Epoch [{}/{}], val_acc: {:.4f}'.format(epoch +
                                            1, self.config['epochs'], val_acc))

        if eval_loader:
            slrt_model.train(False)
            eval_loss, _, _, eval_acc = evaluate(slrt_model, eval_loader, cel_criterion, self.device, print_stats=True)
            _, _, eval_acctop5 = evaluate_top_k(slrt_model, eval_loader, self.device, k=5)
            slrt_model.train(True)

            print("eval_acc : ",eval_acc)              
            if eval_acc > max_eval_acc:
                max_eval_acc = eval_acc
                
                print("better results")  
                print("max_eval_acc : ",max_eval_acc)              
                path_save_epoch = os.path.join(self.path_save_weights, 'best_model')
                print("Saving model in ",path_save_epoch)
                try:
                    os.mkdir(path_save_epoch)
                except OSError:
                    print("error creating path ",path_save_epoch)
                    pass
                
                self.save_weights(slrt_model, path_save_epoch, keypoints_model, self.use_wandb)


                
                
            if eval_acctop5 > max_eval_acc_top5:
                max_eval_acc_top5 = eval_acctop5

            metrics_log['eval_loss' if keypoints_model=="" else f"eval_acc-{keypoints_model}"] = eval_loss
            metrics_log["eval_acc" if keypoints_model=="" else f"eval_acc-{keypoints_model}"] = eval_acc
            metrics_log["eval_acctop5" if keypoints_model=="" else f"eval_acctop5-{keypoints_model}"] = eval_acctop5
            metrics_log["max_eval_acc" if keypoints_model=="" else f"max_eval_acc-{keypoints_model}"] = max_eval_acc
            metrics_log["max_eval_acc_top5" if keypoints_model=="" else f"max_eval_acc_top5-{keypoints_model}"] = max_eval_acc_top5

            print('Epoch [{}/{}], eval_loss: {:.4f}'.format(epoch +
                                                        1, self.config['epochs'], eval_loss))
            print('Epoch [{}/{}], eval_acc: {:.4f}'.format(epoch +
                                            1, self.config['epochs'], eval_acc))
            print('Epoch [{}/{}], eval_acctop5: {:.4f}'.format(epoch +
                                            1, self.config['epochs'], eval_acctop5))
            print('Epoch [{}/{}], max_eval_acc: {:.4f}'.format(epoch +
                                            1, self.config['epochs'], max_eval_acc))
            print('Epoch [{}/{}], max_eval_acc_top5: {:.4f}'.format(epoch +
                                            1, self.config['epochs'], max_eval_acc_top5))

        print("self.config['num_backups']",self.config['num_backups'])
        print("self.config['epochs']",self.config['epochs'])
        print("(epoch+1)",(epoch+1))
        print("self.config['epochs']/self.config['num_backups']+1",self.config['epochs']/self.config['num_backups']+1)
        if ((epoch+1) % int(self.config['epochs']/self.config['num_backups']+1)) == 0:
            path_save_epoch = os.path.join(self.path_save_weights, 'epoch_{}'.format(epoch+1))
            
            print("Saving model in ",path_save_epoch)
            try:
                os.mkdir(path_save_epoch)
            except OSError:
                print("error creating path ",path_save_epoch)
                pass
            
            self.save_weights(slrt_model, path_save_epoch, keypoints_model, self.use_wandb)

        return metrics_log, max_eval_acc, max_eval_acc_top5


    def train_experiment(
        self,
        dict_train_loader,
        dict_val_loader,
        dict_eval_loader,
        dict_dict_labels_dataset,
        dict_inv_dict_labels_dataset    
    ):
        self.dict_train_loader = dict_train_loader
        self.dict_val_loader = dict_val_loader
        self.dict_eval_loader = dict_eval_loader
        self.dict_dict_labels_dataset = dict_dict_labels_dataset
        self.dict_inv_dict_labels_dataset = dict_inv_dict_labels_dataset


        if torch.cuda.is_available():
            print("Training in " + torch.cuda.get_device_name(0))  
        else:
            print("Training in CPU")

        for keypoints_model in self.dict_dict_labels_dataset.keys():
            print(keypoints_model)
            self.save_dict_labels_dataset(self.path_save_weights,
                                        self.dict_dict_labels_dataset[keypoints_model],
                                        self.dict_inv_dict_labels_dataset[keypoints_model],
                                        keypoints_model
                                        )      
          
        self.slrt_model_op = SPOTER(num_classes=NUM_CLASSES[self.config['dataset']], 
                                hidden_dim=HIDDEN_DIM[str(self.config['keypoints_number'])],
                                dim_feedforward=self.config['dim_feedforward'],
                                num_encoder_layers=self.config['num_encoder_layers'],
                                num_decoder_layers=self.config['num_decoder_layers'],
                                nhead=self.config['nhead']
                                )

        self.slrt_model_wp = SPOTER(num_classes=NUM_CLASSES[self.config['dataset']], 
                                hidden_dim=HIDDEN_DIM[str(self.config['keypoints_number'])],
                                dim_feedforward=self.config['dim_feedforward'],
                                num_encoder_layers=self.config['num_encoder_layers'],
                                num_decoder_layers=self.config['num_decoder_layers'],
                                nhead=self.config['nhead']
                                )

        self.slrt_model_mp = SPOTER(num_classes=NUM_CLASSES[self.config['dataset']], 
                                hidden_dim=HIDDEN_DIM[str(self.config['keypoints_number'])],
                            dim_feedforward=self.config['dim_feedforward'],
                                num_encoder_layers=self.config['num_encoder_layers'],
                                num_decoder_layers=self.config['num_decoder_layers'],
                                nhead=self.config['nhead']
                                )

        self.slrt_model_wp.load_state_dict(self.slrt_model_op.state_dict())
        self.slrt_model_mp.load_state_dict(self.slrt_model_op.state_dict())

        dict_slrt_model = {
            "openpose": self.slrt_model_op,
            "wholepose": self.slrt_model_wp,
            "mediapipe": self.slrt_model_mp
        }

        dict_criterion = {}
        dict_sgd_optimizer = {}
        dict_max_eval_acc = {}
        dict_max_eval_acc_top5 = {}

        for keypoints_model in self.dict_dict_labels_dataset.keys():
            dict_slrt_model[keypoints_model].train(True)
            dict_slrt_model[keypoints_model].to(self.device)

            dict_criterion[keypoints_model] = nn.CrossEntropyLoss()
            dict_sgd_optimizer[keypoints_model] = optim.SGD(dict_slrt_model[keypoints_model].parameters(), 
                                                            lr=self.config['lr']
                                                            )
            dict_max_eval_acc[keypoints_model] = 0
            dict_max_eval_acc_top5[keypoints_model] = 0


        metrics_log_epoch_zero = {"train_epoch": 0}
        for keypoints_model in dict_slrt_model.keys():
            metrics_log = get_metrics_epoch_zero(dict_slrt_model[keypoints_model], 
                                                dict_train_loader[keypoints_model],
                                                dict_val_loader[keypoints_model],
                                                dict_eval_loader[keypoints_model],
                                                dict_criterion[keypoints_model],
                                                self.device,
                                                keypoints_model=keypoints_model)
            metrics_log_epoch_zero.update(metrics_log)

        if self.use_wandb:
            wandb.log(metrics_log_epoch_zero)

        for epoch in tqdm(range(self.config['epochs'])):
            
            metrics_log_epoch = {"train_epoch": epoch+1}
            
            for keypoints_model in dict_slrt_model.keys():
                metrics_log, max_eval_acc_new, max_eval_acc_top5_new = self.train_epoch_metrics(dict_slrt_model[keypoints_model],
                                        dict_train_loader[keypoints_model],
                                        dict_val_loader[keypoints_model],
                                        dict_eval_loader[keypoints_model],
                                        dict_criterion[keypoints_model],
                                        dict_sgd_optimizer[keypoints_model],
                                        epoch,
                                        dict_max_eval_acc[keypoints_model],
                                        dict_max_eval_acc_top5[keypoints_model],
                                        keypoints_model
                                    )
                dict_max_eval_acc[keypoints_model] = max_eval_acc_new
                dict_max_eval_acc_top5[keypoints_model] = max_eval_acc_top5_new

                metrics_log_epoch.update(metrics_log)
            
            if self.use_wandb:
                wandb.log(metrics_log_epoch)

        if self.use_wandb:
            wandb.finish()


    def train(
        self,
        train_loader,
        val_loader,
        eval_loader,
        dict_labels_dataset,
        inv_dict_labels_dataset
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eval_loader = eval_loader
        self.dict_labels_dataset = dict_labels_dataset
        self.inv_dict_labels_dataset = inv_dict_labels_dataset

        self.slrt_model = SPOTER(num_classes=NUM_CLASSES[self.config['dataset']],
                                hidden_dim=HIDDEN_DIM[str(self.config['keypoints_number'])],
                                dim_feedforward=self.config['dim_feedforward'],
                                num_encoder_layers=self.config['num_encoder_layers'],
                                num_decoder_layers=self.config['num_decoder_layers'],
                                nhead=self.config['nhead']
                                )

        print(self.slrt_model)

        if torch.cuda.is_available():
            print("Training in " + torch.cuda.get_device_name(0))  
        else:
            print("Training in CPU")

        print("#"*50)
        print("#"*30)
        print("#"*10)
        print("Num Trainable Params: ",sum(p.numel() for p in self.slrt_model.parameters() if p.requires_grad))
        print("#"*10)
        print("#"*30)
        print("#"*50)


        self.save_dict_labels_dataset(self.path_save_weights,
                                    self.dict_labels_dataset,
                                    self.inv_dict_labels_dataset,
                                    self.config['keypoints_model']
                                    )

        self.slrt_model.train(True)
        self.slrt_model.to(self.device)

        cel_criterion = nn.CrossEntropyLoss()
        sgd_optimizer = optim.SGD(self.slrt_model.parameters(), 
                                    lr=self.config['lr']
                                    )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, 
                                                        factor=self.config['scheduler_factor'], 
                                                        patience=self.config['scheduler_patience']
                                                        )

        max_eval_acc = 0
        max_eval_acc_top5 = 0

        try:
            for epoch in tqdm(range(self.config['epochs'])):

                metrics_log, max_eval_acc_new, max_eval_acc_top5_new = self.train_epoch_metrics(self.slrt_model,
                                        self.train_loader,
                                        self.val_loader,
                                        self.eval_loader,
                                        cel_criterion,
                                        sgd_optimizer,
                                        epoch,
                                        max_eval_acc,
                                        max_eval_acc_top5,
                                        keypoints_model=""#str(self.config['keypoints_model'])
                                    )
                max_eval_acc = max_eval_acc_new
                max_eval_acc_top5 = max_eval_acc_top5_new
                
                metrics_log["train_epoch"] = epoch + 1

                if self.use_wandb:
                    wandb.log(metrics_log)
                    
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
                        
            print("finished wandb ")        
            
        if self.use_wandb:
            print("Finish wandb ")
            if self.run !=None:

                path_save_epoch = os.path.join(self.path_save_weights, 'best_model')
                print("Saving model in ",path_save_epoch)
                
                keypoints_model = ""#str(self.config['keypoints_model'])
                name_file = 'spoter-sl.pth' if keypoints_model=="" else f'spoter-sl-{keypoints_model}.pth'
                print("log artifact wandb  : ",os.path.join(path_save_epoch, name_file))

                self.model_artifact.add_file(os.path.join(path_save_epoch, name_file))
                self.run.log_artifact(self.model_artifact)

            wandb.finish()