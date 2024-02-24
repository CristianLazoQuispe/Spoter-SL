import torch
import torch.nn as nn

class SPOTER2(nn.Module):
    """
    Implementación del encoder SPOTER (Sign POse-based TransformER) para reconocimiento de lenguaje de señas a partir de datos esqueléticos.
    """

    def __init__(self, num_classes, num_rows=64,hidden_dim=108, num_heads=9, num_layers_1=6, num_layers_2=6, 
                            dim_feedforward_encoder=64,
                            dim_feedforward_decoder=256,dropout=0.3,norm_first=False,has_mlp=False):

        super(SPOTER2, self).__init__()

        self.hidden_dim = hidden_dim
        self.pos = nn.Parameter(torch.rand(1, 1, hidden_dim))
        self.class_query = nn.Parameter(torch.rand(1, hidden_dim))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward_encoder,
                dropout=dropout,
                norm_first = False),
            num_layers=num_layers_1
        )

        self.linear_class = nn.Linear(hidden_dim, num_classes)
        self.act_relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.has_mlp = has_mlp

        self.linear_class_1 = nn.Linear(hidden_dim, 64)
        self.linear_class_2 = nn.Linear(64, 32)
        self.linear_class_3 = nn.Linear(32, num_classes)
        self.act_relu_2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs):
        h = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
        aux = self.pos + h
        h = self.encoder(aux)
        h = h.mean(dim=0) 

        if self.has_mlp:
            h = self.linear_class_1(h)            
            h = self.act_relu(h)
            h = self.dropout(h)

            h = self.linear_class_2(h)            
            h = self.act_relu_2(h)
            h = self.dropout2(h)
            res = self.linear_class_3(h)

        else:
            h = self.act_relu(h)
            h = self.dropout(h)
            res = self.linear_class(h)
        return res

#python train.py --augmentation=0 --batch_name=mean_1 --batch_size=64 --data_fold=5 --data_seed=95 --device=0 --dim_feedforward_decoder=256 --dim_feedforward_encoder=256 --dropout=0.3 --early_stopping_patience=1000 --epochs=10000 "--experiment_name=NewSpoter fold-5-seed-95-p100" --factor_aug=2 --gaussian_std=0.001 --hidden_dim=108 --label_smoothing=0.1 --loss_weighted_factor=2 --lr=0.0001 --norm_first=0 --not_requires_grad_n_layers=1 --num_heads=2 --num_layers_1=3 --num_layers_2=2 --optimizer=adam --scheduler=plateu --sweep=1 --training_set_path=../SL_ConnectingPoints/split/DGI305-AEC--38--incremental--mediapipe_n_folds_5_seed_95_klod_1-Train.hdf5 --use_spoter2=2 --use_wandb=1 --validation_set_path= --weight_decay=0.0001 --weight_decay_dynamic=0


 #h : torch.Size([11, 1, 108])
 #aux : torch.Size([11, 1, 108])
 #h encoder : torch.Size([11, 1, 108])
 #h mean : torch.Size([1, 108])
 #h : torch.Size([1, 108])

#python train.py --augmentation=0 --batch_name=mean_1 --batch_size=64 --data_fold=5 --data_seed=95 --device=0 --dim_feedforward_decoder=256 --dim_feedforward_encoder=16 --dropout=0.3 --early_stopping_patience=1000 --epochs=10000 "--experiment_name=NewSpoter fold-5-seed-95-p100" --factor_aug=2 --gaussian_std=0.001 --hidden_dim=108 --label_smoothing=0.1 --loss_weighted_factor=2 --lr=0.0001 --norm_first=0 --not_requires_grad_n_layers=1 --num_heads=9 --num_layers_1=6 --num_layers_2=2 --optimizer=adam --scheduler=plateu --sweep=1 --training_set_path=../SL_ConnectingPoints/split/DGI305-AEC--38--incremental--mediapipe_n_folds_5_seed_95_klod_1-Train.hdf5 --use_spoter2=2 --use_wandb=1 --validation_set_path= --weight_decay=0.0001 --weight_decay_dynamic=0
