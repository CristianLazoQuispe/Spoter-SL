import copy
import torch

import torch.nn as nn
from typing import Optional


def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class SPOTERTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Edited TransformerDecoderLayer implementation omitting the redundant self-attention operation as opposed to the
    standard implementation.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(SPOTERTransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)

        del self.self_attn

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class SPOTER(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes, num_rows=64,hidden_dim=108, num_heads=9, num_layers_1=6, num_layers_2=6, 
                            dim_feedforward_encoder=64,
                            dim_feedforward_decoder=256,dropout=0.3,norm_first=False,freeze_decoder_layers=False):
        super().__init__()

        self.hidden_dim  = hidden_dim
        self.pos         = nn.Parameter(torch.rand(1,1, hidden_dim))
        self.class_query = nn.Parameter(torch.rand(1, hidden_dim))
        #print("self.pos",self.pos)
        #print("self.pos",self.pos.shape)
        #https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        self.transformer  = nn.Transformer(d_model=hidden_dim, nhead =num_heads,
        num_encoder_layers= num_layers_1, 
        num_decoder_layers= num_layers_2,
        dim_feedforward = dim_feedforward_encoder,
        dropout=dropout,
        norm_first = norm_first)

        self.linear_class = nn.Linear(hidden_dim, num_classes)

        custom_decoder_layer = SPOTERTransformerDecoderLayer(self.transformer.d_model, self.transformer.nhead,
                                                             dim_feedforward_decoder, dropout=dropout, activation="relu",norm_first=norm_first)

        self.dropout1 = nn.Dropout(dropout)
        self.act_relu = nn.ReLU()
        self.transformer.decoder.layers = _get_clones(custom_decoder_layer, self.transformer.decoder.num_layers)

        if freeze_decoder_layers:
            print("CONGELAR CAPAS")
            print("CONGELAR CAPAS")
            print("CONGELAR CAPAS")
            for param in self.transformer.decoder.layers[:3].parameters():
                param.requires_grad = False # Congelar capas
            
    def forward(self, inputs,show=False):
        h = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
        aux = self.pos + h
        h = self.transformer(aux, self.class_query.unsqueeze(0)).transpose(0, 1)
        h = self.act_relu(h)
        h = self.dropout1(h)
        res = self.linear_class(h)
        return res


if __name__ == "__main__":
    pass
