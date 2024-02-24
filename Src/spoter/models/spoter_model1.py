import copy
import torch
import warnings
from typing import Optional, Any, Union, Callable
from torch import Tensor

import torch.nn as nn

def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class SPOTERTransformerDecoderLayer2(nn.TransformerDecoderLayer):
    """
    Edited TransformerDecoderLayer implementation omitting the redundant self-attention operation as opposed to the
    standard implementation.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation,norm_first):
        super(SPOTERTransformerDecoderLayer2, self).__init__(d_model, nhead, dim_feedforward, dropout, activation,norm_first)

        #del self.self_attn

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_is_causal: bool = False,
                memory_is_causal: bool = False) -> torch.Tensor:

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask,tgt_is_causal)#x + self.dropout1(self.norm1(tgt))
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask,memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask,tgt_is_causal))#self.norm1(x + self.dropout1(tgt))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask,memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)



class SPOTER1(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes, num_rows=64,hidden_dim=108, num_heads=9, num_layers_1=6, num_layers_2=6, 
                            dim_feedforward_encoder=64,
                            dim_feedforward_decoder=256,dropout=0.3,norm_first=False,freeze_decoder_layers=False):
        super(SPOTER1).__init__()

        self.hidden_dim  = hidden_dim
        self.pos         = nn.Parameter(torch.rand(1,1, hidden_dim))
        self.class_query = nn.Parameter(torch.rand(1, hidden_dim))
        #https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        self.transformer  = nn.Transformer(d_model=hidden_dim, nhead =num_heads,
        num_encoder_layers= num_layers_1, 
        num_decoder_layers= num_layers_2,
        dim_feedforward = dim_feedforward_encoder,
        dropout=dropout,
        norm_first = norm_first)

        self.linear_class = nn.Linear(hidden_dim, num_classes)

        custom_decoder_layer = SPOTERTransformerDecoderLayer2(self.transformer.d_model, self.transformer.nhead,
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
