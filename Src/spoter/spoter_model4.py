import torch
import torch.nn as nn
import copy
from torch import Tensor

from typing import Optional
from typing import Dict, List, Optional, Tuple

from torch.nn import functional as F

def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])

def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal

class SPOTERTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(SPOTERTransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)

    def forward(self,
            data: Tuple[Tensor, Tensor],
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:

        x, res = data

        xr  = x
        x   = self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
        res = res +x
        x   = self.norm1(x + xr)
        xr  = x
        x   = self._ff_block(x)
        res = res + x
        x   = self.norm2(x + xr)

        return x, res

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class SPOTERTransformerEncoder(nn.TransformerEncoder):
    def __init__(self, d_model,encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super(SPOTERTransformerEncoder, self).__init__(encoder_layer, num_layers, norm, enable_nested_tensor, mask_check)

        self.layer_norm = nn.LayerNorm(d_model, eps= 1e-5, bias=True)

        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm  = norm

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        res = src

        for mod in self.layers:
            output,res = mod((output,res), src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)
        if self.norm is not None:
            output = self.norm(output)
        output = output  + self.layer_norm(res)

        return output

class SPOTERTransformerDecoder(nn.TransformerDecoder):
    def __init__(self, d_model,decoder_layer, num_layers, norm=None):
        super(SPOTERTransformerDecoder, self).__init__(decoder_layer, num_layers,norm)

        self.layer_norm = nn.LayerNorm(d_model, eps= 1e-5, bias=True)

        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm  = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False) -> Tensor:
        
        output = tgt
        res = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output,res = mod((output,res), memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal=tgt_is_causal,
                         memory_is_causal=memory_is_causal)

        if self.norm is not None:
            output = self.norm(output)

        output = output  + self.layer_norm(res)
        return output

class SPOTERTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Edited TransformerDecoderLayer implementation omitting the redundant self-attention operation as opposed to the
    standard implementation.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(SPOTERTransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)

        #del self.self_attn
        self.norm3 = nn.LayerNorm(d_model, eps= 1e-5, bias=True)

    def forward(self, data: Tuple[Tensor, Tensor], memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_is_causal: bool = False,
                memory_is_causal: bool = False) -> torch.Tensor:

        x, res = data

        xr  = x
        #print("x : ",x.shape) # es None ,"tgt_key_padding_mask:",tgt_key_padding_mask.shape
        #print("original  :",x[0,0,:5].tolist())
        x   = self._sa_block(x, tgt_mask, tgt_key_padding_mask,tgt_is_causal)#x + self.dropout1(self.norm1(tgt))
        #print("after _sa_block:",x[0,0,:5].tolist())
        res = res +x
        x   = self.norm1(x + xr)
        xr  = x
        x   = self._mha_block(x, memory, memory_mask, memory_key_padding_mask,memory_is_causal)
        res = res +x
        x   = self.norm2(x + xr)
        xr  = x
        x   = self._ff_block(x)
        res = res + x
        x   = self.norm3(x + xr)

        return x, res
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


class SPOTER4(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes, num_rows=64,hidden_dim=108, num_heads=9, num_layers_1=6, num_layers_2=6, 
                            dim_feedforward_encoder=64,
                            dim_feedforward_decoder=256,dropout=0.3,norm_first=False,not_requires_grad_n_layers=False):
        super().__init__()

        self.hidden_dim  = hidden_dim
        self.pos         = nn.Parameter(torch.rand(1,1, hidden_dim))
        self.class_query = nn.Parameter(torch.rand(1, hidden_dim))
        print("self.pos",self.pos)
        print("self.pos",self.pos.shape)
        #https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        self.transformer  = nn.Transformer(d_model=hidden_dim, nhead =num_heads,
        num_encoder_layers= num_layers_1, 
        num_decoder_layers= num_layers_2,
        dim_feedforward = dim_feedforward_encoder,
        dropout=dropout,
        norm_first = norm_first)

        self.linear_class = nn.Linear(hidden_dim, num_classes)


        self.dropout1 = nn.Dropout(dropout)
        self.act_relu = nn.ReLU()

        self.layer_norm_encoder = nn.LayerNorm(hidden_dim, eps= 1e-5, bias=True)

        self.transformer.encoder = SPOTERTransformerEncoder(
            d_model=hidden_dim,
            encoder_layer = SPOTERTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward_encoder,
                dropout=dropout, 
                activation="relu"),
            num_layers=num_layers_1,
            norm = self.layer_norm_encoder
        )
        self.layer_norm_decoder = nn.LayerNorm(hidden_dim, eps= 1e-5, bias=True)

        self.transformer.decoder = SPOTERTransformerDecoder(
            d_model=hidden_dim,
            decoder_layer = SPOTERTransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward_encoder,
                dropout=dropout, 
                activation="relu"),
            num_layers=num_layers_2,
            norm = self.layer_norm_decoder
        )
                     
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

#python train.py --augmentation=0 --batch_name=mean_1 --batch_size=64 --data_fold=5 --data_seed=95 --device=0 --dim_feedforward_decoder=256 --dim_feedforward_encoder=256 --dropout=0.3 --early_stopping_patience=1000 --epochs=10000 "--experiment_name=NewSpoter fold-5-seed-95-p100" --factor_aug=2 --gaussian_std=0.001 --hidden_dim=108 --label_smoothing=0.1 --loss_weighted_factor=2 --lr=0.0001 --norm_first=0 --not_requires_grad_n_layers=1 --num_heads=2 --num_layers_1=3 --num_layers_2=2 --optimizer=adam --scheduler=plateu --sweep=1 --training_set_path=../SL_ConnectingPoints/split/DGI305-AEC--38--incremental--mediapipe_n_folds_5_seed_95_klod_1-Train.hdf5 --use_spoter2=4 --use_wandb=1 --validation_set_path= --weight_decay=0.0001 --weight_decay_dynamic=0

#python train.py --augmentation=0 --batch_name=mean_1 --batch_size=64 --data_fold=5 --data_seed=95 --device=0 --dim_feedforward_decoder=1024 --dim_feedforward_encoder=256 --dropout=0.3 --early_stopping_patience=1000 --epochs=10000 "--experiment_name=NewSpoter fold-5-seed-95-p100" --factor_aug=2 --gaussian_std=0.001 --hidden_dim=108 --label_smoothing=0.1 --loss_weighted_factor=2 --lr=0.0001 --norm_first=0 --not_requires_grad_n_layers=1 --num_heads=2 --num_layers_1=3 --num_layers_2=2 --optimizer=adam --scheduler=plateu --sweep=1 --training_set_path=../SL_ConnectingPoints/split/DGI305-AEC--38--incremental--mediapipe_n_folds_5_seed_95_klod_1-Train.hdf5 --use_spoter2=5 --use_wandb=0 --validation_set_path= --weight_decay=0.0001 --weight_decay_dynamic=0

#python train.py --augmentation=0 --batch_name=mean_1 --batch_size=64 --data_fold=5 --data_seed=95 --device=0 --dim_feedforward_decoder=1024 --dim_feedforward_encoder=256 --dropout=0.3 --early_stopping_patience=1000 --epochs=10000 "--experiment_name=NewSpoter fold-5-seed-95-p100" --factor_aug=2 --gaussian_std=0.001 --hidden_dim=108 --label_smoothing=0.1 --loss_weighted_factor=2 --lr=0.0001 --norm_first=0 --not_requires_grad_n_layers=1 --num_heads=2 --num_layers_1=3 --num_layers_2=2 --optimizer=adam --scheduler=plateu --sweep=1 --training_set_path=../SL_ConnectingPoints/split/DGI305-AEC--38--incremental--mediapipe_n_folds_5_seed_95_klod_1-Train.hdf5 --use_spoter2=5 --use_wandb=1 --validation_set_path= --weight_decay=0.0001 --weight_decay_dynamic=0