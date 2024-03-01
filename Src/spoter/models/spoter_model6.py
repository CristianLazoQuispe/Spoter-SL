import torch
import torch.nn as nn
import copy
from torch import Tensor

from typing import Optional
from typing import Dict, List, Optional, Tuple

from torch.nn import functional as F

def _get_clones(mod, n,is_list=False):
    if is_list:
        return nn.ModuleList(mod)
    else:
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

def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


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

        self.dropout_new = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model, eps= 1e-5)

    def forward(self,
            data: Tuple[Tensor, Tensor],
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:

        x, res = data
        x   = self.norm3(x + self.dropout_new(x))
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



class SPOTERTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Edited TransformerDecoderLayer implementation omitting the redundant self-attention operation as opposed to the
    standard implementation.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(SPOTERTransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)

        #del self.self_attn
        self.norm3 = nn.LayerNorm(d_model, eps= 1e-5)
        self.dropout_new = nn.Dropout(dropout)

    def forward(self, data: Tuple[Tensor, Tensor], memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_is_causal: bool = False,
                memory_is_causal: bool = False) -> torch.Tensor:

        x, res = data

        x   = x +self.dropout_new(x)
        x   = self.norm1(x)

        xr  = x
        #print("x : ",x.shape) # es None ,"tgt_key_padding_mask:",tgt_key_padding_mask.shape
        #print("original  :",x[0,0,:5].tolist())
        
        """
        x   = self._sa_block(x, tgt_mask, tgt_key_padding_mask,tgt_is_causal)#x + self.dropout1(self.norm1(tgt))
        #print("after _sa_block:",x[0,0,:5].tolist())
        res = res +x
        x   = self.norm1(x + xr)
        xr  = x
        """
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



class SPOTERTransformerEncoder(nn.TransformerEncoder):
    def __init__(self, d_model,encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super(SPOTERTransformerEncoder, self).__init__(encoder_layer[0], num_layers, norm, enable_nested_tensor, mask_check)

        self.layer_norm = nn.LayerNorm(d_model, eps= 1e-5)

        self.layers = _get_clones(encoder_layer, num_layers,is_list=True)
        self.norm  = norm

        #self.positional_encoder = PositionalEmbedding(200, d_model) # 200 maximo de frames

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
       
        output = output  + self.layer_norm(res)

        if self.norm is not None:
            output = self.norm(output)

        return output

class SPOTERTransformerDecoder(nn.TransformerDecoder):
    def __init__(self, d_model,decoder_layer, num_layers, norm=None):
        super(SPOTERTransformerDecoder, self).__init__(decoder_layer[0], num_layers,norm)

        self.layer_norm = nn.LayerNorm(d_model, eps= 1e-5)

        self.layers = _get_clones(decoder_layer, num_layers,is_list=True)
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

        output = output  + self.layer_norm(res)

        if self.norm is not None:
            output = self.norm(output)

        return output
    
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        #pe = torch.zeros(max_len, d_model)
        print("position.shape:",position.shape)
        pe[:,0,0::2] = torch.sin(position * div_term)
        pe[:,0,1::2] = torch.cos(position * div_term)
        #pe[:,0::2] = torch.sin(position * div_term)
        #pe[:,1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        #print("self.pe.shape:",self.pe.shape)
        
        show = False

        x = x * math.sqrt(self.d_model)
        if show:
            print('**'*20)
            print('**'*20)
            print('x.shape',x.shape)
            print("x",x[0,:4])
            print('self.pe.shape',self.pe.shape)
            print("self.pe",self.pe[:x.size(0),:][0,:8])
            print("self.pe",self.pe[:x.size(0),:][1,:8])
            print("self.pe",self.pe[:x.size(0),:][2,:8])
            print("self.pe",self.pe[:x.size(0),:][3,:8])
            print("self.pe",self.pe[:x.size(0),:][-1,:8])
        x = x + torch.autograd.Variable(self.pe[:x.size(0),:], requires_grad=False)
        if show:
            print("x",x[0,:4])
        x = self.dropout(x)
        if show:
            print("x",x[0,:4])
            print('**'*20)
            print('**'*20)
        return x
    
class SPOTER6(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes,hidden_dim=108, num_heads=3, num_layers_1=3, num_layers_2=3, 
                            dim_feedforward_encoder=1024,
                            dim_feedforward_decoder=2048,dropout=0.3):
        super(SPOTER6,self).__init__()

        self.hidden_dim  = hidden_dim
        self.pos         = nn.Parameter(torch.rand(1,1, hidden_dim))
        self.class_query = nn.Parameter(torch.rand(1,1,hidden_dim))
        #self.class_query = nn.Parameter(torch.rand(1,hidden_dim))
        #https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        
        #self.transformer  = nn.Transformer(d_model=hidden_dim, nhead =num_heads,
        #num_encoder_layers= num_layers_1, 
        #num_decoder_layers= num_layers_2,
        #dim_feedforward = dim_feedforward_encoder,
        #dropout=dropout) # norm_first is not necessary because we change the encoder decoder with dual residual norm

        self.linear_class = nn.Linear(hidden_dim, num_classes)


        self.dropout1 = nn.Dropout(dropout)
        self.act_relu = nn.ReLU()

        self.layer_norm_encoder = nn.LayerNorm(hidden_dim, eps= 1e-5)

        self.encoder = SPOTERTransformerEncoder(
            d_model=hidden_dim,
            encoder_layer = [SPOTERTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward_encoder//(i+1),
                dropout=dropout, 
                activation="relu")
                for i in range(num_layers_1)
                ],
            num_layers=num_layers_1,
            norm = self.layer_norm_encoder
        )
        self.layer_norm_decoder = nn.LayerNorm(hidden_dim, eps= 1e-5)

        self.decoder_gen = SPOTERTransformerDecoder(
            d_model=hidden_dim,
            decoder_layer  = [SPOTERTransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward_decoder//(i+1),
                dropout=dropout, 
                activation="relu")
                for i in range(num_layers_2)
                ],
            num_layers=num_layers_2,
            norm = self.layer_norm_decoder
        )
        self.decoder_class = SPOTERTransformerDecoder(
            d_model=hidden_dim,
            decoder_layer = [SPOTERTransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=3,
                dim_feedforward=dim_feedforward_decoder//(i+1),
                dropout=dropout, 
                activation="relu")
                for i in range(num_layers_2)
                ],
            num_layers=num_layers_2,
            norm = self.layer_norm_decoder
        )
        self.linear_class_1 = nn.Linear(hidden_dim, 64) # 64 to avoid overfitting because hidden_dim is 108
        self.linear_class_2 = nn.Linear(64,num_classes)
        self.act_relu_2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.positional_encoder = PositionalEncoding(hidden_dim)
        self.positional_decoder = PositionalEncoding(hidden_dim)
        
        #self.ff_softmax = nn.Softmax(dim=1) # esto no se hace cuando se usa crossentropy loss
        self.cnt = 0

    def forward(self, inputs,show=False):
        show= True if self.cnt==0 else False
        self.cnt+=1
        print("") if show else None
        #inputs = inputs+0.5
        print("inputs.shape",inputs.shape) if show else None #inputs.shape torch.Size([28, 54, 2])
        
        h = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
        #h = inputs.flatten(start_dim=1).float()

        print("h.shape",h.shape) if show else None#h.shape torch.Size([28, 1, 108])

        generation = None

        src        = h[:-2,:] if h.shape[0]>2 else h[:-1,:]
        tgt        = h[-2,:].unsqueeze(0) if h.shape[0]>2 else h[-1,:].unsqueeze(0)
        tgt_future = h[-1,:].unsqueeze(0) 

        print("") if show else None
        print("src.shape",src.shape) if show else None
        print("tgt.shape",tgt.shape) if show else None
        #output = self.transformer(src,tgt)#.transpose(0, 1)
        print("self.positional_encoder(src).shape:",self.positional_encoder(src).shape) if show else None
        if h.shape[0]>5:
            print("self.positional_encoder(src):",src[0,:4],self.positional_encoder(src)[0,:4]) if show else None
            print("self.positional_encoder(src):",src[1,:4],self.positional_encoder(src)[1,:4]) if show else None
            print("self.positional_encoder(src):",src[2,:4],self.positional_encoder(src)[2,:4]) if show else None

        memory = self.encoder(self.positional_encoder(src))
        print("memory.shape",memory.shape)  if show else None# torch.Size([14, 108])
        print("memory:",memory[0,:4].tolist())  if show else None#[0.057049673050642014, -2.7947468757629395, -2.5759713649749756, -1.494513988494873]
        #generation = torch.sigmoid(self.decoder_gen(tgt, memory)) #encoder decoder generation: [0.19776038825511932, -3.6382975578308105, 1.1325629949569702, -0.1620892882347107]
        embedding  = self.decoder_class(self.class_query, memory) #encoder decoder generation: [0.19776038825511932, -3.6382975578308105, 1.1325629949569702, -0.1620892882347107]
        if generation is not None:
            print("generation.shape",generation.shape)  if show else None# torch.Size([1, 108])
            print("generation:",generation[0,:4].tolist())  if show else None#transformer generation: [[0.19776038825511932, -3.6382975578308105, 1.1325629949569702, -0.1620892882347107]]
        # we prove that is equal implementation
        #embedding = torch.mean(memory,0)#.unsqueeze(0)
        print("embedding.shape",embedding.shape)  if show else None# torch.Size([1,108])
        #h = self.linear_class_1(embedding)
        h = embedding
        h = self.act_relu(h)
        h = self.dropout1(h)
        res = self.linear_class(h)
        print("res.shape",res.shape)  if show else None# torch.Size([1, 38])

        #tgt_future = tgt_future-0.5
        #generation= generation-0.5
        #raise
        return res,tgt_future,torch.sigmoid(embedding)#generation


if __name__ == "__main__":
    pass

##tmux a -t session_02  python train.py --augmentation=0 --batch_size=64 --data_fold=5 --data_seed=95 --device=1 --dim_feedforward_decoder=1024 --dim_feedforward_encoder=512 --early_stopping_patience=1000 --epochs=20000  --model_name=generative_class_residual_piramidal --num_heads=3 --num_layers_1=3 --num_layers_2=3 --sweep=1 --training_set_path=../SL_ConnectingPoints/split/DGI305-AEC--38--incremental--mediapipe_n_folds_5_seed_95_klod_1-Train.hdf5 --validation_set_path= --weight_decay_dynamic=0 --experiment_name="Gen6Piramidalv1" --draw_points=0 --use_wandb=1 --resume=1

"""
h.shape torch.Size([15, 1, 108])

src.shape torch.Size([13, 1, 108])
tgt.shape torch.Size([1, 1, 108])
self.positional_encoder(src).shape: torch.Size([13, 1, 108])
self.positional_encoder(src): tensor([[0.5020, 1.3591, 0.5232, 1.3152]], device='cuda:0')
self.positional_encoder(src): tensor([[0.5031, 1.3589, 0.5289, 1.3182]], device='cuda:0')
self.positional_encoder(src): tensor([[0.5034, 1.3598, 0.5296, 1.3182]], device='cuda:0')
memory.shape torch.Size([13, 1, 108])
memory: [[-1.7996032238006592, 1.9307384490966797, 0.34947311878204346, -0.11882045865058899]]
generation.shape torch.Size([1, 1, 108])
generation: [[0.26965904235839844, 0.5806722044944763, 0.2910144627094269, 0.8225837349891663]]
embedding.shape torch.Size([1, 108])
res.shape torch.Size([1, 38])


inputs.shape torch.Size([15, 54, 2])
h.shape torch.Size([15, 108])

src.shape torch.Size([13, 108])
tgt.shape torch.Size([1, 108])
self.positional_encoder(src).shape: torch.Size([13, 108])
self.positional_encoder(src): tensor([0.5020, 0.3591, 0.5232, 0.3152], device='cuda:0') tensor([0.5020, 1.3591, 0.5232, 1.3152], device='cuda:0')
self.positional_encoder(src): tensor([0.5031, 0.3589, 0.5289, 0.3182], device='cuda:0') tensor([1.3445, 0.8992, 1.2756, 0.9833], device='cuda:0')
self.positional_encoder(src): tensor([0.5034, 0.3598, 0.5296, 0.3182], device='cuda:0') tensor([ 1.4127, -0.0564,  1.5229,  0.2028], device='cuda:0')
memory.shape torch.Size([13, 108])
memory: [-2.5679965019226074, 3.177328586578369, 0.5847898125648499, -1.8924801349639893]
embedding.shape torch.Size([1, 108])
res.shape torch.Size([1, 38])

"""
#datos

