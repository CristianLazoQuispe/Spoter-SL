from .models.spoter_model import SPOTER
from .models.spoter_model1 import SPOTER1
from .models.spoter_model2 import SPOTER2
from .models.spoter_model3 import SPOTER3
from .models.spoter_model4 import SPOTER4


def get_slrt_model(args):

    ############################### MODELOS BASE_ATTN ##################################################

    if args.model_name =="base_freeze_attn":
        print("USING SPOTER base + attn + freeze")
        slrt_model = SPOTER1(num_classes=args.num_classes, num_rows=args.num_rows,
                            hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                            num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                            dim_feedforward_encoder=args.dim_feedforward_encoder,
                            dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout,
                            norm_first = False,
                            freeze_decoder_layers = True)
        args.norm_first = 0
        args.freeze_decoder_layers = 1
        args.has_mlp = None

    elif args.model_name == "base_attn":
        print("USING SPOTER base + attn")
        slrt_model = SPOTER1(num_classes=args.num_classes, num_rows=args.num_rows,
                            hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                            num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                            dim_feedforward_encoder=args.dim_feedforward_encoder,
                            dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout,
                            norm_first = False,
                            freeze_decoder_layers = False)
        args.norm_first = 0
        args.freeze_decoder_layers = 0
        args.has_mlp = None

    ############################### MODELOS ENCODER ##################################################

    elif args.model_name == "encoder":
        print("USING SPOTER encoder")
        slrt_model = SPOTER2(num_classes=args.num_classes, num_rows=args.num_rows,
                            hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                            num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                            dim_feedforward_encoder=args.dim_feedforward_encoder,
                            dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout,
                            norm_first = False,
                            has_mlp=False)
        args.norm_first = 0
        args.freeze_decoder_layers = None
        args.has_mlp = 0

    elif args.model_name == "encoder_mlp":
        print("USING SPOTER encoder+mlp")
        slrt_model = SPOTER2(num_classes=args.num_classes, num_rows=args.num_rows,
                            hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                            num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                            dim_feedforward_encoder=args.dim_feedforward_encoder,
                            dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout,
                            norm_first = False,
                            has_mlp=True)
        args.norm_first = 0
        args.freeze_decoder_layers = None
        args.has_mlp = 1

    ############################### MODELOS Encoder with ResiDual Connections ##################################################
    #https://arxiv.org/abs/2304.14802
    elif args.model_name == "encoder_residual":
        print("USING SPOTER Encoder+ResiDual")
        slrt_model = SPOTER3(num_classes=args.num_classes, num_rows=args.num_rows,
                            hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                            num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                            dim_feedforward_encoder=args.dim_feedforward_encoder,
                            dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout,
                            has_mlp=False)
        args.norm_first = None
        args.freeze_decoder_layers =None
        args.has_mlp = 0

    elif args.model_name == "encoder_residual_mlp":
        print("USING SPOTER Encoder+ResiDual+mlp")
        slrt_model = SPOTER3(num_classes=args.num_classes, num_rows=args.num_rows,
                            hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                            num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                            dim_feedforward_encoder=args.dim_feedforward_encoder,
                            dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout,
                            has_mlp=True)
        args.norm_first = None
        args.freeze_decoder_layers = None
        args.has_mlp = 1

    ############################### MODELOS Encoder Decoder with ResiDual Connections ##################################################

    elif args.model_name == "transformer_residual":
        print("USING SPOTER   Transformer Residual : encoder ResiDual + decoder ResiDual")
        slrt_model = SPOTER4(num_classes=args.num_classes, num_rows=args.num_rows,
                            hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                            num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                            dim_feedforward_encoder=args.dim_feedforward_encoder,
                            dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout
                            )
        args.norm_first = None
        args.freeze_decoder_layers = None
        args.has_mlp = None

    ############################### MODELOS BASE ##################################################
    elif  args.model_name== "base_freeze":
        slrt_model = SPOTER(num_classes=args.num_classes, num_rows=args.num_rows,
                            hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                            num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                            dim_feedforward_encoder=args.dim_feedforward_encoder,
                            dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout,
                            norm_first=False,freeze_decoder_layers=True)
        args.norm_first = 0
        args.freeze_decoder_layers = 1
        args.has_mlp = None
    else:
        slrt_model = SPOTER(num_classes=args.num_classes, num_rows=args.num_rows,
                            hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
                            num_layers_1=args.num_layers_1, num_layers_2=args.num_layers_2, 
                            dim_feedforward_encoder=args.dim_feedforward_encoder,
                            dim_feedforward_decoder=args.dim_feedforward_decoder,dropout=args.dropout,
                            norm_first=False,freeze_decoder_layers=False)
        args.norm_first = 0
        args.freeze_decoder_layers = 0
        args.has_mlp = None
        args.model_name = "base"

    return slrt_model,args