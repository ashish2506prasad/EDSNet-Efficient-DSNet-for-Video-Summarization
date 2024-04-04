import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from helpers import init_helper, data_helper
from anchor_based import anchor_helper
from anchor_based.dsnet import DSNet, DSNet_DeepAttention, DSNet_MultiAttention
from evaluate import evaluate
from helpers import data_helper, vsumm_helper, bbox_helper
from torchinfo import summary

from anchor_free.dsnet_af import DSNetAF, DSNetAF_DeepAttention
logger = logging.getLogger()


def xavier_init(module):
    cls_name = module.__class__.__name__
    if 'Linear' in cls_name or 'Conv' in cls_name:
        nn.init.xavier_uniform_(module.weight, gain=np.sqrt(2.0))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.1)

def tuple_dimensions(t):
    if not isinstance(t, tuple):
        return ()
    return (len(t),) + tuple_dimensions(t[0])

def model_summary(model):
    if model == 'anchor-based':
        print("printing model summary (anchor based): ")
        if args.model_depth == 'shallow':
            print("printing model summary (shallow): ")
            summary(DSNet(base_model=args.base_model, num_feature=args.num_feature,
                    num_hidden=args.num_hidden, anchor_scales=args.anchor_scales,
                    num_head=args.num_head))
        elif args.model_depth == 'deep':
            print("printing model summary (deep attention): ")
            summary(DSNet_DeepAttention(base_model=args.base_model, num_feature=args.num_feature,
                    num_hidden=args.num_hidden, anchor_scales=args.anchor_scales,
                    num_head=args.num_head))
        elif args.model_depth == 'local-global-attention':
            print("printing model summary (local-global-attention): ")
            summary(DSNet_MultiAttention(base_model=args.base_model, num_feature=args.num_feature,
                    num_hidden=args.num_hidden, anchor_scales=args.anchor_scales,
                    num_head=args.num_head))
        else:
            raise ValueError(f'Invalid model type: {model}')

    
    elif model == 'anchor-free':
        print("printing model summary (anchor free): ")
        if args.model_depth == 'shallow':
            print("printing model summary (shallow): ")
            summary(DSNetAF(base_model=args.base_model, num_feature=args.num_feature,
                    num_hidden=args.num_hidden, num_head=args.num_head))
        elif args.model_depth == 'deep':
            print("printing model summary (deep attention): ")
            summary(DSNetAF_DeepAttention(base_model=args.base_model, num_feature=args.num_feature,
                    num_hidden=args.num_hidden, num_head=args.num_head))
        else:
            raise ValueError(f'Invalid model type: {model}')
    else:
        raise ValueError(f'Invalid model type: {model}')
    
    return


if __name__ == '__main__':
    args = init_helper.get_arguments()
    split = args.splits
    model = args.model
    model_summary(model)

    # input_dimensions(split)


    
   


    