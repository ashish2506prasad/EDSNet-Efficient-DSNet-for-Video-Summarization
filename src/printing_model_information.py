import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from helpers import init_helper, data_helper
from anchor_based import anchor_helper
from anchor_based.dsnet import DSNet, DSNet_DeepAttention
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
        if args.depth == 'shallow':
            print("printing model summary (shallow): ")
            summary(DSNet(base_model=args.base_model, num_feature=args.num_feature,
                    num_hidden=args.num_hidden, anchor_scales=args.anchor_scales,
                    num_head=args.num_head))
        else:
            print("printing model summary (deep attention): ")
            summary(DSNet_DeepAttention(base_model=args.base_model, num_feature=args.num_feature,
                    num_hidden=args.num_hidden, anchor_scales=args.anchor_scales,
                    num_head=args.num_head))
    
    elif model == 'anchor-free':
        print("printing model summary (anchor free): ")
        if args.depth == 'shallow':
            print("printing model summary (shallow): ")
            summary(DSNetAF(base_model=args.base_model, num_feature=args.num_feature,
                    num_hidden=args.num_hidden, num_head=args.num_head))
        else:
            print("printing model summary (deep attention): ")
            summary(DSNetAF_DeepAttention(base_model=args.base_model, num_feature=args.num_feature,
                    num_hidden=args.num_hidden, num_head=args.num_head))
    else:
        raise ValueError(f'Invalid model type: {model}')
    
    return

def input_dimensions(split):

    for split_path in args.splits:
            split_path = Path(split_path)
            splits = data_helper.load_yaml(split_path)


            for split_idx, split in enumerate(splits):
                logger.info(f'Start training on {split_path.stem}: split {split_idx}')

                train_set = data_helper.VideoDataset(split['train_keys'])
                train_loader = data_helper.DataLoader(train_set, shuffle=True)
                ## printing the training loader
                for a, seq, gtscore, cps, n_frames, nfps, picks, b in train_loader:
                    print("printing the train loader: ")
                    # print(a.shape)
                    print(seq.shape)
                    print(gtscore.shape)
                    print(cps.shape)
                    print(n_frames.shape)
                    print(nfps.shape)
                    print(picks.shape)
                    print(b.shape)
                    break
                
        
    return
    




if __name__ == '__main__':
    args = init_helper.get_arguments()
    split = args.splits
    model = args.model
    model_summary(model)

    # input_dimensions(split)


    
   


    