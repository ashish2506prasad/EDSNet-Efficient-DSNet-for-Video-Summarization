import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

"""
You might have to change the directories in the following functions to match the directories in your local machine/kaggle
"""


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_logger(log_dir: str, log_file: str) -> None:
    logger = logging.getLogger()
    format_str = r'[%(asctime)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_dir / log_file))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # model type
    parser.add_argument('model', type=str,
                        choices=('anchor-based'))
    parser.add_argument('--model-depth', type=str, default='shallow',
                        choices=['shallow', 'deep', 'local-global-attention', 'original'])
    parser.add_argument('--fft-attention-orientation', type=str, choices=['paper', 'temporal', 'feature_wise'], default='paper')  # orientation as given in fnet paper
    parser.add_argument('--pooling-type', type=str, default = 'roi', choices=['roi', 'flat-pooling', 'fft', 'dwt'])

    # training & evaluation
    parser.add_argument('--device', type=str, default='cuda',
                        choices=('cuda', 'cpu', 'cuda:0', 'cuda:1'))
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--splits', type=str, nargs='+', default=[])
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--model-dir', type=str, default='../models/model')
    parser.add_argument('--log-file', type=str, default='log.txt')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--lambda-reg', type=float, default=1.0)
    parser.add_argument('--nms-thresh', type=float, default=0.5)
    parser.add_argument('--fc-depth', type=int, default=7)
    parser.add_argument('--attention-depth', type=int, default=2)
    parser.add_argument('--encoder-type', type=str, default='classic',
                        choices=['classic', 'local-global'])

    # inference
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--sample-rate', type=int, default=15)
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--save-path', type=str, default=None)

    # common model config
    parser.add_argument('--base-model', type=str, default='attention',
                        choices=['attention', 'lstm', 'linear', 'bilstm',
                                 'gcn', 'nystromformer', 'fourier', 'linformer', 'performer', 'dwt'])
    parser.add_argument('--num-head', type=int, default=8)
    parser.add_argument('--num-feature', type=int, default=1024)
    parser.add_argument('--num-hidden', type=int, default=128)
    
    
    # anchor based
    parser.add_argument('--neg-sample-ratio', type=float, default=2.0)
    parser.add_argument('--incomplete-sample-ratio', type=float, default=1.0)
    parser.add_argument('--pos-iou-thresh', type=float, default=0.6)
    parser.add_argument('--neg-iou-thresh', type=float, default=0.0)
    parser.add_argument('--incomplete-iou-thresh', type=float, default=0.3)
    parser.add_argument('--anchor-scales', type=int, nargs='+',
                        default=[4, 8, 16, 32])

    # anchor free
    parser.add_argument('--lambda-ctr', type=float, default=1.0)
    parser.add_argument('--cls-loss', type=str, default='focal',
                        choices=['focal', 'cross-entropy'])
    parser.add_argument('--reg-loss', type=str, default='soft-iou',
                        choices=['soft-iou', 'smooth-l1'])
    
    parser.add_argument('--where', type=str, choices = ['kaggle', 'local'], default = 'local')  # this repo can be run on kaggle as well, so this argument is added to handle the difference in file paths

    return parser


def get_arguments() -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args()
    return args
