# -*- coding: utf-8 -*-

import os
import argparse
import random
import torch
import numpy as np


def init_config():
    parser = argparse.ArgumentParser(description='usn')
# path info
    parser.add_argument('-save_path', type=str, default='tmp/toys-checkpoints/')
    parser.add_argument('-embed_path', type=str, default='../../embedding/glove/glove.aligned.txt')
    parser.add_argument('-train_dir', type=str, default='../../data/toys_data/train/')
    parser.add_argument('-valid_dir', type=str, default='../../data/toys_data/test/')
    parser.add_argument('-test_dir', type=str, default='../../data/toys_data/valid/')
    parser.add_argument('-load_model', type=str, default='')
    parser.add_argument('-output_dir', type=str, default='tmp/output/')
    parser.add_argument('-log_file', type=str, default='tmp/log')
    parser.add_argument('-example_num', type=int, default=4)
    # hyper paras

    parser.add_argument('gpu_id', nargs='+', type=int, help='if use gpu, list the id of gpu device')

    parser.add_argument('-embed_dim', type=int, default=300)
    parser.add_argument('-embed_num', type=int, default=0)
    parser.add_argument('-word_min_cnt', type=int, default=30)
    parser.add_argument('-beam_size', type=int, default=1)

    parser.add_argument('-MLE_TRAIN_EPOCHS', type=int, default=40)

    parser.add_argument('-attr_dim', type=int, default=300)
    parser.add_argument('-user_num', type=int, default=0)
    parser.add_argument('-product_num', type=int, default=0)

    parser.add_argument('-use_cuda', type=bool, default=True)
    parser.add_argument('-repetition', type=bool, default=False)
    parser.add_argument('-test', action='store_true')

    parser.add_argument('-sum_max_len', type=int, default=15)
    parser.add_argument('-hidden_size', type=int, default=512)
    parser.add_argument('-num_layers', type=int, default=1)
    parser.add_argument('-encoder_dropout', type=float, default=0.2)
    parser.add_argument('-decoder_dropout', type=float, default=0.2)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-lr_decay', type=float, default=0.5)
    parser.add_argument('-lr_decay_start', type=int, default=6)
    parser.add_argument('-max_norm', type=float, default=5.0)
    parser.add_argument('-batch_size', type=int, default=24)
    parser.add_argument('-seed', type=int, default=2333)
    parser.add_argument('-print_every', type=int, default=10)
    parser.add_argument('-valid_every', type=int, default=800)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    return args

