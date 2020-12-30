# -*- coding: utf-8 -*-

import sys

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from helper.config import init_config
from helper.utils import load_data
from helper.dataset import Dataset

from seq2seq.generator import Generator

from helper.logging import init_logger
from trainer import train_generator_MLE


def train(logger, args, example_idx):
    # load data
    embed, vocab, train_data, val_data, test_data = load_data(logger, args)
    args.embed_num = len(embed)
    args.embed_dim = len(embed[0])
    args.user_num = vocab.user_num
    args.product_num = vocab.product_num

    # data iter
    train_dataset = Dataset(train_data)
    val_dataset = Dataset(val_data)
    train_iter = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_iter = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # define generator and discriminator in GAN
    gen = Generator(args, embed)

    # define GPU devices
    if args.use_cuda:
        gen = gen.cuda()
        gen = nn.DataParallel(gen, device_ids=args.gpu_id)

    # GENERATOR MLE TRAINING
    logger.info('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=args.lr)
    train_generator_MLE(logger, args, gen, gen_optimizer, args.MLE_TRAIN_EPOCHS, vocab,
                        train_data, example_idx, train_iter, val_iter)


if __name__ == '__main__':
    args = init_config()
    logger = init_logger(args.log_file)
    example_idx = np.random.choice(range(5000), args.example_num)
    train(logger, args, example_idx)
