# -*- coding: utf-8 -*-
import os
import json
from tqdm import tqdm

import torch
from torch.autograd import Variable

from helper.processUtils import PreVocab


def collate_fn(batch):
    return zip(batch)


def adjust_learning_rate(args, optimizer, index):
    lr = args.lr * (args.lr_decay ** index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_data(logger, args):
    embed = None
    if args.embed_path is not None and os.path.exists(args.embed_path):
        logger.info('Loading pretrained word embedding...')
        embed = {}
        with open(args.embed_path, 'r') as f:
            f.readline()
            for line in f.readlines():
                line = line.strip().split()
                vec = [float(_) for _ in line[1:]]
                embed[line[0]] = vec
    vocab = PreVocab(args, embed)
    logger.info('Loading datasets...')
    train_data, val_data, test_data = [], [], []
    fns = os.listdir(args.train_dir)
    fns.sort(key=lambda p: int(p.split('.')[0]))
    for fn in tqdm(fns):
        f = open(args.train_dir + fn, 'r')
        train_data.append(json.load(f))
        f.close()
        vocab.add_sentence(train_data[-1]['reviewText'].split())
        vocab.add_sentence(train_data[-1]['summary'].split())
        vocab.add_user(train_data[-1]['userID'])
        vocab.add_product(train_data[-1]['productID'])

    fns = os.listdir(args.valid_dir)
    fns.sort(key=lambda p: int(p.split('.')[0]))
    count = 0
    for fn in tqdm(fns):
        if count >= 1000:
            break
        count += 1
        f = open(args.valid_dir + fn, 'r')
        val_data.append(json.load(f))
        f.close()
        vocab.add_sentence(val_data[-1]['reviewText'].split())
        vocab.add_sentence(val_data[-1]['summary'].split())
        vocab.add_user(val_data[-1]['userID'])
        vocab.add_product(val_data[-1]['productID'])

    fns = os.listdir(args.test_dir)
    fns.sort(key=lambda p: int(p.split('.')[0]))
    cnt = 0
    for fn in tqdm(fns):
        if cnt >= 1000:
            break
        cnt += 1
        f = open(args.test_dir + fn, 'r')
        test_data.append(json.load(f))
        f.close()
        vocab.add_sentence(test_data[-1]['reviewText'].split())
        vocab.add_sentence(test_data[-1]['summary'].split())
        vocab.add_user(test_data[-1]['userID'])
        vocab.add_product(test_data[-1]['productID'])

    logger.info('Deleting rare words...')
    embed = vocab.trim(logger)
    return embed, vocab, train_data, val_data, test_data


def prepare_discriminator_data(pos_samples, pos_sample_lens, neg_samples, neg_sample_lens, gpu=True):
    """
    Takes positive (target) samples, negative (generator) samples and prepares inp and target data for discriminator.

    Inputs: pos_samples, neg_samples
        - pos_samples: pos_size x seq_len
        - neg_samples: neg_size x seq_len

    Returns: inp, target
        - inp: (pos_size + neg_size) x seq_len
        - target: pos_size + neg_size (boolean 1/0)
    """

    inp_len = torch.cat((pos_sample_lens, neg_sample_lens), 0)
    inp = torch.cat((pos_samples, neg_samples), 0).type(torch.LongTensor)
    target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
    target[pos_samples.size()[0]:] = 0

    # shuffle
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]
    inp_len = inp_len[perm]

    inp = Variable(inp)
    target = Variable(target)
    inp_len = Variable(inp_len)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()
        inp_len = inp_len.cuda()

    return inp, inp_len, target
