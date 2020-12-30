# -*- coding: utf-8 -*-

from math import ceil
import random
import pdb
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from helper.utils import load_data, adjust_learning_rate, prepare_discriminator_data
from eval import evaluate


def train_generator_MLE(logger, args, gen, gen_optim, epochs, vocab, train_data, example_idx,
                        train_iter, val_iter):
    """
    Max Likelihood Pretraining for the generator
    """

    logger.info('Begin training...')
    for epoch in range(1, epochs + 1):
        if epoch >= args.lr_decay_start:
            adjust_learning_rate(args, gen_optim, epoch - args.lr_decay_start + 1)

        for i, batch in enumerate(train_iter):
            # if len(batch['userID']) < args.batch_size:
            #     continue
            src, src_user, src_product, rating, u_product, u_review, p_review, p_user,\
                trg, src_embed, trg_embed, src_mask, src_lens, trg_lens, _1, _2, specific_vocab = \
                vocab.read_batch(batch, train_data)
            function = 'batchNLLLoss'

            gen_optim.zero_grad()
            if args.repetition:
                summary_loss, rating_loss, _1, _2, cov_loss = \
                    gen(function, src,  src_user, src_product, u_product, u_review, p_review, p_user,
                        src_embed, trg_embed, vocab.word_num, src_lens, trg_lens, trg,
                        rating, specific_vocab)
                loss = summary_loss + args.alpha * rating_loss + args.beta * cov_loss
            else:
                summary_loss, _ = \
                    gen(function, src,  src_user, src_product, u_product, u_review, p_review, p_user,
                        src_embed, trg_embed, vocab.word_num, src_lens, trg_lens, trg,
                        rating, specific_vocab)
                loss = summary_loss

            loss.mean().backward()

            clip_grad_norm_(gen.parameters(), args.max_norm)
            gen_optim.step()

            cnt = (epoch - 1) * len(train_iter) + i
            if cnt % args.print_every == 0:
                logger.info('EPOCH [%d/%d]: BATCH_ID=[%d/%d] loss=%f' % (
                    epoch, args.MLE_TRAIN_EPOCHS, i, len(train_iter), loss.mean().data.item()))
            if cnt % args.valid_every == 0:
                valid_generator(logger, args, gen, vocab, val_iter, train_data, example_idx, epoch)
    return


def valid_generator(logger, args, gen, vocab, val_iter, train_data, example_idx, epoch):
    logger.info('Begin valid... Epoch %d' % (epoch))
    cur_loss, loss_rate, r1, r2, rl = evaluate(logger, args, gen, vocab,
                                               val_iter, train_data, example_idx, True)
    save_path = args.save_path + 'valid_%.4f_%.4f_%.4f_%.4f_%.4f' % (cur_loss, loss_rate, r1, r2, rl)
    with open(save_path, 'w') as check_file:
        check_file.write('valid_%.4f_%.4f_%.4f_%.4f_%.4f' % (cur_loss, loss_rate, r1, r2, rl) + '\n')
    # net.save(save_path)
    logger.info('Epoch: %2d Cur_Val_Loss: %f Cur_RMSE_Loss: %f Rouge-1: %f Rouge-2: %f Rouge-l: %f' % (
                    epoch, cur_loss, loss_rate, r1, r2, rl))
