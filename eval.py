# -*- coding: utf-8 -*-

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from sumeval.metrics.rouge import RougeCalculator
from tqdm import tqdm

from helper.processUtils import PreVocab
from helper.dataset import Dataset
from seq2seq.generator import Generator
from helper.config import init_config


def evaluate(logger, args, gen, vocab, data_iter, train_data, example_idx, train_next=True):
    gen.eval()
    sums = 0
    loss, loss_rate, r1, r2, rl = .0, .0, .0, .0, .0
    rouge = RougeCalculator(stopwords=False, lang="en")
    with torch.no_grad():
        for batch in tqdm(data_iter):
            if len(batch['userID']) < args.batch_size:
                break
            src, src_user, src_product, rating, u_product, u_review,  p_review, p_user, trg,\
                src_embed, trg_embed, src_mask, src_lens, trg_lens, src_text,  trg_text,\
                specific_vocab = vocab.read_batch(batch, train_data)
            function = 'batchNLLLoss'
            summary_loss, pre_output = \
                gen(function, src, src_user, src_product, u_product, u_review, p_review, p_user,
                    src_embed, trg_embed, vocab.word_num, src_lens, trg_lens, trg,
                    rating, specific_vocab, test=True)
            loss_data = summary_loss
            loss += loss_data.mean().data.item()

            pre_output[:, :, 3] = float('-inf')
            rst = torch.argmax(pre_output, dim=-1).tolist()
            for i, summary in enumerate(rst):
                cur_sum = ['']
                for idx in summary:
                    if idx == vocab.EOS_IDX:
                        break
                    w = vocab.id_word(idx)
                    cur_sum.append(w)
                cur_sum = ' '.join(cur_sum).strip()
                if len(cur_sum) == 0:
                    cur_sum = '<EMP>'
                logger.info(src_text[i])
                logger.info(trg_text[i])
                logger.info(cur_sum)
                logger.info('#' * 40)
                # sums.append(cur_sum)
                sums += 1
                r1 += rouge.rouge_n(cur_sum, trg_text[i], n=1)
                r2 += rouge.rouge_n(cur_sum, trg_text[i], n=2)
                rl += rouge.rouge_l(cur_sum, trg_text[i])
    loss /= len(data_iter)
    loss_rate /= len(data_iter)
    r1 /= sums
    r2 /= sums
    rl /= sums
    if train_next:
        gen.train()
    return loss, loss_rate, r1, r2, rl

