# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

import pdb

from seq2seq.enc import Encoder
from seq2seq.attn import Attention, Review_Attention


class Generator(nn.Module):

    def __init__(self, args, embed):
        super(Generator, self).__init__()

        self.args = args
        # Embedding layer
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        # User Embeding
        self.u_emb = nn.Embedding(args.user_num, args.attr_dim)
        self.p_emb = nn.Embedding(args.product_num, args.attr_dim)

        # define encoder and decoder
        self.encoder = Encoder(args)

        # attention in decoder
        self.attention = Attention(args.hidden_size)
        self.attention_u = Attention(args.hidden_size, args.embed_dim)

        # Decoder
        self.decoder_rnn = nn.GRU(args.embed_dim, args.hidden_size, args.num_layers,
                                  batch_first=True, dropout=args.decoder_dropout)
        # init decoder hidden from encoder final hidden
        self.review_attention = Review_Attention(args.hidden_size)
        self.history_proj = nn.Linear(4 * args.hidden_size, args.hidden_size)

        self.dropout_layer = nn.Dropout(p=args.decoder_dropout)
        # mix hidden and context into a context_hidden vector
        self.context_hidden = nn.Linear(3 * args.hidden_size + 2 * args.embed_dim,
                                        args.hidden_size, bias=False)
        # generate mode probability layer
        self.gen_p = nn.Linear(3 * args.hidden_size + args.embed_dim, 1)
        # generate mode layer, context_hidden => word distribution over fixed vocab, P(changeable vocab) = 0
        self.generator = nn.Linear(args.hidden_size, args.embed_num, bias=False)
        # loss for text generation and rating predict
        self.criterion = nn.NLLLoss(ignore_index=0, reduction='sum')
        self.rate_criterion = nn.MSELoss(reduce=True, size_average=True)

    def forward(self, function, src, src_user, src_product, u_product, u_review, p_review, p_user, src_, tgt_,
                vocab_size, src_lens, tgt_lengths, tgt, rating, specific_vocab,
                vocab=None, reward=None, test=False):
        if function == 'batchNLLLoss':
            loss_t, pre_output_vectors = \
                self.batchNLLLoss(src, src_user, src_product, u_product, u_review, p_review,
                                  p_user, src_, tgt_, vocab_size, src_lens, tgt_lengths, tgt,
                                  rating, specific_vocab, test)
            return loss_t, pre_output_vectors
        if function == 'sample':
            summary = self.sample(vocab, src, src_user, src_product, u_product, u_review, p_review,
                                  p_user, src_, vocab_size, src_lens, rating, specific_vocab, test=False)
            return summary
        if function == 'batchPGLoss':
            if reward is not None:
                loss = self.batchPGLoss(src, src_user, src_product, u_product, u_review, p_review,
                                        p_user, src_, tgt_, vocab_size, src_lens, tgt_lengths,
                                        rating, specific_vocab, reward, test=False)
                return loss
            else:
                print('Reward can not be None in PGLoss')

    def step(self, src, prev_embed, encoder_hidden, src_mask, hidden, context_hidden,
             user_vocab_emb, user_embed, specific_vocab, vocab_size):
        """Perform a single decoder step (1 word)"""

        # update rnn hidden state
        # rnn_input = torch.cat([prev_embed, context_hidden], dim=-1)
        rnn_input = prev_embed
        output, hidden = self.decoder_rnn(rnn_input, hidden)

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [B, 1, H]

        # alignment with input review
        proj_key = self.attention.key_layer(encoder_hidden)
        context_c, _ = self.attention(
            query=query, proj_key=proj_key, value=encoder_hidden, mask=src_mask)
        context_u, attn_probs = self.attention_u(
            query=query, proj_key=self.attention_u.key_layer(user_vocab_emb), value=user_vocab_emb)

        # generate mode word distribution
        context_hidden = torch.tanh(self.context_hidden(
            torch.cat([query, context_c, context_u, user_embed.unsqueeze(1)], dim=2)))
        context_hidden_1 = self.dropout_layer(context_hidden)
        gen_prob = F.softmax(self.generator(context_hidden_1), dim=-1)

        if vocab_size > gen_prob.size(2):
            gen_prob = torch.cat([gen_prob, torch.zeros(gen_prob.size(0), gen_prob.size(1),
                                                        vocab_size - gen_prob.size(2)).cuda()], dim=-1)

        # copy mode word distribution
        specific_vocab = specific_vocab.unsqueeze(1)
        copy_prob = torch.zeros(specific_vocab.size(0), specific_vocab.size(1),
                                vocab_size).cuda().scatter_add(2, specific_vocab, attn_probs)

        # generate probability p
        copy_p = torch.sigmoid(self.gen_p(torch.cat([context_c, query, context_u], -1)))
        mix_prob = copy_p * copy_prob + (1 - copy_p) * gen_prob
        return hidden, context_hidden_1, mix_prob, attn_probs

    def batchNLLLoss(self, src, src_user, src_product, u_product, u_review, p_review, p_user, src_, tgt_,
                     vocab_size, src_lens, tgt_lengths, tgt, rating, specific_vocab, test=False):
        """
            Returns the NLL Loss for predicting target sequence.

            Inputs: inp, target
                - inp: batch_size x seq_len
                - target: batch_size x seq_len

                inp should be target with <s> (start letter) prepended
        """
        max_len = self.args.sum_max_len
        pre_output_vectors = []

        src = src[:, :src_lens[0].data]
        src_mask = torch.sign(src)

        u_review_lens = torch.sum(torch.sign(u_review), dim=-1).data
        # encode soure review text

        src_embed = self.embed(src_)  # x: [B, S, D]
        u_review = self.embed(u_review)  # x: [B, S, D]

        user_embed = self.u_emb(src_user)
        # user_embed = self.u_emb(src_user)
        hidden, encoder_hidden, context_hidden = self.encoder(src_embed, src_lens, u_review, u_review_lens, user_embed)

        tgt_embed = self.embed(tgt_)
        user_vocab_emb = self.embed(specific_vocab)
        init_prev_embed = self.embed(torch.LongTensor([1]).cuda()).repeat(len(src), 1).unsqueeze(1)

        for i in range(max_len):
            if i == 0:  # <SOS> embedding
                prev_embed = init_prev_embed
            else:
                if not test:  # last trg word embedding
                    prev_embed = tgt_embed[:, i - 1].unsqueeze(1)
                else:  # last predicted word embedding
                    prev_idx = torch.argmax(pre_output_vectors[-1], dim=-1)
                    for j in range(0, prev_idx.size(0)):
                        if prev_idx[j][0] >= self.args.embed_num:
                            prev_idx[j][0] = 3  # UNK_IDX
                    prev_embed = self.embed(prev_idx)
            # step
            hidden, context_hidden, word_prob, attn_probs =\
                self.step(src, prev_embed, encoder_hidden, src_mask, hidden,
                          context_hidden, user_vocab_emb, user_embed, specific_vocab, vocab_size)
            pre_output_vectors.append(word_prob)

        # calculate nll loss
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        pre_output = torch.log(pre_output_vectors.view(-1, pre_output_vectors.size(-1)) + 1e-20)
        trg_output = tgt.view(-1)
        loss_t = self.criterion(pre_output, trg_output) / len(src_lens)

        # rating = rating.view(-1)
        # loss_r = self.rate_criterion(rating_predict, rating.float())


        if self.args.repetition:
            # return loss_t, loss_r, pre_output_vectors, rating_predict, coverage_output_loss
            # return loss_t, pre_output_vectors, coverage_output_loss
            return loss_t, pre_output_vectors
        else:
            return loss_t, pre_output_vectors
