# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb

from seq2seq.attn import Attention


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args

        # Encoder
        self.encoder_rnn = nn.GRU(args.embed_dim, args.hidden_size, args.num_layers,
                                  batch_first=True, bidirectional=True, dropout=args.encoder_dropout)

        self.dropout_layer = nn.Dropout(p=args.decoder_dropout)

        self.init_hidden = nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.input_proj = nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.gate_proj = nn.Linear(2 * args.hidden_size + args.embed_dim, 2 * args.hidden_size)

    def forward(self, src_embed, src_lengths, u_review, u_review_lens, user_embed):

        # feed input to encoder RNN
        packed = pack_padded_sequence(src_embed, src_lengths, batch_first=True, enforce_sorted=False)
        encoder_hidden, encoder_final = self.encoder_rnn(packed)
        encoder_hidden, _ = pad_packed_sequence(encoder_hidden, batch_first=True)  # encoder_hidden: [B, S, 2H]

        packed = pack_padded_sequence(u_review, u_review_lens, batch_first=True, enforce_sorted=False)
        _, review_final = self.encoder_rnn(packed)
        u_review_final = torch.cat([review_final[0:review_final.size(0):2], review_final[1:review_final.size(0):2]], dim=2)[-1]
        u_review_final = u_review_final.view(self.args.batch_size, self.args.mem_size, -1)

        # user_embed = u_review_final.mean(1)

        gate = torch.sigmoid(self.gate_proj(torch.cat(
            [encoder_hidden, user_embed.unsqueeze(1).repeat(1, encoder_hidden.size(1), 1)], dim=-1)))
        encoder_hidden = encoder_hidden.mul(gate)

        # get encoder final state, will be used as decoder initial state
        # fwd_final = encoder_final[0:encoder_final.size(0):2]
        # bwd_final = encoder_final[1:encoder_final.size(0):2]
        # encoder_final = torch.cat([fwd_final, bwd_final], dim=2)  # encoder_final: [num_layers, B, 2H]

        # gate_h = torch.sigmoid(self.gate_proj(torch.cat([encoder_final, user_embed.unsqueeze(0).repeat(2, 1, 1)], dim=-1)))
        # encoder_final = encoder_final.mul(gate_h)
        encoder_final = self.input_proj(encoder_hidden[:,-1,:])
        # hidden = torch.tanh(self.init_hidden(encoder_final))
        hidden = encoder_final.unsqueeze(0)
        context_hidden = encoder_final.unsqueeze(1)

        return hidden, encoder_hidden, context_hidden
