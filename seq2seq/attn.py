# -*- coding: utf-8 -*-
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Review_Attention(nn.Module):

    def __init__(self,  hidden_size):
        super(Review_Attention, self).__init__()

        self.query_proj = nn.Linear(hidden_size, 1, bias=False)
        self.review_proj = nn.Linear(2 * hidden_size, 1, bias=False)

    def forward(self, query, reviews, batch_size, mem_size):
        enc_query = self.query_proj(query)
        enc_reviews = self.review_proj(reviews)
        key_score = F.softmax((enc_query + enc_reviews).view(batch_size, mem_size, -1), dim=-1)
        output = torch.bmm(key_score.view(batch_size, 1, mem_size), reviews).view(batch_size, -1)
        return output


class Attention(nn.Module):

    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(Attention, self).__init__()
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        # additive attention components, score(hi, hj) = v * tanh(W1 * hi + W2 + hj)
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        # to store attention scores
        self.alphas = None
        self.coverage_proj = nn.Linear(1, hidden_size)

    def forward(self, query=None, proj_key=None, value=None, mask=None, coverage=None):
        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        batch_size, max_len = value.size(0), value.size(1)
        query = self.query_layer(query)
        if coverage is not None:
            coverage_input = coverage.view(-1, 1)
            coverage_feature = self.coverage_proj(coverage_input).view(batch_size, max_len, -1)
            scores = self.energy_layer(torch.tanh(query + proj_key + coverage_feature))
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores.data.masked_fill_(mask == 0, -float('inf'))

        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas  # [B, 1, max_len]

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)  # [B, 1, 2*H]

        if coverage is None:
            return context, alphas
        else:
            coverage = coverage + alphas.squeeze()
            return context, alphas, coverage
