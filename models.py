import os
import sys
import yaml
import math
import json
import torch
import torch.nn as nn
from einops import rearrange

import copy
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# res connection + layer norm
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, in_size, cond_len, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.h = h
        self.in_size = in_size
        self.d_k = d_model // h

        self.attn = None
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_len, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, d_model, bias=False),
        )
        
    def forward(self, query, key, value, cond=None):
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # inject conditional info
        if cond is not None:
            cond = self.cond_mlp(cond)
            cond = cond[:, None, None, :].repeat(1, self.h, self.in_size, 1)
            value = value + cond
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# main transformer block
class TransformerLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, cond):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, cond))
        return self.sublayer[1](x, self.feed_forward)

class Transformer(torch.nn.Module):
    def __init__(self, in_size, out_size, d_model, cond_len=64):
        super(Transformer, self).__init__()
        self.in_size = in_size
        self.n_filters = 32
        self.out_size = out_size
        self.d_model = d_model

        self.N = 4
        self.n_head = 1
        self.d_ff = 512
        self.dropout = 0.1

        # initial convolutions
        a = self.n_filters * self.in_size
        self.conv_in = nn.Conv1d(self.in_size, a, kernel_size=5, stride=1, padding=2)

        # make main model
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_head, self.d_model, a, cond_len)
        ff = FeedForward(self.d_model, self.d_ff)

        block = TransformerLayer(self.d_model, c(attn), c(ff), self.dropout)
        self.blocks = clones(block, self.N)
        self.norm = LayerNorm(self.d_model)
        self.linear = nn.Linear(a * self.d_model, self.out_size)

        # init weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, x, cond):
        x = self.conv_in(x)

        for block in self.blocks:
            x = block(x, cond)
        x = self.norm(x)
        x = rearrange(x, 'b f d -> b (f d)')

        x = self.linear(x).squeeze(-1)
        #x = nn.functional.softmax(x, dim=-1)
    
        return x