# -*- coding: utf-8 -*-
import torch
from torch import nn
import math

padToken = 256 #256, 17, 50
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, ntoken, d_model, nlayers=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        nhead = d_model // 64

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.decoder = nn.Embedding(ntoken, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        #self.inscale = math.sqrt(intoken)
        #self.outscale = math.sqrt(outtoken)

        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=nlayers,
                                          num_decoder_layers=nlayers, dim_feedforward=d_model, dropout=dropout)
        self.fc_out = nn.Linear(d_model, ntoken)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        #return (inp == 0).transpose(0, 1)
        return (inp == padToken).transpose(0, 1)

    def forward(self, src, trg):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        #src_pad_mask = self.make_len_mask(src)
        trg_pad_mask = self.make_len_mask(trg)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        #trg = trg.squeeze(2)
        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, tgt_mask=self.trg_mask, tgt_key_padding_mask=trg_pad_mask)
        # output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask,
        #                           memory_mask=self.memory_mask,
        #                           src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask,
        #                           memory_key_padding_mask=src_pad_mask)
        output = self.fc_out(output)

        return output