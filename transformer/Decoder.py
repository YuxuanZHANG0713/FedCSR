import torch.nn as nn
from transformer.Mask import *
from transformer.Modules import Linear, SELayer
from transformer.Position import *
from transformer.Attention import VanillaMultiHeadAttention

from transformer.SubLayers import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, ffd_expansion_factor, n_head, d_k, d_v, attn_dropout=0.1, ffd_dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.slf_attn = VanillaMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=attn_dropout)
        self.enc_attn = VanillaMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=attn_dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, ffd_expansion_factor*d_model, dropout=ffd_dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask) 
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output



class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, 
            d_word_vec,
            n_trg_vocab, 
            n_layers, 
            n_head, 
            d_k, 
            d_v, 
            d_model, 
            ffd_expansion_factor, 
            n_position = 200,
            dec_dropout=0.1, 
            attn_dropout=0.1, 
            ffd_dropout=0.1, 
            scale_emb=False):

        super().__init__()

        self.n_trg_vocab = n_trg_vocab

        self.scale_emb = scale_emb
        self.d_model = d_model

        self.inp_se = SELayer(d_model=d_model)

        # self.projection = Linear(2*d_model, d_model)
        # print(n_position)
        
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dec_dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.layer_stack =  nn.ModuleList([
            DecoderLayer(d_model, ffd_expansion_factor, n_head, d_k, d_v, attn_dropout=attn_dropout, ffd_dropout=ffd_dropout) 
            for _ in range(n_layers) ])

        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

    def forward(self, trg_seq, enc_output, src_mask=None):

        # trg_seq = self.projection(trg_seq)
        # dec_enc_attn_mask = get_attn_pad_mask(src_mask, trg_seq.size(1))

        dec_output = self.dropout(self.position_enc(trg_seq))
        dec_output = self.layer_norm(dec_output)

        for layer in self.layer_stack:
            dec_output = self.inp_se(dec_output)
            dec_output = layer(dec_output, enc_output, slf_attn_mask=src_mask, dec_enc_attn_mask=src_mask)


        # print(dec_output)
        return dec_output