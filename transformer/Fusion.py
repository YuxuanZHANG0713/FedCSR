import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformer.Activation import Swish
from transformer.Attention import MultiHeadAttentionFusion, VanillaMultiHeadAttention
from transformer.Mask import get_attn_pad_mask

from transformer.Modules import Linear
from transformer.SubLayers import PositionwiseFeedForward


class Fusion(torch.nn.Module):
    def __init__(self, din, fusion_dim):
        super(Fusion,self).__init__()
        
        self.linear1 = Linear(din, fusion_dim)  
        self.bn1 = torch.nn.BatchNorm1d(fusion_dim, momentum=0.01, eps=0.001)

    def forward(self, lipBatch, handBatch):   
        jointBatch = torch.cat((lipBatch, handBatch), dim=2)
        outputBatch = self.linear1(jointBatch)
        outputBatch = outputBatch.transpose(1,2)
        outputBatch = self.bn1(outputBatch)
        outputBatch = outputBatch.transpose(1,2)
        outputBatch = F.relu(outputBatch)

        
        return outputBatch



class FusionLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, ffd_expansion_factor, n_head, d_k, d_v, attn_dropout=0.1, ffd_dropout=0.1):
        super(FusionLayer, self).__init__()

        self.slf_attn = MultiHeadAttentionFusion(n_head, d_model, d_k, d_v, dropout=attn_dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, ffd_expansion_factor*d_model, dropout=ffd_dropout)

    def forward(self, enc_input, enc_input_f, slf_attn_mask=None):
        dec_output, dec_enc_attn = self.slf_attn(enc_input, enc_input_f, enc_input, enc_input_f, mask=slf_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output


class AttnFusion(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_head, 
            d_k, 
            d_v, 
            d_model, 
            ffd_expansion_factor, 
            dec_dropout=0.1, 
            attn_dropout=0.1, 
            ffd_dropout=0.1):

        super().__init__()

        self.d_model = d_model

        self.dropout = nn.Dropout(p=dec_dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.layer = FusionLayer(d_model, ffd_expansion_factor, n_head, d_k, d_v, attn_dropout=attn_dropout, ffd_dropout=ffd_dropout) 
           
    def forward(self, enc_output, enc_output_f, src_mask):

        sub_length = enc_output.size(1) 

        slf_attn_mask = get_attn_pad_mask(src_mask, sub_length)

        enc_output = self.dropout(enc_output)
        enc_output_f = self.dropout(enc_output_f)
        # dec_output = self.layer_norm(dec_output)

        fusion_out = self.layer(enc_output, enc_output_f, slf_attn_mask=slf_attn_mask)
        
        return fusion_out