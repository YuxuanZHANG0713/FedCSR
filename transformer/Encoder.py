import torch.nn as nn
import torch
import numpy as np
from transformer.Feed_forward import FeedForwardModule
from transformer.Mask import *
from transformer.Modules import Linear, ResidualConnectionModule, SELayer, SubLayer
from transformer.Position import *
from transformer.SubLayers import PositionwiseFeedForward
from transformer.Attention import MMRelativeMultiHeadedSelfAttention, RelativeMultiHeadedSelfAttention
from transformer.Convolution import ConformerConvModule, Conv2dSubampling



class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, n_head, d_k, d_v, ffd_dropout=0.1, ffd_expansion_factor=4, attn_dropout=0.1, 
        conv_expansion_factor= 2, conv_kernel_size =31, conv_dropout=0.1, half_step_residual=True):
        super(EncoderLayer, self).__init__()

        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1


        self.ffd1 = ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=d_model,
                    expansion_factor=ffd_expansion_factor,
                    dropout_p=ffd_dropout,
                ),
                module_factor=self.feed_forward_residual_factor,
            )

        self.slf_attn = RelativeMultiHeadedSelfAttention(n_head, d_model, d_k, d_v, dropout=attn_dropout)

        self.conv = ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=d_model,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout,
                ),
            )

        self.ffd2 = ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=d_model,
                    expansion_factor=ffd_expansion_factor,
                    dropout_p=ffd_dropout,
                ),
                module_factor=self.feed_forward_residual_factor,
            )

        self.layer_narm = nn.LayerNorm(d_model)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):

        enc_output = self.ffd1(enc_input)
        enc_output, enc_slf_attn = self.slf_attn(enc_output, mask=slf_attn_mask)
        enc_output = self.conv(enc_output)
        enc_output = self.ffd2(enc_output)
        enc_output = self.layer_narm(enc_output)
        

        return enc_output, enc_slf_attn



class MMEncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, n_head, d_k, d_v, ffd_dropout=0.1, ffd_expansion_factor=4, attn_dropout=0.1, 
        conv_expansion_factor= 2, conv_kernel_size =31, conv_dropout=0.1, half_step_residual=True):
        super(MMEncoderLayer, self).__init__()

        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1


        self.ffd1 = ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=d_model,
                    expansion_factor=ffd_expansion_factor,
                    dropout_p=ffd_dropout,
                ),
                module_factor=self.feed_forward_residual_factor,
            )

        self.slf_attn = MMRelativeMultiHeadedSelfAttention(n_head, d_model, d_k, d_v, dropout=attn_dropout)

        self.conv = ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=d_model,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout,
                ),
            )

        self.ffd2 = ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=d_model,
                    expansion_factor=ffd_expansion_factor,
                    dropout_p=ffd_dropout,
                ),
                module_factor=self.feed_forward_residual_factor,
            )

        self.layer_narm = nn.LayerNorm(d_model)

    def forward(self, enc_input_1, enc_input_2, slf_attn_mask=None):

        enc_input_1 = self.ffd1(enc_input_1)
        enc_input_2 = self.ffd1(enc_input_2)
        enc_output_1, enc_slf_attn_1, enc_output_2, enc_slf_attn_2 = self.slf_attn(enc_input_1, enc_input_2, mask=slf_attn_mask)
        enc_output_1 = self.conv(enc_output_1)
        enc_output_1 = self.ffd2(enc_output_1)
        enc_output_1 = self.layer_narm(enc_output_1)

        enc_output_2 = self.conv(enc_output_2)
        enc_output_2 = self.ffd2(enc_output_2)
        enc_output_2 = self.layer_narm(enc_output_2)
        
        return enc_output_1, enc_slf_attn_1, enc_output_2, enc_slf_attn_2


class MMEncoder(nn.Module):
    def __init__(
            self, 
            dataset,
            n_layers, 
            n_head, 
            d_k, 
            d_v,
            d_model,
            ling_model,
            subsampling_dropout = 0.1, 
            ffd_dropout=0.1, 
            ffd_expansion_factor=4, 
            attn_dropout=0.1, 
            conv_expansion_factor= 2, 
            conv_kernel_size =31, 
            conv_dropout=0.1, 
            half_step_residual=True):

        super().__init__()


        self.ling_projection = Linear(d_model+ling_model, ling_model)
        self.spe_projection = Linear(d_model+ling_model, d_model)
        self.mm_layer = nn.ModuleList([
            MMEncoderLayer(d_model+ling_model, n_head, d_k, d_v, 
                        ffd_dropout = ffd_dropout, 
                        ffd_expansion_factor = ffd_expansion_factor, 
                        attn_dropout = attn_dropout, 
                        conv_expansion_factor = conv_expansion_factor, 
                        conv_kernel_size = conv_kernel_size, 
                        conv_dropout = conv_dropout, 
                        half_step_residual=half_step_residual)
            for _ in range(n_layers)])

        self.lip_se = SELayer(d_model=d_model+ling_model)
        self.hand_se = SELayer(d_model=d_model+ling_model)
        
        self.dropout = nn.Dropout(p=subsampling_dropout)
        self.layer_norm = nn.LayerNorm(d_model+ling_model, eps=1e-6)
        self.d_model = d_model
        self.ling_model = ling_model
        self.n_layers = n_layers
        
        self.dataset = dataset

        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)


    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, lip_seq, hand_seq, lip_ling, hand_ling, src_mask, return_attns=False):
       
        lip_enc_slf_attn_list = []
        hand_enc_slf_attn_list = []

        lip_seq = torch.cat((lip_seq, lip_ling), dim=-1)
        hand_seq = torch.cat((hand_seq, hand_ling), dim=-1)

        lip_enc_input = self.dropout(lip_seq)
        hand_enc_input = self.dropout(hand_seq)

        lip_enc_input = self.layer_norm(lip_enc_input)
        hand_enc_input = self.layer_norm(hand_enc_input)

        layer_idx = 0

        if src_mask!=None:
            src_mask = get_attn_pad_mask(src_mask, lip_seq.size(1))

        for enc_layer in self.mm_layer:
            layer_idx += 1

            lip_enc_input = self.lip_se(lip_enc_input)
            hand_enc_input = self.hand_se(hand_enc_input)

            lip_enc_input, _, hand_enc_input, _ = enc_layer(lip_enc_input, hand_enc_input, slf_attn_mask=src_mask)
           
            # lip_enc_input, lip_ling = torch.split(lip_enc_input, dim=-1, split_size_or_sections=[self.d_model, self.ling_model])
            # hand_enc_input, hand_ling = torch.split(hand_enc_input, dim=-1, split_size_or_sections=[self.d_model, self.ling_model])

            lip_enc = self.spe_projection(lip_enc_input)
            lip_ling = self.ling_projection(lip_enc_input)

            hand_enc = self.spe_projection(hand_enc_input)
            hand_ling = self.ling_projection(hand_enc_input)

            fusion_ling = (lip_ling + hand_ling)/2

            if layer_idx != self.n_layers:
                lip_enc_input = torch.cat([lip_enc, fusion_ling], dim=-1)
                hand_enc_input = torch.cat([hand_enc, fusion_ling], dim=-1)
            else:
                lip_enc_input = lip_enc
                hand_enc_input = hand_enc
                
        # post_lip_enc = []
        # post_hand_enc = []
        
            

        if return_attns:
            return lip_enc_input, hand_enc_input, fusion_ling, lip_enc_slf_attn_list, hand_enc_slf_attn_list

        return lip_enc_input, hand_enc_input, fusion_ling



class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, 
            n_layers, 
            n_head, 
            d_k, 
            d_v,
            d_model,
            subsampling_dropout = 0.1, 
            ffd_dropout=0.1, 
            ffd_expansion_factor=4, 
            attn_dropout=0.1, 
            conv_expansion_factor= 2, 
            conv_kernel_size =31, 
            conv_dropout=0.1, 
            half_step_residual=True):

        super().__init__()

        
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_k, d_v, 
                        ffd_dropout = ffd_dropout, 
                        ffd_expansion_factor = ffd_expansion_factor, 
                        attn_dropout = attn_dropout, 
                        conv_expansion_factor = conv_expansion_factor, 
                        conv_kernel_size = conv_kernel_size, 
                        conv_dropout = conv_dropout, 
                        half_step_residual=half_step_residual)
            for _ in range(n_layers)])

        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(p=subsampling_dropout)
        self.d_model = d_model


    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, src_seq, src_mask, return_attns=False):
       
        enc_slf_attn_list = []
        enc_input = self.dropout(src_seq)
        enc_input = self.layer_norm(enc_input)

        slf_attn_mask = get_attn_pad_mask(src_mask, src_seq.size(1) )

        for enc_layer in self.layer_stack:

            enc_input, slf_attn = enc_layer(enc_input, non_pad_mask=src_mask, slf_attn_mask=slf_attn_mask)
            
            enc_slf_attn_list += [slf_attn] if return_attns else []

        if return_attns:
            return enc_input, enc_slf_attn_list

        return enc_input