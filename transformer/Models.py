''' Define the Transformer model '''
import sys
import torch
import torch.nn as nn
import numpy as np
from transformer.Attention import MultiHeadAttentionFusion
from transformer.Decoder import Decoder
from transformer.Fusion import AttnFusion, Fusion
from transformer.Modules import Linear, SELayer, Swish
from transformer.visual_frontend import Hand_ANN, Lip_Hand_CNN, VisualFrontend, PosFrontend
from transformer.Encoder import Encoder, MMEncoder
# from transformer.Decoder import Decoder
import torch.nn.functional as F
import math
from transformer.Mask import *

class VisionBackbone(nn.Module): # two different network for lip and hand
    def __init__(self):
        super().__init__()
        self.front_end_lip = VisualFrontend()
        self.front_end_hand = VisualFrontend()
        self.pos_end = PosFrontend(d_in=2, d_out=512)

    def forward(self, src_seq, src_length):
        lip, hand, hand_pos = src_seq
        batch_size, frame_num = lip.size(0), lip.size(1)
        hand_pos = hand_pos.view(batch_size, frame_num, -1)
        if batch_size > 1:
            src_mask = get_input_mask(lip, src_length)
        else:
            src_mask = None

        lip = self.front_end_lip(lip, src_mask)
        hand = self.front_end_hand(hand, src_mask)
        hand_pos = self.pos_end(hand_pos)
        
        hand = hand + hand_pos

        return lip, hand

class CSTransformerHead(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, dataset, d_src_seq, n_trg_vocab,
            d_word_vec=512, 
            d_model=512, 
            n_layers=6, 
            n_layers_dec = 1,
            n_head=8, d_k=64, d_v=64, 
            n_position = 1000,
            subsampling_dropout = 0.1,
            ffd_dropout=0.1, 
            ffd_expansion_factor=4, 
            attn_dropout=0.1, 
            conv_expansion_factor= 2, 
            conv_kernel_size =31, 
            conv_dropout=0.1, 
            half_step_residual=True):

        super().__init__()
        self.dataset = dataset

        self.d_model = d_model

        self.encoder = MMEncoder(
            dataset = dataset,
            d_model=d_model,
            ling_model = int(d_model/2),
            n_layers=n_layers, 
            n_head=n_head, 
            d_k=d_k, d_v=d_v,
            subsampling_dropout = subsampling_dropout,
            ffd_dropout=ffd_dropout, 
            ffd_expansion_factor=ffd_expansion_factor, 
            attn_dropout=attn_dropout, 
            conv_expansion_factor= conv_expansion_factor, 
            conv_kernel_size =conv_kernel_size, 
            conv_dropout=conv_dropout, 
            half_step_residual=half_step_residual
            )
        
        self.modal_fusion = Fusion(din=2*d_model, fusion_dim=d_model)
    
        self.sub_rate = 4
        self.ling_model=int(d_model/2)
        
        self.decoder = Decoder(
            d_word_vec=d_word_vec,
            n_trg_vocab=n_trg_vocab, 
            n_layers=n_layers_dec, 
            n_head=n_head, 
            d_k=d_k, 
            d_v=d_v,
            d_model=d_model, 
            ffd_expansion_factor=ffd_expansion_factor, 
            n_position= n_position,
            ffd_dropout=ffd_dropout,
            attn_dropout=attn_dropout,
            dec_dropout=attn_dropout, 
            scale_emb=True)

        # self.front_end = VisualFrontend()
        # self.pos_end = PosFrontend(d_in=2, d_out=512)

        self.code_emb = torch.nn.Embedding(n_trg_vocab, 512)
        self.code_emb.requires_grad_(False) #fix embedding layer 
        
        self.ling_projection =  nn.Sequential(
            nn.LayerNorm(d_model),
            Swish(),
            Linear(d_model, int(d_model/2))
            )

        self.ling_recover = nn.Sequential(
            Linear(int(d_model/2), d_model),
            Swish(),
            nn.LayerNorm(d_model))
        
        self.linear = nn.Linear(d_model, n_trg_vocab)
        
        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, lip, hand, src_length, code_book):

        # lip, hand, hand_pos = src_seq # (batch_size, frame, channel, width, height)
        batch_size, frame_num = lip.size(0), lip.size(1)

        # print(lip.shape)
        # hand_pos = hand_pos.view(batch_size, frame_num, -1)

        # mask from the padding frame
        if batch_size > 1:
            src_mask = get_input_mask(lip, src_length)
        else:
            src_mask = None

        # lip = self.front_end(lip, src_mask)
        # hand = self.front_end(hand, src_mask)
        # hand_pos = self.pos_end(hand_pos)
        
        # hand = hand + hand_pos
        code_book = self.code_emb(code_book)

        lip_scores = F.softmax(torch.matmul(lip, code_book.T), dim=-1)
        hand_scores = F.softmax(torch.matmul(hand, code_book.T), dim=-1)
        lip_linguistic = torch.matmul(lip_scores, code_book)
        hand_linguistic = torch.matmul(hand_scores, code_book)
        
        lip_ling = self.ling_projection(lip_linguistic)
        hand_ling = self.ling_projection(hand_linguistic)

        lip, hand, fusion_ling = self.encoder(lip, hand, lip_ling, hand_ling, src_mask)

        if self.dataset == 'Chinese' or self.dataset == 'English':
            if lip.shape[1]%self.sub_rate != 0:
               
                shape = lip.shape
                lip = lip[:,:shape[1]-shape[1]%self.sub_rate,:]
                hand = hand[:,:shape[1]-shape[1]%self.sub_rate,:]
                fusion_ling = fusion_ling[:,:shape[1]-shape[1]%self.sub_rate,:]
                src_length = src_length - shape[1]%self.sub_rate
        
            lip = torch.split(lip, self.sub_rate, dim=1)
            hand = torch.split(hand, self.sub_rate, dim=1)
            fusion_lings = torch.split(fusion_ling, self.sub_rate, dim=1)

            lip = [torch.mean(lip_enc, dim=1, keepdim=True) for lip_enc in lip]
            hand = [torch.mean(hand_enc, dim=1, keepdim=True) for hand_enc in hand]
            fusion_ling = [torch.mean(fusion_ling, dim=1, keepdim=True) for fusion_ling in fusion_lings]

            lip = torch.cat(lip, dim=1)
            hand = torch.cat(hand, dim=1)
            fusion_ling = torch.cat(fusion_ling, dim=1)

            output_lengths = src_length >> 2
            # output_lengths = torch.div(src_length, 3, rounding_mode='floor')
            if src_mask != None:
                src_mask = src_mask[:,::4,:]
                if src_mask.shape[1] > lip.shape[1]:
                    src_mask = src_mask[:,:lip.shape[1],:]
                sub_length = lip.size(1)
                src_mask = get_attn_pad_mask(src_mask, sub_length)

        else:
            
            output_lengths = src_length

        fusion_ling = self.ling_recover(fusion_ling)

        lip = self.decoder(fusion_ling, lip, src_mask)
        hand = self.decoder(fusion_ling, hand, src_mask)
        fusion_modality = self.modal_fusion(lip, hand)
        
        post_feature = fusion_modality.transpose(0, 1)
        post_feature = F.log_softmax(post_feature, dim=2)
        output_post = self.linear(fusion_modality)
        output_post = F.log_softmax(output_post, dim=2)
        output_post = output_post.transpose(0, 1)

        ling_feature = fusion_ling.transpose(0, 1)
        ling_feature = F.log_softmax(ling_feature, dim=2) #feature for mutual learing
        output_ling = self.linear(fusion_ling)
        output_ling = F.log_softmax(output_ling, dim=2)
        output_ling = output_ling.transpose(0, 1)

        return lip_ling, hand_ling, ling_feature, output_ling, output_post, output_lengths # distill ling feature
        # return lip_ling, hand_ling, post_feature, output_ling, output_post, output_lengths # distill post feature


    
    # def distance(self, src_seq, src_length, code_book):

    #     lip, hand, hand_pos = src_seq # (batch_size, frame, channel, width, height)
    #     batch_size, frame_num = lip.size(0), lip.size(1)

    #     # print(lip.shape)
    #     hand_pos = hand_pos.view(batch_size, frame_num, -1)

    #     # mask from the padding frame
    #     if batch_size > 1:
    #         src_mask = get_input_mask(lip, src_length)
    #     else:
    #         src_mask = None

    #     lip = self.front_end(lip, src_mask)
    #     hand = self.front_end(hand, src_mask)
    #     hand_pos = self.pos_end(hand_pos)
       
    #     hand = hand + hand_pos
    #     code_book = self.code_emb(code_book)

    #     lip_scores = F.softmax(torch.matmul(lip, code_book.T), dim=-1)
    #     hand_scores = F.softmax(torch.matmul(hand, code_book.T), dim=-1)
    #     lip_linguistic = torch.matmul(lip_scores, code_book)
    #     hand_linguistic = torch.matmul(hand_scores, code_book)
    #     linguistic = (lip_linguistic + hand_linguistic)/2

    #     # scores = (torch.matmul(lip, code_book.T)+torch.matmul(hand, code_book.T))/0.5
    #     # scores = F.softmax(scores, dim=-1)
    #     # linguistic = torch.matmul(scores, code_book)
       
    #     ling = self.ling_projection(linguistic)
    
    #     lip = torch.cat((lip, ling), dim=-1)
    #     hand = torch.cat((hand, ling), dim=-1)

    #     lip, hand, fusion_ling = self.encoder(lip, hand, src_mask)

    #     fusion_ling = self.ling_recover(fusion_ling)

    #     lip_distance = torch.matmul(lip, fusion_ling.T).squeeze(0)
    #     hand_distance = torch.matmul(hand, fusion_ling.T).squeeze(0)

    #     lip_distance = torch.diag(lip_distance)
    #     hand_distance = torch.diag(hand_distance)

    #     sum_ = lip_distance + hand_distance

    #     lip_utilizate = lip_distance / sum_
    #     hand_utilizate = hand_distance / sum_
  
    #     return lip_utilizate, hand_utilizate

class CuedSpeechTransformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, dataset, d_src_seq, n_trg_vocab,
            d_word_vec=512, 
            d_model=512, 
            n_layers=6, 
            n_layers_dec = 1,
            n_head=8, d_k=64, d_v=64, 
            n_position = 1000,
            subsampling_dropout = 0.1,
            ffd_dropout=0.1, 
            ffd_expansion_factor=4, 
            attn_dropout=0.1, 
            conv_expansion_factor= 2, 
            conv_kernel_size =31, 
            conv_dropout=0.1, 
            half_step_residual=True):

        super().__init__()
        self.feature_extractor = VisionBackbone()
        self.head = CSTransformerHead(dataset = dataset, 
                                        d_src_seq = d_src_seq, 
                                        n_trg_vocab = n_trg_vocab,
                                        d_word_vec=d_word_vec, 
                                        d_model=d_model, 
                                        n_layers=n_layers, 
                                        n_layers_dec = n_layers_dec,
                                        n_head=n_head, d_k=d_k, d_v=d_v, 
                                        n_position = n_position,
                                        subsampling_dropout = subsampling_dropout,
                                        ffd_dropout=ffd_dropout, 
                                        ffd_expansion_factor=ffd_expansion_factor, 
                                        attn_dropout=attn_dropout, 
                                        conv_expansion_factor= conv_expansion_factor, 
                                        conv_kernel_size = conv_kernel_size, 
                                        conv_dropout=conv_dropout, 
                                        half_step_residual=half_step_residual)

    def forward(self, src_seq, src_length, code_book):
        lip, hand = self.feature_extractor(src_seq, src_length)
        lip_ling, hand_ling, ling_feature, output_ling, output_post, output_lengths = self.head(lip, hand, src_length, code_book)
        # return lip, hand, output_ling, output_post, output_lengths
        return lip_ling, hand_ling, ling_feature, output_ling, output_post, output_lengths


# class LSTM(nn.Module):
#     def __init__(self, n_trg_vocab = 43, 
#                     d_words = 512, 
#                     hidden_size = 512,
#                     num_layers = 2):
#         super().__init__()
#         self.lstm = torch.nn.LSTM(d_words, hidden_size, num_layers = num_layers, batch_first = True, bidirectional = True)
#         self.linear = torch.nn.Linear(hidden_size*2, n_trg_vocab)
#         self.sub_rate = 4

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         # subsampling
#         if out.shape[1]%self.sub_rate != 0:
#             shape = out.shape
#             out = out[:,:shape[1]-shape[1]%self.sub_rate,:]
#         out = torch.split(out, self.sub_rate, dim=1)
#         out = [torch.mean(out_enc, dim=1, keepdim=True) for out_enc in out]
#         out = torch.cat(out, dim=1)

#         out = self.linear(out)
#         out = F.log_softmax(out, dim=2)

#         return out


# class LingLSTM(nn.Module):
#     def __init__(self, n_trg_vocab = 43, 
#                     d_words = 512, 
#                     hidden_size = 512,
#                     num_layers = 2):
#         super().__init__()
#         self.code_emb = torch.nn.Embedding(n_trg_vocab, d_words)
#         self.head = LSTM(n_trg_vocab=n_trg_vocab,
#                             d_words=d_words,
#                             hidden_size=hidden_size,
#                             num_layers=num_layers
#                             )

#     def forward(self, x):
#         v_emb = self.code_emb(x)
#         return self.head(v_emb)

# if __name__ == "__main__":
#     lstm = LingLSTM()
#     test_ts = torch.tensor([ 3, 32, 41, 22, 34, 41, 14, 22, 38, 41, 14, 22, 38, 41,  5, 27, 41,  7, 25, 41,  3, 27, 41, 11, 33])
#     test_ts = test_ts.unsqueeze(0)
#     out = lstm(test_ts)
#     print(out.shape)