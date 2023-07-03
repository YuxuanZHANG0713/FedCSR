import torch

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-1)


def get_input_mask(seq, input_lengths):
    batchsize = seq.size(0)
    non_pad_mask = seq.new_ones(seq.size()[:2])  # N x T

    for i in range(batchsize):
        non_pad_mask[i, input_lengths[i]:] = 0

    return non_pad_mask.unsqueeze(-1)


def get_attn_pad_mask(src_mask, expand_length):
    """mask position is set to 1"""
    pad_mask = src_mask.squeeze(-1).lt(1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    
    return subsequent_mask



