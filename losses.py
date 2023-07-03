import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def focal_loss(input_values, gamma):
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)



class FCTCLoss(nn.Module):
    def __init__(self, zero_infinity= False, use_focal_loss=True, **kwargs):
        super(FCTCLoss, self).__init__()
        # self.loss_func = nn.CTCLoss(blank=0, zero_infinity=zero_infinity, reduction='mean')
        self.loss_func = nn.CTCLoss(blank=0, zero_infinity=False) # batch_size = 1

        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, targets, preds_lengths, label_lengths):
        
        loss = self.loss_func(predicts, targets, preds_lengths, label_lengths)
        # print(loss.shape)
        if self.use_focal_loss:
            weight = torch.exp(-loss)
            weight = (1 - weight) ** 2 * loss
            
        # loss = loss.mean() # batch_size = 1
        return loss