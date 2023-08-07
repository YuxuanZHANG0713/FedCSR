from re import L
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from tools import AverageMeter, accuracy
from losses import FCTCLoss

from transformer.Optim import ScheduledOptim

from chinese.CCS_dataset import CCSDataset

from utils.general import num_params, train, evaluate
from utils.matrix import *
from torch.cuda.amp import GradScaler
import random


class Client:
    def __init__(self, cfg, client_idx, glob_model, ling_model):
        self.model = copy.deepcopy(glob_model)
        self.ling_model = copy.deepcopy(ling_model)
        self.local_model = copy.deepcopy(glob_model.state_dict())
        self.id = client_idx  # integer

        self.cfg = cfg
        self.local_epoches = self.cfg["local_epoches"]

        self.loss_CTC = FCTCLoss(blank=self.cfg['blank'], zero_infinity=False)
        self.cosine_loss = nn.CosineEmbeddingLoss(reduction='mean')
        self.ensemble_loss=nn.KLDivLoss(log_target=True, reduction='batchmean')
        self.optimizer = ScheduledOptim(
            optim.Adam(self.model.parameters(), betas=(self.cfg["momentum1"], self.cfg["momentum2"]), weight_decay=self.cfg["weight_decay"], eps=1e-09), 
            self.cfg["lr_mul"], self.cfg["d_model"], self.cfg["warmup_steps"])

        self.init_data()

    def init_data(self):
        id_map = {0:'hs', 1: 'lf', 2:'wt', 3:'xp'}
        self.train_data = CCSDataset(self.cfg["data_path"], is_train = True, phone2index=self.cfg["phone_to_index"], client = id_map[self.id])
        kwargs = {"num_workers": self.cfg["num_workers"], "pin_memory": True}
        self.trainLoader = DataLoader(self.train_data, batch_size=self.cfg["batch_size"], shuffle=True, drop_last=True, **kwargs)
        self.train_samples = len(self.train_data)


    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr


    def clone_model_paramenter(self, model, clone_model):
        model_state = model.state_dict()
        for key in model_state.keys():
            clone_model[key] = model_state[key]
        return clone_model


    def set_parameters(self, state_dict_model, state_dict_ling, beta=1):

        self.model.load_state_dict(state_dict_model)
        self.ling_model.load_state_dict(state_dict_ling)
        
        self.local_model = copy.deepcopy(self.model.state_dict())
   
    def local_train(self, client_id, glob_iter, early_stop, verbose, regularization):
        device = torch.device('cuda:0' if self.cfg['cuda'] else 'cpu')

        scaler = GradScaler()
        for iter in range(self.local_epoches):
            start_time = time.time()
            trainingLoss, trainingCER, trainingWER = train(self.model, self.ling_model, self.trainLoader, self.optimizer, self.loss_CTC, self.ensemble_loss, device, scaler, self.cfg)
            output = ("Epoch: %03d || Tr.Loss: %.6f|| Tr.CER: %.3f || Tr.WER: %.3f || Time: %.2f"
              %(iter, trainingLoss, trainingCER, trainingWER, time.time()-start_time))  # TODO
            print(output)
                  

        self.local_model = self.clone_model_paramenter(self.model, self.local_model)



    def evaluate(self, eval_loader):
        device = torch.device('cuda:0' if self.cfg['cuda'] else 'cpu')
        validationLoss, validationCER, validationWER, predictions, targets = evaluate(self.model, eval_loader, self.loss_CTC, device, return_result=True, args=self.cfg)
        output = ("Val.Loss: %.6f ||Val.CER: %.3f ||Val.WER: %.3f"
              %(validationLoss, validationCER, validationWER))  # TODO
        print(output)

    
        
       

