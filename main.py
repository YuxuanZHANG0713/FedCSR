#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

from re import A
import torch
from torchvision import datasets
from server import Server
from chinese.chi_config import chi_args


if __name__ == '__main__':

    # training
    cfg = chi_args
    device = torch.device('cuda:0' if cfg['cuda'] else 'cpu')
    print(cfg)
    glob_server = Server(cfg)
    glob_server.train()


