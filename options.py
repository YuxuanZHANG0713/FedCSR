# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # Python version: 3.6

# import argparse
# # model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

# def args_parser():
#     parser = argparse.ArgumentParser()


#     # federated arguments for server
#     parser.add_argument('--glob_iters', type=int, default=100, help="number of global iterations")
#     # parser.add_argument('--model_name', type=str, default='CSTFmodel', help="model name")
#     # parser.add_argument('--gen_lr', type=float, default=0.0004, help="rounds of training")
#     # parser.add_argument('--gen_batchsize', type=int, default=32, help="batchsize of training generator")
#     # parser.add_argument('--gen_epoches', type=int, default=50, help="epochs of training generator")
#     # parser.add_argument('--temperature', type=float, default=1.0, help="temperature of KD")
#     # parser.add_argument('--gen_alpha', type=float, default=1.0, help="gen_alpha")
#     # parser.add_argument('--gen_beta', type=float, default=0, help="gen_beta")
#     # parser.add_argument('--gen_eta', type=float, default=1.0, help="gen_eta")
#     # parser.add_argument('--gen_weight_decay', type=float, default=0.02, help="weight_decay")
#     # parser.add_argument('--generative_alpha', type=float, default=1.0, help="generative_alpha")
#     # parser.add_argument('--generative_beta', type=float, default=1.0, help="generative_beta")


#     # federated arguments for client
#     parser.add_argument('--num_clients', type=int, default=4, help="number of local clients")
#     parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
#     # parser.add_argument('--dataset', type=str, default='CSdata', help="name of dataset")
#     # parser.add_argument('--local_lr', type=float, default=0.1, help="rounds of training")
#     # parser.add_argument('--local_batchsize', type=int, default=32, help="batchsize of local training")
#     parser.add_argument('--local_epoches', type=int, default=1, help="epochs of local training")

#     # parser.add_argument('--store_name', type=int, default='20221208', help="name of log dir")

#     # parser.add_argument('--model_name', type=str, default='resnet32', choices=model_names, help='model name')
    

#     # federated arguments for dataset
#     # parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
#     # parser.add_argument('--imb_factor', default=0.02, type=float, help='imbalance factor')
#     # parser.add_argument('--iid', type=bool, default=False, help='whether i.i.d or not')
#     # parser.add_argument('--dataset_alpha', default=0.5, type=float, help='alpha for spliting dataset')

#     # parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
#     # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
#     parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
#     # parser.add_argument('--root_log',type=str, default='log')
#     # parser.add_argument('--checkpoints',type=str, default='checkpoints')


#     args = parser.parse_args()
#     return args
