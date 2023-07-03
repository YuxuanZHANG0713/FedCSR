"""
Author: Liu Lei
Copyright (c) 2022 Lei Liu
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, shutil

from sklearn.manifold import TSNE

from losses import FCTCLoss
# import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

from chinese.chi_config import chi_args
from transformer.Models import CuedSpeechTransformer
from chinese.CCS_dataset import CCSDataset

from transformer.Optim import ScheduledOptim
from utils.general import num_params, evaluate, evaluate_test
from utils.matrix import *
from torch.cuda.amp import GradScaler
import random

def prepare_folders(args, store_name):

    if not args["checkpoint_path"]:
        print('No experiment result will be saved.')
        return

    path_list = []
    path_list.append(args["checkpoint_path"])

    if args['single_speaker']:
        dire = args["checkpoint_path"] + '/single/'
    else:
        dire = args["checkpoint_path"] + '/multi/'

    path_list.append(dire)
    path_list.append(dire + args["modal"])
    path_list.append(dire + args["modal"] + '/' + store_name)
    path_list.append(dire + args["modal"] + '/' + store_name + "/models")
    path_list.append(dire + args["modal"] + '/' + store_name + "/plots")
    path_list.append(dire + args["modal"] + '/' + store_name + "/matrix")
    path_list.append(dire + args["modal"] + '/' + store_name + "/logs")


    args["log_path"] = dire + args["modal"] + '/' + store_name + "/logs"
    args["matrix"]   = dire + args["modal"] + '/' + store_name + "/matrix"
    args["plots"]    = dire + args["modal"] + '/' + store_name + "/plots"
    args["models"]   = dire + args["modal"] + '/' + store_name + "/models"


    for path_item in path_list:
        if not os.path.exists(path_item):
            os.mkdir(path_item)

def tSNE(lingFeatures, outputBatch):

    # outputBatch = outputBatch.cpu()
    # inputLenBatch = inputLenBatch.cpu()
    outputBatch[:,:,0] = np.log(np.exp(outputBatch[:,:,0]) + np.exp(outputBatch[:,:,42]))
    reqIxs = np.arange(outputBatch.shape[2])
    reqIxs = reqIxs[reqIxs != 42]
    outputBatch = outputBatch[:,:,reqIxs]

    predCharIxs = np.squeeze(np.argmax(outputBatch, axis=2).T, axis = 0) # list of labels [num of samples]
    lingFeatures = np.squeeze(lingFeatures, axis=1) # list of features [num of samples, 512]

    selectedPhone = {"b":1, "p":2, "m":3, "f":4, "d":5, "t":6, "n":7, "l":8, "g":9, "k":10, "h":11, "j":12,
                         "q":13, "x":14, "zh":15, "ch":16, "sh":17, "r":18, "z":19, "c":20, "s":21, "y":22, "w":23, "yu":24,
                         "a":25, "o":26, "e":27, "i":28, "u":29, "v":30, "ai":31, "ei":32, "ao":33, "ou":34, "er":35, "an":36,
                         "en":37, "ang":38, "eng":39, "ong":40}
    # newSelectedPhone = {1: 1, 2: 2, 3: 3, 4: 4, 25: 5, 26: 6, 27: 7, 28: 8}
    # selectedPhoneIndex = {1:"b", 2:"p", 3:"m", 4:"f", 5:"d", 25:"a", 26:"o", 27:"e", 28:"i", 29:"u"}

    selectedIndexes = np.isin(predCharIxs, list(selectedPhone.values()))

    predCharIxs = predCharIxs[selectedIndexes]
    # predCharIxs = [newSelectedPhone[i] for i in predCharIxs]
    # colorIxs = [selectedPhoneIndex[i] for i in predCharIxs]
    lingFeatures = lingFeatures[selectedIndexes, :]

    lingFeaturesEmbed = TSNE(n_components=2, init="pca").fit_transform(lingFeatures)

    return lingFeaturesEmbed, predCharIxs

    


def main(dataset, store_name):

    matplotlib.use("Agg")

    args=chi_args
    #training dataloader
    # trainData = CCSDataset(args["data_path"], is_train=True, phone2index=args["phone_to_index"], single = args["single_speaker"])
    #evaluate dataloader
    # valData = CCSDataset(args["data_path"], is_train=False, phone2index=args["phone_to_index"], single = args["single_speaker"])
    valData = CCSDataset(args["data_path"], is_train = False, phone2index=args["phone_to_index"], client = 'multi_speaker')
    # store_name = '_'.join(['test', str(args["n_layers"]), str(args["warmup_steps"]), str(args["weight_decay"])])
    prepare_folders(args, store_name)

    # For reproducibility
    if args['seed'] is not None:
        torch.manual_seed(args["seed"])
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(args["seed"])
        random.seed(args["seed"])

    device = torch.device('cuda:0' if args['cuda'] else 'cpu')

    #========= Loading Dataset =========#
    kwargs = {"num_workers": args["num_workers"], "pin_memory": False}
    
    valLoader = DataLoader(valData, batch_size=args["batch_size"], shuffle=False, **kwargs)

    persons = valData.persons
    person2id = {'hs': 1, 'lf':2, 'wt':3, 'xp':4}
    personsIdx = [person2id[i] for i in persons]

    # Define the transformer
    model = CuedSpeechTransformer(
        dataset = dataset,
        d_src_seq=args["d_word_vec"],
        n_trg_vocab=args["n_trg_vocab"],
        d_k=args["d_k"],
        d_v=args["d_v"],
        d_model=args["d_model"],
        d_word_vec=args["d_word_vec"],
        n_layers=args["n_layers"],
        n_layers_dec = args["n_layers_dec"],
        n_head=args["n_head"],
        n_position=args["n_position"],
        subsampling_dropout = args["subsampling_dropout"],
        ffd_dropout=args["ffd_dropout"],
        ffd_expansion_factor=args["ffd_expansion_factor"],
        attn_dropout=args["attn_dropout"],
        conv_expansion_factor=args["conv_expansion_factor"],
        conv_kernel_size=args["conv_kernel_size"],
        conv_dropout=args["conv_dropout"],
        half_step_residual=args["half_step_residual"],
       )

    # for k,v in model.named_parameters():
    #     if "front_end" in k or "pos_end" in k 

    # Load pretrained weights for frontend
    
    # model_dict = model.state_dict()
    weight_path = '/home/20230405ours_server_.pt'
    # weight = torch.load(weight_path, map_location=device)
    # model.load_state_dict(weight)
    model = torch.load(weight_path)

    model.to(device)
   
    loss_CTC = FCTCLoss(blank=args['blank'], zero_infinity=False)
    # loss_CE = nn.CrossEntropyLoss().cuda()
    # loss_CTC = nn.CTCLoss(blank=args['blank'], zero_infinity=False)

    
    # validationLossCurve = list()
    
    # validationCERCurve = list()
    
    # validationWERCurve = list()


    #printing the total and trainable parameters in the model
    numFeatureParams, featureTrainableParams = num_params(model)

    print("\nNumber of total parameters in the model = %d" %(numFeatureParams))
    print("Number of trainable parameters in the model = %d\n" %(featureTrainableParams))
    print("\nEvaluating the model .... \n")

    # lip_utilize, hand_utilize = modality_utilize(model, valLoader, device, args)
    lingFeatures, outputPosts, validationLoss, validationCER, validationWER, predictions, targets, personsIdx = evaluate_test(model, valLoader, loss_CTC, device, return_result=True, args=args, persons=personsIdx)

    print("ValCER:", validationCER)

    all_labels = targets+predictions
    all_labels = list(np.unique(all_labels))
    
    cm = confusion_matrix(targets, predictions, labels=all_labels)  # 计算混淆矩阵
    
    cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis]+1e-9)  # 归一化
    labels = list(args["index_to_phone"][i] for i in all_labels)  # 类别集合
    np.set_printoptions(precision=2)
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', vmin=0, vmax=0.9)  # 在特定的窗口上显示图像
    plt.title('confusion matrix')  # 图像标题
    plt.colorbar(fraction=0.046, pad=0.04)
    num_local = np.array(range(len(labels)))
    plt.xticks(num_local, labels, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(args["matrix"] + "/eval-matrix.png")
    plt.close()

    print("\Evaluation Done.\n")

    #tsne

    lingFeaturesEmbed, predChar = tSNE(lingFeatures, outputPosts)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(lingFeaturesEmbed[:,0], lingFeaturesEmbed[:,1],s=5, c=predChar)
    selectedPhone = {"b":1, "p":2, "m":3, "f":4, "a":5, "o":6, "e":7, "i":8}
    # legend1 = ax.legend(scatter.legend_elements()[0], labels = list(selectedPhone.keys()), loc="lower left", title="phones")
    # ax.add_artist(legend1)
    plt.savefig(args["matrix"] + "/tSNE.png")
    plt.close()

    #tsne for persons
    fig, ax = plt.subplots()
    scatter = ax.scatter(lingFeaturesEmbed[:,0], lingFeaturesEmbed[:,1],s=5, c=personsIdx)
    # legend1 = ax.legend(scatter.legend_elements()[0], labels = list(selectedPhone.keys()), loc="lower left", title="phones")
    # ax.add_artist(legend1)
    plt.savefig(args["matrix"] + "/tSNE_persons.png")

    print("\TSNE Done.\n")



    return


if __name__ == "__main__":
    dataset = 'Chinese'
    store_name = 'ours'
    main(dataset, store_name)
    
