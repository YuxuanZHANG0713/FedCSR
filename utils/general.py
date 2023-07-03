"""
Author: Liu Lei
Copyright (c) 2022 Liu Lei
"""
import numpy as np
from sklearn.linear_model import MultiTaskLasso
import torch
from tqdm import tqdm
import torch.nn.functional as F

from .metrics import compute_cer, compute_wer, accuracy
from .matrix import sentence_align
from .decoders import ctc_greedy_decode, ctc_search_decode
from torch.cuda.amp import autocast as autocast
import editdistance
import sys

# import pynvml

def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return numTotalParams, numTrainableParams


def kl_div(pred, tgrt, outputLenBatch, args):
    loss = 0
    avg_sum = 0
    # tgrt = tgrt.transpose(0, 1)

    pred = pred.view(-1, args["n_trg_vocab"])
    tgrt = tgrt.view(-1, args["n_trg_vocab"])

    loss_function = torch.nn.KLDivLoss(reduction="sum", log_target=True)
    for idx, len in enumerate(outputLenBatch):
        avg_sum += len
        loss += loss_function(pred, tgrt)

    loss /= avg_sum

    return loss


# def train_lstm(model, trainLoader, optimizer, loss_CTC, loss_CE, device, scaler, args):

#     """
#     Function to train the model for one iteration. (Generally, one iteration = one epoch, but here it is one step).
#     It also computes the training loss, CER and WER. The CTC decode scheme is always 'greedy' here.
#     """

#     trainingLoss = 0
#     trainingCER = 0
#     trainingWER = 0

#     for batch, (inputBatch, targetBatch, _, _, inputLenBatch, targetLenBatch) in enumerate(trainLoader):
        
#     # for batch, (lip, hand, targetBatch, lipLenBatch, handLenBatch, targetLenBatch) in enumerate(tqdm(trainLoader, leave=False, desc="Train", ncols=60)):
#         inputBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device), (inputBatch[2].float()).to(device))
#         targetBatch = (targetBatch.int()).to(device)
#         targetBatch = targetBatch.squeeze(0)
#         inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device),  (targetLenBatch.int()).to(device)
        
        
#         optimizer.zero_grad()
#         model.train()
    
#         outputPostBatch, outputLenBatch = model(inputBatch, inputLenBatch)
    
#         with torch.backends.cudnn.flags(enabled=False):
#             loss = loss_CTC(outputPostBatch, targetBatch, outputLenBatch, targetLenBatch)
            
#         loss.backward()
#         optimizer.step()
#         torch.autograd.set_detect_anomaly(True)
       
#         trainingLoss = trainingLoss + loss.item()
        
#         predictionBatch, predictionLenBatch = ctc_greedy_decode(outputPostBatch, outputLenBatch, args["eosIx"])

#         trainingCER = trainingCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
#         trainingWER = trainingWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, args["spaceIx"])

#     trainingLoss = trainingLoss/len(trainLoader)
#     trainingCER = trainingCER/len(trainLoader)
#     trainingWER = trainingWER/len(trainLoader)
#     return trainingLoss, trainingCER, trainingWER


# def evaluat_lstm(model, evalLoader,  loss_CTC, loss_CE, device, return_result, args):

#     """
#     Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
#     The CTC decode scheme can be set to either 'greedy' or 'search'.
#     """

#     evalLoss = 0
#     evalCER = 0
#     evalWER = 0

#     predictions = []
#     tagrets = []


#     for batch, (inputBatch, targetBatch, _, _, inputLenBatch, targetLenBatch) in enumerate(evalLoader):
        
#     # for batch, (lip, hand, targetBatch, lipLenBatch, handLenBatch, targetLenBatch) in enumerate(tqdm(trainLoader, leave=False, desc="Train", ncols=60)):
#         inputBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device), (inputBatch[2].float()).to(device))
#         targetBatch = (targetBatch.int()).to(device)
#         targetBatch = targetBatch.squeeze(0)
#         inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device),  (targetLenBatch.int()).to(device)
        
#         model.eval()
#         with torch.no_grad():
            
#             outputPostBatch, outputLenBatch = model(inputBatch, inputLenBatch)
#             with torch.backends.cudnn.flags(enabled=False):
#                 loss = loss_CTC(outputPostBatch, targetBatch, outputLenBatch, targetLenBatch)
        
#         evalLoss = evalLoss + loss.item()
#         if args["decodeScheme"] == "greedy":
#             predictionBatch, predictionLenBatch = ctc_greedy_decode(outputPostBatch, outputLenBatch, args["eosIx"])
#         elif args["decodeScheme"] == "search":
#             predictionBatch, predictionLenBatch = ctc_search_decode(outputPostBatch, outputLenBatch, args["beamSearchParams"],
#                                                                     args["spaceIx"], args["eosIx"], args["lm"])
#         else:
#             print("Invalid Decode Scheme")
#             exit()

#         # print(targetLenBatch.cpu()-predictionLenBatch)
#         evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
#         evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, args["spaceIx"])

#         # if return_result:
#         #     alignedPred, alignedTar = sentence_align(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
#         #     # alignedPred, alignedTar = frame_label_align(outputBatch, inputLenBatch, frameBatch, frameLenBatch, evalParams["eosIx"])
#         #     predictions.extend(alignedPred)
#         #     tagrets.extend(alignedTar)



#     evalLoss = evalLoss/len(evalLoader)
#     evalCER = evalCER/len(evalLoader)
#     evalWER = evalWER/len(evalLoader)
#     return evalLoss, evalCER, evalWER, None, None


def train(model, ling_model, trainLoader, optimizer, loss_CTC, ensemble_loss, device, scaler, args):

    """
    Function to train the model for one iteration. (Generally, one iteration = one epoch, but here it is one step).
    It also computes the training loss, CER and WER. The CTC decode scheme is always 'greedy' here.
    """

    trainingLoss = 0
    # trainingLossSim = 0
    trainingCER = 0
    trainingWER = 0
    
    model.to(device)

    for batch, (inputBatch, targetBatch, frameTarget, inputCode, inputLenBatch, targetLenBatch) in enumerate(trainLoader):
        
        #for debugging

        # pynvml.nvmlInit()
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0表示显卡标号
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print('-----before train-----')
        # print('inputLenBatch:', inputLenBatch)
        # print('Total gpumem: %.3f, used: %.3f, left: %.3f'%(meminfo.total/1024**2, meminfo.used/1024**2, meminfo.free/1024**2))
        # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated()/1024/1024/1024))
        # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved()/1024/1024/1024))
        # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved()/1024/1024/1024))

    # for batch, (lip, hand, targetBatch, lipLenBatch, handLenBatch, targetLenBatch) in enumerate(tqdm(trainLoader, leave=False, desc="Train", ncols=60)):
        inputBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device), (inputBatch[2].float()).to(device))
        targetBatch = (targetBatch.int()).to(device)
        targetBatch = targetBatch.squeeze(0)
        inputCode = inputCode.squeeze(0)
        inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device),  (targetLenBatch.int()).to(device)
        inputCode = (inputCode.int()).to(device)
        
        optimizer.zero_grad()
        model.train()
       
        # outputPostBatch, outputLenBatch = model(inputBatch, inputLenBatch)
        lip_ling, hand_ling, outputLingFeatureBatch, outputLingBatch, outputPostBatch, outputLenBatch = model(inputBatch, inputLenBatch, inputCode)
        early_fusion_ling = torch.cat((lip_ling, hand_ling), dim=-1)
        # get output of semantic model
        sem_model = ling_model.head
        sem_model.to(device)
        sem_model.eval()

        with torch.no_grad():
            # outputSemLip = sem_model(lip)
            # outputSemHand = sem_model(hand)
            # outputSemLip = sem_model(lip, lip, int(inputLenBatch/4))
            # outputSemHand = sem_model(hand, hand, int(inputLenBatch/4))
            outputSemFeature, outputSem = sem_model(early_fusion_ling, early_fusion_ling, int(inputLenBatch/4))
            outputSemFeatureFlat = outputSemFeature.reshape(-1, outputSemFeature.shape[2])
            outputSem = outputSem.transpose(0,1)
            # outputSemLipFlat = outputSemLip.reshape(-1, outputSemLip.shape[2])
            # outputSemHandFlat = outputSemHand.reshape(-1, outputSemHand.shape[2])
            # outputSemLip = outputSemLip.transpose(0,1)
            # outputSemHand = outputSemHand.transpose(0,1)
        
        outputLingFeatureFlat = outputLingFeatureBatch.reshape(-1, outputLingFeatureBatch.shape[2])
        # loss_sim_lip = ensemble_loss(outputPost, outputSemLip, torch.ones(outputPost.shape[0]).to(device))
        # loss_sim_hand = ensemble_loss(outputPost, outputSemHand, torch.ones(outputPost.shape[0]).to(device))
        # loss_sim_lip = ensemble_loss(outputPostFlat, outputSemLipFlat)
        # loss_sim_hand = ensemble_loss(outputPostFlat, outputSemHandFlat)
        loss_sim = ensemble_loss(outputLingFeatureFlat, outputSemFeatureFlat)
        # loss_sim = ensemble_loss(F.log_softmax(outputLingFeatureFlat, dim=-1), F.log_softmax(outputSemFeatureFlat, dim=-1))

            
    
        with torch.backends.cudnn.flags(enabled=False):
            # loss_sim = loss_sim_hand + loss_sim_lip
            loss_acc = loss_CTC(outputPostBatch, targetBatch, outputLenBatch, targetLenBatch) \
                        + loss_CTC(outputLingBatch, targetBatch, outputLenBatch, targetLenBatch)
            # loss_sem_acc = loss_CTC(outputSemHand, targetBatch, outputLenBatch, targetLenBatch) \
            #             + loss_CTC(outputSemLip, targetBatch, outputLenBatch, targetLenBatch)
            loss_sem_acc = loss_CTC(outputSem, targetBatch, outputLenBatch, targetLenBatch)
            loss = loss_acc + args['beta'] * loss_sim + args['gamma'] * loss_sem_acc
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.autograd.set_detect_anomaly(True)
       
        trainingLoss = trainingLoss + loss.item()
        
        predictionBatch, predictionLenBatch = ctc_greedy_decode(outputPostBatch, outputLenBatch, args["eosIx"])

        trainingCER = trainingCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
        trainingWER = trainingWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, args["spaceIx"])

        # print('loss:', loss)
        # print('loss_sim:', args['beta'] * (loss_sim_hand + loss_sim_lip))
        
        # pynvml.nvmlInit()
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0表示显卡标号
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # torch.cuda.empty_cache()

        # print('-----after backward and empty cache-----')
        # print('Total gpumem: %.3f, used: %.3f, left: %.3f'%(meminfo.total/1024**2, meminfo.used/1024**2, meminfo.free/1024**2))
        # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated()/1024/1024/1024))
        # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved()/1024/1024/1024))
        # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved()/1024/1024/1024))


    trainingLoss = trainingLoss/len(trainLoader)
    trainingCER = trainingCER/len(trainLoader)
    trainingWER = trainingWER/len(trainLoader)

    model.cpu()
    

    return trainingLoss, trainingCER, trainingWER


def evaluate(model, evalLoader,  loss_CTC, device, return_result, args):

    """
    Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
    The CTC decode scheme can be set to either 'greedy' or 'search'.
    """

    evalLoss = 0
    evalCER = 0
    evalWER = 0

    model.to(device)

    predictions = []
    tagrets = []


    for batch, (inputBatch, targetBatch, frameTarget, inputCode, inputLenBatch, targetLenBatch) in enumerate(evalLoader):
        inputBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device), (inputBatch[2].float()).to(device))

        targetBatch = (targetBatch.int()).to(device)
        targetBatch = targetBatch.squeeze(0)
        inputCode = inputCode.squeeze(0)

        inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device),  (targetLenBatch.int()).to(device)
        inputCode = (inputCode.int()).to(device)

        model.eval()
        with torch.no_grad():
            # lip_scores, hand_scores, output_linguistic, output_post, updated_length
            _, _, _, outputLingBatch, outputPostBatch, outputLenBatch = model(inputBatch, inputLenBatch, inputCode)
            with torch.backends.cudnn.flags(enabled=False):
                
                loss = 0
              
                loss += loss_CTC(outputPostBatch, targetBatch, outputLenBatch, targetLenBatch)
                loss += loss_CTC(outputLingBatch, targetBatch, outputLenBatch, targetLenBatch)
                # loss += loss_function(linguisticBatch, targetBatch, outputLenBatch, targetLenBatch)

                
        evalLoss = evalLoss + loss.item()
        if args["decodeScheme"] == "greedy":
            predictionBatch, predictionLenBatch = ctc_greedy_decode(outputPostBatch, outputLenBatch, args["eosIx"])
        elif args["decodeScheme"] == "search":
            predictionBatch, predictionLenBatch = ctc_search_decode(outputPostBatch, outputLenBatch, args["beamSearchParams"],
                                                                    args["spaceIx"], args["eosIx"], args["lm"])
        else:
            print("Invalid Decode Scheme")
            exit()

        # print(targetLenBatch.cpu()-predictionLenBatch)
        evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
        evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, args["spaceIx"])

        if return_result:
            alignedPred, alignedTar = sentence_align(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
            # alignedPred, alignedTar = frame_label_align(outputBatch, inputLenBatch, frameBatch, frameLenBatch, evalParams["eosIx"])
            predictions.extend(alignedPred)
            tagrets.extend(alignedTar)

    model.cpu()
    torch.cuda.empty_cache()

    evalLoss = evalLoss/len(evalLoader)
    evalCER = evalCER/len(evalLoader)
    evalWER = evalWER/len(evalLoader)
    return evalLoss, evalCER, evalWER, predictions, tagrets






def evaluate_test(model, evalLoader,  loss_CTC, device, return_result, args, persons):

    """
    Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
    The CTC decode scheme can be set to either 'greedy' or 'search'. Return raw hidden features for t-SNE visualization.
    """

    evalLoss = 0
    evalCER = 0
    evalWER = 0

    model.to(device)

    predictions = []
    tagrets = []

    lingFeatures = np.empty((0, 1, 512))
    outputPosts = np.empty((0, 1, 43))
    personIdx = np.empty((0))


    for batch, (inputBatch, targetBatch, frameTarget, inputCode, inputLenBatch, targetLenBatch) in enumerate(evalLoader):
        inputBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device), (inputBatch[2].float()).to(device))

        targetBatch = (targetBatch.int()).to(device)
        targetBatch = targetBatch.squeeze(0)
        inputCode = inputCode.squeeze(0)

        inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device),  (targetLenBatch.int()).to(device)
        inputCode = (inputCode.int()).to(device)

        model.eval()
        with torch.no_grad():
            # lip_scores, hand_scores, output_linguistic, output_post, updated_length
            lip_ling, hand_ling, outputLingFeatureBatch, outputLingBatch, outputPostBatch, outputLenBatch = model(inputBatch, inputLenBatch, inputCode)
            with torch.backends.cudnn.flags(enabled=False):
                
                loss = 0
              
                loss += loss_CTC(outputPostBatch, targetBatch, outputLenBatch, targetLenBatch)
                loss += loss_CTC(outputLingBatch, targetBatch, outputLenBatch, targetLenBatch)
                # loss += loss_function(linguisticBatch, targetBatch, outputLenBatch, targetLenBatch)

                
        evalLoss = evalLoss + loss.item()
        if args["decodeScheme"] == "greedy":
            predictionBatch, predictionLenBatch = ctc_greedy_decode(outputPostBatch, outputLenBatch, args["eosIx"])
        elif args["decodeScheme"] == "search":
            predictionBatch, predictionLenBatch = ctc_search_decode(outputPostBatch, outputLenBatch, args["beamSearchParams"],
                                                                    args["spaceIx"], args["eosIx"], args["lm"])
        else:
            print("Invalid Decode Scheme")
            exit()

        personId = np.array([persons[batch] for i in range(len(outputLingFeatureBatch))])
        lingFeatures = np.append(lingFeatures, outputLingFeatureBatch.detach().cpu().numpy(), axis=0)
        outputPosts = np.append(outputPosts, outputPostBatch.detach().cpu().numpy(), axis=0)
        personIdx = np.append(personIdx, personIdx, axis=0)

        # print(targetLenBatch.cpu()-predictionLenBatch)
        evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
        evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, args["spaceIx"])

        if return_result:
            alignedPred, alignedTar = sentence_align(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
            # alignedPred, alignedTar = frame_label_align(outputBatch, inputLenBatch, frameBatch, frameLenBatch, evalParams["eosIx"])
            predictions.extend(alignedPred)
            tagrets.extend(alignedTar)

        if batch == 100:
            break

        

    model.cpu()
    torch.cuda.empty_cache()

    evalLoss = evalLoss/len(evalLoader)
    evalCER = evalCER/len(evalLoader)
    evalWER = evalWER/len(evalLoader)
    return lingFeatures, outputPosts, evalLoss, evalCER, evalWER, predictions, tagrets, personIdx



# def train_eng(model, trainLoader, optimizer,  loss_CTC, loss_CE, device, scaler, args):

#     """
#     Function to train the model for one iteration. (Generally, one iteration = one epoch, but here it is one step).
#     It also computes the training loss, CER and WER. The CTC decode scheme is always 'greedy' here.
#     """

#     trainingLoss = 0
#     trainingCER = 0
#     trainingWER = 0

#     for batch, (inputBatch, targetBatch, frameTarget, inputCode, inputLenBatch, targetLenBatch) in enumerate(trainLoader):
        
#         inputBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device), (inputBatch[2].float()).to(device))
#         targetBatch = (targetBatch.to(torch.int64)).to(device)
#         targetBatch = targetBatch.squeeze(0)
#         frameTarget = (frameTarget.to(torch.int64)).to(device)
#         frameTarget = frameTarget.squeeze(0)
#         inputCode = inputCode.squeeze(0)
#         inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device),  (targetLenBatch.int()).to(device)
#         inputCode = (inputCode.int()).to(device)
        
#         optimizer.zero_grad()
#         model.train()
       
#         outputPostBatch, outputLenBatch = model(inputBatch, inputLenBatch, inputCode)
        
#         with torch.backends.cudnn.flags(enabled=False):

#             loss = loss_CTC(outputPostBatch, targetBatch, outputLenBatch, targetLenBatch) #+ loss_CTC(outputLingBatch, targetBatch, outputLenBatch, targetLenBatch)

#         loss.backward()
#         optimizer.step()
       
#         torch.autograd.set_detect_anomaly(True)
       
#         trainingLoss = trainingLoss + loss.item()

        
#         predictionBatch, predictionLenBatch = ctc_greedy_decode(outputPostBatch, outputLenBatch, args["eosIx"])
       
#         # print(predictionBatch)
       
#         trainingCER = trainingCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
#         trainingWER = trainingWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, args["spaceIx"])

#     trainingLoss = trainingLoss/len(trainLoader)
#     trainingCER = (trainingCER/len(trainLoader))
#     trainingWER = trainingWER/len(trainLoader)
#     return trainingLoss, trainingCER, trainingWER


# def evaluate_eng(model, evalLoader,  loss_CTC, loss_CE, device, return_result, args):

#     """
#     Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
#     The CTC decode scheme can be set to either 'greedy' or 'search'.
#     """

#     evalLoss = 0
#     evalCER = 0
#     evalWER = 0

#     predictions = []
#     tagrets = []
    
#     for batch, (inputBatch, targetBatch, frameTarget, inputCode, inputLenBatch, targetLenBatch) in enumerate(evalLoader):
#         inputBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device), (inputBatch[2].float()).to(device))

#         targetBatch = (targetBatch.to(torch.int64)).to(device)
#         targetBatch = targetBatch.squeeze(0)
#         inputCode = inputCode.squeeze(0)

#         frameTarget = (frameTarget.to(torch.int64)).to(device)
#         frameTarget = frameTarget.squeeze(0)

#         inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device),  (targetLenBatch.int()).to(device)
#         inputCode = (inputCode.int()).to(device)

#         model.eval()
#         with torch.no_grad():
#             outputPostBatch, outputLenBatch = model(inputBatch, inputLenBatch, inputCode)
#             with torch.backends.cudnn.flags(enabled=False):
#                 loss = loss_CTC(outputPostBatch, targetBatch, outputLenBatch, targetLenBatch) #+ loss_CTC(outputLingBatch, targetBatch, outputLenBatch, targetLenBatch)
               
#         evalLoss = evalLoss + loss.item()
#         if args["decodeScheme"] == "greedy":
#             predictionBatch, predictionLenBatch = ctc_greedy_decode(outputPostBatch, outputLenBatch, args["eosIx"])
#         elif args["decodeScheme"] == "search":
#             predictionBatch, predictionLenBatch = ctc_search_decode(outputPostBatch, outputLenBatch, args["beamSearchParams"],
#                                                                     args["spaceIx"], args["eosIx"], args["lm"])
#         else:
#             print("Invalid Decode Scheme")
#             exit()

        
#         evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
#         evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, args["spaceIx"])

#     evalLoss = evalLoss/len(evalLoader)
#     evalCER = evalCER/len(evalLoader)
#     evalWER = evalWER/len(evalLoader)
#     return evalLoss, evalCER, evalWER, None, None


# def train_fre(model, trainLoader, optimizer,  loss_CTC, loss_CE, device, scaler, args):

#     """
#     Function to train the model for one iteration. (Generally, one iteration = one epoch, but here it is one step).
#     It also computes the training loss, CER and WER. The CTC decode scheme is always 'greedy' here.
#     """
#     trainingLoss = 0
#     trainingCER = 0
#     trainingWER = 0

#     for batch, (inputBatch, targetBatch, frameTarget, inputCode, inputLenBatch, targetLenBatch) in enumerate(trainLoader):
        
#         inputBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device), (inputBatch[2].float()).to(device))
#         targetBatch = (targetBatch.to(torch.int64)).to(device)
#         targetBatch = targetBatch.squeeze(0)
#         frameTarget = (frameTarget.to(torch.int64)).to(device)
#         frameTarget = frameTarget.squeeze(0)
#         inputCode = inputCode.squeeze(0)
#         inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device),  (targetLenBatch.int()).to(device)
#         inputCode = (inputCode.int()).to(device)
        
#         optimizer.zero_grad()
#         model.train()
       
#         outputPostBatch, outputLenBatch = model(inputBatch, inputLenBatch, inputCode)
        
#         with torch.backends.cudnn.flags(enabled=False):

#             loss = loss_CTC(outputPostBatch, targetBatch, outputLenBatch, targetLenBatch) #+ loss_CTC(outputLingBatch, targetBatch, outputLenBatch, targetLenBatch)

#         loss.backward()
#         optimizer.step()
       
#         torch.autograd.set_detect_anomaly(True)
       
#         trainingLoss = trainingLoss + loss.item()
#         predictionBatch, predictionLenBatch = ctc_greedy_decode(outputPostBatch, outputLenBatch, args["eosIx"])
       
#         trainingCER = trainingCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
#         trainingWER = trainingWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, args["spaceIx"])

#     trainingLoss = trainingLoss/len(trainLoader)
#     trainingCER = (trainingCER/len(trainLoader))
#     trainingWER = trainingWER/len(trainLoader)
#     return trainingLoss, trainingCER, trainingWER


# def evaluate_fre(model, evalLoader,  loss_CTC, loss_CE, device, return_result, args):

#     """
#     Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
#     The CTC decode scheme can be set to either 'greedy' or 'search'.
#     """

#     evalLoss = 0
#     evalCER = 0
#     evalWER = 0

#     predictions = []
#     tagrets = []

#     for batch, (inputBatch, targetBatch, frameTarget, inputCode, inputLenBatch, targetLenBatch) in enumerate(evalLoader):
#         inputBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device), (inputBatch[2].float()).to(device))

#         targetBatch = (targetBatch.to(torch.int64)).to(device)
#         targetBatch = targetBatch.squeeze(0)
#         inputCode = inputCode.squeeze(0)

#         frameTarget = (frameTarget.to(torch.int64)).to(device)
#         frameTarget = frameTarget.squeeze(0)

#         inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device),  (targetLenBatch.int()).to(device)
#         inputCode = (inputCode.int()).to(device)

#         model.eval()
#         with torch.no_grad():
#             # lip_scores, hand_scores, output_linguistic, output_post, updated_length

#             outputPostBatch, outputLenBatch = model(inputBatch, inputLenBatch, inputCode)
#             # print(outputCEBatch.view(outputCEBatch.shape[1],-1).shape)
#             with torch.backends.cudnn.flags(enabled=False):
#                 loss = loss_CTC(outputPostBatch, targetBatch, outputLenBatch, targetLenBatch)
            
                
                
#         evalLoss = evalLoss + loss.item()
#         if args["decodeScheme"] == "greedy":
#             predictionBatch, predictionLenBatch = ctc_greedy_decode(outputPostBatch, outputLenBatch, args["eosIx"])
#         elif args["decodeScheme"] == "search":
#             predictionBatch, predictionLenBatch = ctc_search_decode(outputPostBatch, outputLenBatch, args["beamSearchParams"],
#                                                                     args["spaceIx"], args["eosIx"], args["lm"])
#         else:
#             print("Invalid Decode Scheme")
#             exit()

#         # print(targetLenBatch.cpu()-predictionLenBatch)
#         # print(predictionBatch)
#         evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
#         evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, args["spaceIx"])

#         # if return_result:
#         #     alignedPred, alignedTar = sentence_align(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
#         #     # alignedPred, alignedTar = frame_label_align(outputBatch, inputLenBatch, frameBatch, frameLenBatch, evalParams["eosIx"])
#         #     predictions.extend(alignedPred)
#         #     tagrets.extend(alignedTar)

#         # evalCER = evalCER + acc1.item()


#     evalLoss = evalLoss/len(evalLoader)
#     evalCER = (evalCER/len(evalLoader))
#     evalWER = evalWER/len(evalLoader)
#     return evalLoss, evalCER, evalWER, None, None