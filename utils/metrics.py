"""
Author: Liu Lei
Copyright (c) 2022 Liu Lei
"""

import torch
import numpy as np
import editdistance



def compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch):

    """
    Function to compute the Character Error Rate using the Predicted character indices and the Target character
    indices over a batch.
    CER is computed by dividing the total number of character edits (computed using the editdistance package)
    with the total number of characters (total => over all the samples in a batch).
    The <EOS> token at the end is excluded before computing the CER.
    """

    targetBatch = targetBatch.cpu()
    targetLenBatch = targetLenBatch.cpu()

    # preds = list(torch.split(predictionBatch, predictionLenBatch.tolist()))
    # trgts = list(torch.split(targetBatch, targetLenBatch.tolist()))
    
    preds = [predictionBatch]
    trgts = [targetBatch]
    
    # print(trgts)
    totalEdits = 0
    totalChars = 0

    for n in range(len(preds)):
        pred = preds[n].numpy()
        trgt = trgts[n].numpy()

        # trgt = trgt[trgt != 42]
        # trgt = trgt[trgt != 43]
        # print(trgt)
        # trgt = trgt[trgt != 41] # added
        
        numEdits = editdistance.eval(pred, trgt)
        totalEdits = totalEdits + numEdits
        totalChars = totalChars + len(trgt)

    return totalEdits/totalChars

# 
#                          "a":24, "o":25, "e":26, "i":27, "u":28, "v":29, "ai":30, "ei":31, "ao":32, "ou":33, "er":34, "an":35,
#                          "en":36, "ang":37, "eng":38, "ong":39}
def compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, spaceIx):

    """
    Function to compute the Word Error Rate using the Predicted character indices and the Target character
    indices over a batch. The words are obtained by splitting the output at spaces.
    WER is computed by dividing the total number of word edits (computed using the editdistance package)
    with the total number of words (total => over all the samples in a batch).
    The <EOS> token at the end is excluded before computing the WER. Words with only a space are removed as well.
    """

    targetBatch = targetBatch.cpu()
    targetLenBatch = targetLenBatch.cpu()

    # preds = list(torch.split(predictionBatch, predictionLenBatch.tolist()))
    # trgts = list(torch.split(targetBatch, targetLenBatch.tolist()))
    preds = [predictionBatch]
    trgts = [targetBatch]
    totalEdits = 0
    totalWords = 0

    for n in range(len(preds)):
        pred = preds[n].numpy()
        trgt = trgts[n].numpy()
        # trgt = trgt[trgt != 42]
        predWords = np.split(pred, np.where(pred == spaceIx)[0])
        predWords = [predWords[0].tostring()] + [predWords[i][1:].tostring() for i in range(1, len(predWords)) if len(predWords[i][1:]) != 0]

        trgtWords = np.split(trgt, np.where(trgt == spaceIx)[0])
        trgtWords = [trgtWords[0].tostring()] + [trgtWords[i][1:].tostring() for i in range(1, len(trgtWords))]

        

        numEdits = editdistance.eval(predWords, trgtWords)
        totalEdits = totalEdits + numEdits
        totalWords = totalWords + len(trgtWords)

    return totalEdits/totalWords



def accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res