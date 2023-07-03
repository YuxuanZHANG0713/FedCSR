import torch
import shutil
import os
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def sentence_align(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch):
    targetBatch = targetBatch.cpu()
    targetLenBatch = targetLenBatch.cpu()

    preds = list(torch.split(predictionBatch, predictionLenBatch.tolist()))
    trgts = list(torch.split(targetBatch, targetLenBatch.tolist()))

    align_preds = list()
    align_trgts = list()

    for n in range(len(preds)):
        pred = preds[n].numpy()
        trgt = trgts[n].numpy()

        pred = list(pred)
        trgt = list(trgt)

        if len(pred) < len(trgt):
            pred.extend([42 for i in range(len(trgt) - len(pred))])
        elif len(pred) > len(trgt):
            trgt.extend([42 for i in range(len(pred) - len(trgt))])
        pass

        align_preds.extend(pred)
        align_trgts.extend(trgt)

    return align_preds, align_trgts


def frame_label_align(outputBatch, inputLenBatch, frameBatch, frameLenBatch, eosIx):
    outputBatch = outputBatch.cpu()
    frameBatch = frameBatch.cpu()
    frameLenBatch = frameLenBatch.cpu()

    trgts = list(torch.split(frameBatch, frameLenBatch.tolist()))
    predCharIxs = torch.argmax(outputBatch, dim=2).T.numpy()


    align_preds = list()
    align_trgts = list()
    for i in range(len(predCharIxs)):
        pred = predCharIxs[i]
        trgt = trgts[i].numpy()
        pred = list(pred)
        trgt = list(trgt)

        trgt = trgt[:len(pred)]

        align_preds.extend(pred)
        align_trgts.extend(trgt)

    return align_preds, align_trgts




# def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
#     if not title:
#         if normalize:
#             title = 'Normalized confusion matrix'
#         else:
#             title = 'Confusion matrix, without normalization'

#     # Compute confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     # classes = [str(i) for i in range(42)]

#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            # ... and label them with the respective list entries
#            xticklabels=classes, yticklabels=classes)

#     # Rotate the tick labels and set their alignment.
#     # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#     # plt.title(title, fontsize=18)
#     plt.xlabel('Predicted label', fontsize=17)
#     plt.ylabel('True label', fontsize=17)

#     # Loop over data dimensions and create text annotations.
#     fmt = '.2f' if normalize else 'd'
#     norm = 1000 if normalize else 1
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j] / norm, fmt),
#                     ha="center", va="center",
#                     color="black")  # color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     return ax