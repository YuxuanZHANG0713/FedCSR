
import torch
from torch.nn.utils.rnn import pad_sequence

from chinese.chi_config import chi_args

def collate_fn(dataBatch):
    """
    Collate function definition used in Dataloaders.
    """
    # lip/hand list in a batch
    # olipBatch = [data[0][0] for data in dataBatch]
    lipBatch = [data[0][0].reshape(-1) for data in dataBatch]
    handBatch = [data[0][1].reshape(-1) for data in dataBatch]
    posBatch = [data[0][2].reshape(-1) for data in dataBatch]
    batchsize = len(lipBatch)

    # lipBatch = [int(data[0][0].reshape(-1).shape[0]) for data in dataBatch]
    

    # frame count for each video a batch 
    inputLenBatch = torch.stack([data[2] for data in dataBatch])

    # padding 0 to max frame count in a batch: args["src_pad_idx"] = 0

    lipBatch = pad_sequence(lipBatch, batch_first=True).reshape((batchsize, -1, 3, chi_args["roi_size"], chi_args["roi_size"]))
    
    handBatch = pad_sequence(handBatch, batch_first=True).reshape((batchsize, -1, 3, chi_args["roi_size"], chi_args["roi_size"]))

    posBatch = pad_sequence(posBatch, batch_first=True).reshape((batchsize, -1, 1, 1, 1))

    
    inputBatch = (lipBatch, handBatch, posBatch)
   

    if not any(data[1] is None for data in dataBatch):
        # args["trg_pad_idx"] = 0
        targetBatch = torch.cat([data[1] for data in dataBatch])
    else:
        targetBatch = None

    if not any(data[3] is None for data in dataBatch):
        targetLenBatch = torch.stack([data[3] for data in dataBatch])
    else:
        targetLenBatch = None
    
    # print(inputLenBatch-targetLenBatch)
    return inputBatch, targetBatch, inputLenBatch, targetLenBatch


# a = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
# b = torch.tensor([[[2,2,2],[2,2,2]]])

# print(a.shape)
# print(a)
# a = a.reshape(-1)
# print(a.shape)
# print(a)

# a = a.reshape(1,3,3)
# # print(a.shape)
# print(a.shape)
# print(a)

# data = [a.reshape(-1),b.reshape(-1)]
# b = torch.tensor([[[2,2,2],[2,2,2], [2, 2,2]]])
# print(b.shape)

# c = torch.tensor([[[3,3,3],[3,3,3], [3, 3, 3], [3, 3, 3]]])
# print(c.shape)

# data = [a.reshape(-1),b.reshape(-1)]

# data = pad_sequence(data, batch_first=True).reshape(2, 1, -1, 3)
# # print(data.shape)
# print(data)