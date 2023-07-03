import torch

front_end_weight = torch.load('lrw_resnet18_mstcn_adamw_s3.pth.tar')
print(front_end_weight.keys())
# print(front_end_weight['model_state_dict'])
for name in iter(front_end_weight['model_state_dict']):
    print(name)
    
