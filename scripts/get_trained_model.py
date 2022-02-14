import torch
import torch.nn as nn
from torchvision import models

def get_trained_resnet(filename: str):
    ''' Returns the specific trained resnet-152 on our dataset
    
        .pt (pytorch model) file must be in present working directory (pwd)
    '''

    model = models.resnet152(pretrained= True, progress = True) # import a pretrained PyTorch implementation of Resnet-152
    model.fc = nn.Linear(model.fc.in_features,431) # update fully connected layer to reflect the numbef of classes for our task (431 unique car models)

    if filename == 'SGD.pt':
        model.load_state_dict(torch.load(filename))
    elif filename == 'SGD_w_OneCycleLR.pt':
        model.load_state_dict(torch.load(filename))
    elif filename == 'AdamW.pt':
        model.load_state_dict(torch.load(filename))
    else:
        return 'Incorrect filename or path. Please try again.'
    
    return model