import torch
import torch.nn as nn
import models
import numpy as np
import time
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device = ', device)

def train(data, model, criterion, optimizer):

    model.train()
    num_batches = int(len(data) / data.batch_size)
    for batch_idx in range(num_batches):
        """
        Forward steps
        Set up the variables used in training process ( imgs, latents ...)
        
            train on GPU (** to(device) or .cuda() **)
            ToTensor (** Variable(torch.Tensor(xxx) **)
            
            Forward steps:
                1.
                2.
                ...
                output = 
            
        """

        """
        Backward steps
        
                optimizer.zero_grad()
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
        """

    # return whatever you need
    return None

def test(data, model, loss):
    model.eval()

    """
    Forward steps
    Set up the variables used in training process ( imgs, latents ...)

        train on GPU (** to(device) or .cuda() **)
        ToTensor (** Variable(torch.Tensor(xxx) **)

        Forward steps:
            1.
            2.
            ...
            output = 

    """

    # return whatever you need
    return None

def set_params_requires_grad(model, feature_extract):

    if feature_extract:
        for param in model.parameters():
            # 不需要更新梯度，冻结某些层的梯度
            param.requires_grad = False

def pretrain_model(model_name, feature_extract=True):

    model_init = None

    if model_name == 'resnext101_32x16d':
        # load pretrain_model
        model_init = models.resnext101_32x16d_wsl()
        # set_layers to train
        set_params_requires_grad(model_init, feature_extract)

        ###
        # example for changing the classes to 4
        ###
        num_input = model_init.fc.in_features
        model_init.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=num_input, out_features=4)
        )

    return model_init

def init_models(params):
    model = None

    return model

