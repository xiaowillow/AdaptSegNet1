####################################################
##### This is focal loss class for multi class #####
##### University of Tokyo Doi Kento            #####
####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

class FocalLoss2d(nn.Module):

    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):#[1, 2, 416, 416]
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)#[1, 2, 173056]
            input = input.transpose(1,2)#[1, 173056, 2]
            input = input.contiguous().view(-1, input.size(2)).squeeze()#[173056, 2]
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)#[1, 1, 173056]
            target = target.transpose(1,2)#[1, 173056, 1]
            target = target.contiguous().view(-1, target.size(2)).squeeze()#[173056]
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
