import torch.nn as nn
import math
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from .loss import OhemCrossEntropy2d
from .lovasz_losses import lovasz_softmax
import scipy.ndimage as nd

class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, use_weight=True, reduction='mean'):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        if not reduction:
            print("disabled the reduction.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if len(preds) >= 2:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            loss1 = self.criterion(scale_pred, target)

            scale_pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
            loss2 = self.criterion(scale_pred, target)
            return loss1 + loss2*0.4
        else:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            loss = self.criterion(scale_pred, target)
            return loss

class CriterionOhemDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduction='mean'):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2d(ignore_index, thresh, min_kept)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, preds, target):#target[2, 416, 416]
        h, w = target.size(1), target.size(2)
        #h, w = target.size(2), target.size(3)
        scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)#[2, 2, 416, 416]
        loss1 = self.criterion1(scale_pred, target)

        scale_pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)

        scale_pred = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        loss3 = self.criterion2(scale_pred, target)
        
        pred = preds[0] + preds[1]
        loss4 = lovasz_softmax(F.softmax(pred, dim=1), target, ignore=self.ignore_index)
        # L = λa · la + λc · lc + λf · lf
        return 0.7*loss1 + 0.6*loss2 + 0.4*loss3


class CriterionOhemDSN2(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduction='mean'):
        super(CriterionOhemDSN2, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)
        loss2 = lovasz_softmax(F.softmax(scale_pred, dim=1), target, ignore=self.ignore_index)

        return loss1 + loss2
