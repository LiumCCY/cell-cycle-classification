import torch
import torch.nn as nn
import torch.nn.functional as F

'''Regression Loss'''
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, input, target):
        diff = torch.abs(input - target)
        loss = torch.where(diff < self.delta, 0.5 * diff**2, self.delta * (diff - 0.5 * self.delta))
        return loss.mean()

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, input, target):
        diff = input - target
        loss = torch.log(torch.cosh(diff))
        return loss.mean()
    
'''Classification Loss'''
class FocalBCELoss(nn.Module):
    def __init__(self, gamma=0.5, alpha=None, reduction='mean'):
        super(FocalBCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce_loss = nn.BCELoss(reduction='none')
        self.reduction = reduction
        
    def forward(self, input, target):
        bce_loss = self.bce_loss(input, target)
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
        
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1, smooth=1e-5):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        
    def dice_loss(self, label, prediction):
        label = input.view(-1)
        prediction = prediction.view(-1)
        intersection = (label * prediction).sum()
        dice = (2. * intersection + self.smooth) / (label.sum() + prediction.sum() + self.smooth)
        return 1-dice
    
    def focal_loss(self, label, prediction):
        BCE = F.binary_cross_entropy_with_logits(label, prediction, reduction='none')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
        return focal_loss.mean()
    
    def forward(self, label, prediction):
        dice_loss = self.dice_loss(label, prediction)
        focal_loss = self.focal_loss(label, prediction)
        return dice_loss + focal_loss
        

def dice_loss(pred, target):
    smooth = 0.00001
    intersection = (pred * target).sum()
    dice_score = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice_score


