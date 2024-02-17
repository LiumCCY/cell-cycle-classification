import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F

'''torch
def dice_coefficient(pred_mask, true_mask,threshold):
    smooth = 1e-7
    pred_mask_binary = torch.where(pred_mask >= threshold, torch.ones_like(pred_mask), torch.zeros_like(pred_mask))
    true_mask_binary = torch.where(true_mask >= threshold, torch.ones_like(true_mask), torch.zeros_like(true_mask))
    pred_mask_binary  = pred_mask_binary .contiguous().view(-1)
    true_mask_binary = true_mask_binary.contiguous().view(-1)
    intersection = (pred_mask_binary  * true_mask_binary).sum()
    dice = (2.0 * intersection + smooth) / (pred_mask_binary .sum() + true_mask_binary.sum() + smooth)
    return dice

def f_score(y_true, y_pred, beta=1, eps=1e-7, threshold=None):
    
    if threshold is not None:
        y_pred = (y_pred > threshold).float()

    tp = torch.sum(y_true * y_pred)
    fp = torch.sum(y_pred) - tp
    fn = torch.sum(y_true) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + eps)
    return f_score
'''
'''numpy'''
import numpy as np

def dice_coefficient(pred_mask, true_mask, threshold):
    smooth = 1e-7
    pred_mask_binary = np.where(pred_mask >= threshold, np.ones_like(pred_mask), np.zeros_like(pred_mask))
    true_mask_binary = np.where(true_mask >= threshold, np.ones_like(true_mask), np.zeros_like(true_mask))
    pred_mask_binary = pred_mask_binary.flatten()
    true_mask_binary = true_mask_binary.flatten()
    intersection = np.sum(pred_mask_binary * true_mask_binary)
    dice = (2.0 * intersection + smooth) / (np.sum(pred_mask_binary) + np.sum(true_mask_binary) + smooth)
    return dice

def f_score(y_true, y_pred, beta=1, eps=1e-7, threshold=None):
    if threshold is not None:
        y_pred = (y_pred > threshold).astype(np.float32)

    tp = np.sum(y_true * y_pred)
    fp = np.sum(y_pred) - tp
    fn = np.sum(y_true) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + eps)
    return f_score

def calculate_iou(pred, target, threshold=0.5):
    # 将预测和标签通过阈值转换为二进制图像
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # 计算交集
    intersection = (pred_binary * target_binary).sum().item()
    
    # 计算并集
    union = ((pred_binary + target_binary) > 0).sum().item()
    
    # 计算 IoU
    iou = intersection / (union + 1e-7)  # 加上一个小的常数以防止分母为零
    
    return iou


