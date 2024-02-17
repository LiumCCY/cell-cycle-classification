import sys
sys.path.append('/home/ccy/cellcycle')

import argparse
import cv2
import config
import os

from dataset import FucciDataset
from transform_alb import train_transform, val_transform
from torch.utils.data import DataLoader

from model.UNet import Modified_UNet, UNet
from model.UNet3Plus import UNet_3Plus
from model.EUNet import EffUNet

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from utils.record import *
from utils.score import *
from utils.loss_function import *
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

print(f"torch.__version__{torch.__version__}")
print(f"torch.cuda.is_available {torch.cuda.is_available()}")
print(f"torch.cuda.device_count {torch.cuda.device_count()}")
#print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
#print(torch.cuda.current_device())
torch.cuda.empty_cache()

'''Image Path  =============================================='''
train_img_path = '/home/ccy/cellcycle/Data/FUCCI/Split/train/bf_mito(REAL)_nuclei'
train_label_path = '/home/ccy/cellcycle/Data/FUCCI/Split/train/label'
val_img_path = '/home/ccy/cellcycle/Data/FUCCI/Split/val/bf_mito(REAL)_nuclei'
val_label_path = '/home/ccy/cellcycle/Data/FUCCI/Split/val/label'

def count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
train_img_count = count_files(train_img_path)
train_label_count = count_files(train_label_path)
val_img_count = count_files(val_img_path)
val_label_count = count_files(val_label_path)
assert (train_img_count == train_label_count) and (val_img_count == val_label_count), "File count mismatch!"
print("Pass!")

'''Data Ready ==============================================='''
trainDS = FucciDataset(image_dir=train_img_path, label_dir=train_label_path, transform=train_transform)
valDS =  FucciDataset(image_dir=val_img_path, label_dir=val_label_path, transform=val_transform)
print(f"[INFO] {len(trainDS)} examples in the training set...")
print(f"[INFO] {len(valDS)} examples in the val set...")
trainSteps = len(trainDS) // config.BATCH_SIZE
valSteps = len(valDS) // config.BATCH_SIZE

trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=0)
valLoader = DataLoader(valDS, shuffle=False,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=0)

'''Check Image  ============================================='''
images, labels = next(iter(trainLoader))
for i in range(config.BATCH_SIZE):
    image1 = images[i] #(C, H, W)
    label1 = labels[i] #(C, H, W)
    image_np = image1.permute(1,2,0).numpy()
    label_np = label1.numpy()
    print(image_np.shape)
    print(label_np.shape)
    cv2.imwrite(f"/home/ccy/cellcycle/test/image{i}.png", image_np)
    cv2.imwrite(f"/home/ccy/cellcycle/test/label{i}.png", label_np)

'''Model Selection ==============================================='''
net = Modified_UNet()
unet= UNet(3,1)
unet3plus = UNet_3Plus(3,1)
eunet = EffUNet(3,3)

'''Loss function Selection  ======================================'''
focaldice_loss = FocalDiceLoss(alpha=0.5, gamma=1, smooth=1e-5)

def parse_args():
    parser = argparse.ArgumentParser(description="Training a model for cell cycle classification")
    #parser.add_argument("--model_type", type=str, default="eunet")
    #parser.add_argument("--loss_type", type=str, default="focaldice_loss")
    parser.add_argument("--lr", type=float, default=config.INIT_LR)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    return parser.parse_args()

def train(model, trainLoader, optimizer, loss_fn, device):
    model.train()
    total_loss, total_f1 = 0, 0
    for x, y in trainLoader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        pred_np, y_np = pred.flatten().cpu().numpy(), y.flatten().cpu().numpy()
        confu_metric = confusion_matrix(y_np, pred_np)
        precision, recall, f1, _ = precision_recall_fscore_support(y_np, pred_np, average = None)
        total_f1 += f1
        #total_pcc += pearsonr(pred_np, y_np)[0]
    return total_loss / len(trainLoader), total_f1 / len(trainLoader)

def validate(model, valLoader, loss_fn, device):
    model.eval()
    total_loss, total_f1 = 0, 0
    with torch.no_grad():
        for x, y in valLoader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            pred_np, y_np = pred.flatten().cpu().numpy(), y.flatten().cpu().numpy()
            confu_metric = confusion_matrix(y_np, pred_np)
            precision, recall, f1, _ = precision_recall_fscore_support(y_np, pred_np, average = None)
            total_f1 += f1
            #total_pcc += pearsonr(pred_np, y_np)[0]
    return total_loss / len(valLoader), total_f1 / len(valLoader)

def main():
    args = parse_args()

    # Model selection
    model = eunet.to(config.DEVICE)

    # Loss function selection
    loss_fn = focaldice_loss

    # Optimizer
    opt = Adam(model.parameters(), lr=config.INIT_LR,betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(opt,mode='min', factor=0.1, patience=10, verbose=True, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    stats = TrainingStatistics(config.RECORD_PATH)
    if os.path.exists(config.RECORD_PATH):
        train_losses, validlosses, train_f1s, valid_f1s = stats.load()
    
    epoch_train_losses, epoch_valid_losses = [], []
    epoch_train_f1s, epoch_valid_f1s = [], []
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_f1 = train(model, trainLoader, opt, loss_fn, config.DEVICE)
        valid_loss, valid_f1 = validate(model, valLoader, loss_fn, config.DEVICE)
        scheduler.step(valid_loss)

        if valid_f1 > best_acc:
            best_acc = valid_f1
            ModelStats.save_checkpoint(epoch+1, model, opt, best_acc, config.CHECKPOINT_PATHS)
            torch.save(model, config.MODEL_PATH)
        
        epoch_train_losses.append(train_loss)
        epoch_valid_losses.append(valid_loss)
        epoch_train_f1s.append(train_f1)
        epoch_valid_f1s.append(valid_f1)    
        stats.save(epoch_train_losses, epoch_valid_losses, epoch_train_f1s, epoch_valid_f1s)
            
        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, Train PCC: {train_f1}, Valid PCC: {valid_f1}")

if __name__ == "__main__":
    main()