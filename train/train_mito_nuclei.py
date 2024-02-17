import sys
sys.path.append('/home/ccy/cellcycle')

from dataset import Mydataset
from transform import train_transform, test_transform, RandomTransform
from torch.utils.data import DataLoader

from model.UNet import Modified_UNet, UNet, UNetpp
from model.UNet3Plus import UNet_3Plus
from model.ResUNet import ResUnet

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.optim import Adam
from utils.record import *
from utils.score import *
from utils.loss_function import *
from scipy.stats import pearsonr

import config
import time
from tqdm.auto import tqdm
import logging
from utils.record import*
import matplotlib.pyplot as plt

torch.cuda.empty_cache()

'''Image Path  =============================================='''
train_img_path = "/home/ccy/cellcycle/Data/Hoechst/Hoechst_Mito_img/train/brightfield_png1"
train_mito_path = '/home/ccy/cellcycle/Data/Hoechst/Hoechst_Mito_img/train/mito_png1'
train_nucleus_path = '/home/ccy/cellcycle/Data/Hoechst/Hoechst_Mito_img/train/n_png1'
val_img_path = '/home/ccy/cellcycle/Data/Hoechst/Hoechst_Mito_img/val/brightfield_png1'
val_mito_path = '/home/ccy/cellcycle/Data/Hoechst/Hoechst_Mito_img/val/mito_png1'
val_nucleus_path = '/home/ccy/cellcycle/Data/Hoechst/Hoechst_Mito_img/val/n_png1'

'''Data Ready ==============================================='''
train_transform = RandomTransform(train_transform)
test_transform = RandomTransform(test_transform)
trainDS = Mydataset(image_dir=train_img_path, label_dir=train_nucleus_path, transform=train_transform)
valDS =  Mydataset(image_dir=val_img_path, label_dir=val_nucleus_path, transform=test_transform)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(valDS)} examples in the test set...")

trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=0)
valLoader = DataLoader(valDS, shuffle=False,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=0)

trainSteps = len(trainDS) // config.BATCH_SIZE
valSteps = len(valDS) // config.BATCH_SIZE

'''Model Selection ==============================================='''
net = Modified_UNet().to(config.DEVICE)
unet= UNet(3,1).to(config.DEVICE)
unet3plus = UNet_3Plus(3,1).to(config.DEVICE)
unetpp = UNetpp(3,1).to(config.DEVICE)
model = net

'''Loss function Selection  ======================================'''
mse = nn.MSELoss()
huberloss = HuberLoss()
logcosh = LogCoshLoss()
bce = nn.BCELoss()
ce = nn.CrossEntropyLoss()
loss = huberloss

'''Optimizer  ===================================================='''
opt = Adam(model.parameters(), lr=config.INIT_LR,betas=(0.9, 0.999))
scheduler = ReduceLROnPlateau(opt,mode='min', factor=0.1, patience=10, verbose=True, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

trainloss = []
validloss = []
trainPCC = []
validPCC = []
best_acc = 0.0
stale = 0

'''Load the record  ==========================================='''
#trainloss, validloss, trainPCC, validPCC= load_(config.RECORD_PATH)
#model, _ = loadmodel(config.CHECKPOINT_PATHS, model, opt, device= config.DEVICE)

'''Learning rate adjustment manually   ==========================================='''
#new_learning_rate = 0.0001
#for param_group in opt.param_groups:
#    param_group['lr'] = new_learning_rate

'''training ==========================================='''
print("[INFO] training the network...")
logging.basicConfig(filename='', level=logging.INFO, filemode="a")
progress = tqdm(range(config.NUM_EPOCHS))
gradient_accumulation_steps = 4

for epoch in range(config.NUM_EPOCHS):
	model.train()
 
	train_loss = 0.0
	train_PCC = 0.0
	trainacc = 0.0
	
	for batch_idx, (x, y) in enumerate(trainLoader):
		x = x.to(config.DEVICE)
		y = y.to(config.DEVICE)
		pred = model(x)
	
		train_loss += loss(pred,y)
    		
		y_np = y.flatten().detach().cpu().numpy()
		pred_np = pred.flatten().detach().cpu().numpy()
		corr, _ = pearsonr(pred_np, y_np)  # 忽略p值
		train_PCC += corr

		train_loss.backward()
		if (batch_idx + 1) % gradient_accumulation_steps == 0:
			opt.step()
			opt.zero_grad()
		torch.cuda.empty_cache()

	with torch.no_grad():
		model.eval()
		valid_loss = 0.0
		valid_PCC = 0.0
		validacc = 0.0

		for x, y in valLoader:
			x = x.to(config.DEVICE) 
			y = y.to(config.DEVICE)
			pred = model(x)
			
			valid_loss += loss(pred,y)
   
			y_np = y.flatten().cpu().numpy()
			pred_np = pred.flatten().cpu().numpy()
			corr, _ = pearsonr(pred_np, y_np)  # 忽略p值
			valid_PCC += corr
			
	trainloss.append(train_loss/trainSteps)
	validloss.append(valid_loss/valSteps)

	trainPCC.append(train_PCC/trainSteps)
	validPCC.append(valid_PCC/valSteps)
	
	scheduler.step(valid_loss/valSteps)
 
	if valid_PCC/valSteps > best_acc:
		print(f"Best model found at epoch {epoch}, saving model")
		torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, config.CHECKPOINT_PATHS)
		torch.save(model, config.MODEL_PATH)
		best_acc = valid_PCC/valSteps
		stale = 0 
	else:
		stale += 1
		if stale >= 20:
			print(f"No improvment 20 consecutive epochs, early stopping")
			break
	
	savestatis(trainloss, validloss, trainPCC, validPCC)
 
	'''Print in terminal'''
	tqdm.write(f"Epoch: {epoch}/{100}")
	tqdm.write(f"trainingLoss:{trainloss/trainSteps}  validationLoss:{valid_loss/valSteps}")
	tqdm.write(f"trainingAccuracy:{trainacc/trainSteps}  validationAccuracy:{validacc/valSteps}")
	current_lr = scheduler.optimizer.param_groups[0]['lr']
	tqdm.write("Learning rate: {}".format(current_lr))
 
	logging.info(f"Epoch {epoch + 1}/{epoch}, Train Loss: {trainloss/trainSteps}, Val Loss: {valid_loss/valSteps}, Train Accuracy: {trainacc/trainSteps}, Val Accuracy: {validacc/valSteps}, Learning Rate: {current_lr}")
	progress.update(1)