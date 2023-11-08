import sys
sys.path.append('/home/ccy/cellcycle')

from dataset import FucciDataset
from transform_alb import train_transform, val_transform
from torch.utils.data import DataLoader

from model.unet import Modified_UNet, UNet
from model.Unet3Plus import UNet_3Plus

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from utils.record import *
from utils.score import *
from utils.loss_function import *
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import cv2
import config
from tqdm.auto import tqdm
import logging

'''Image Path  =============================================='''
train_img_path = '/home/ccy/cellcycle/Data/FUCCI_SR/train/bf_mito_nuclei'
train_GFP_path = '/home/ccy/cellcycle/Data/FUCCI_SR/train/GFP'
train_RFP_path = '/home/ccy/cellcycle/Data/FUCCI_SR/train/RFP'
val_img_path = '/home/ccy/cellcycle/Data/FUCCI_SR/val/bf_mito_nuclei'
val_GFP_path = '/home/ccy/cellcycle/Data/FUCCI_SR/val/GFP'
val_RFP_path = '/home/ccy/cellcycle/Data/FUCCI_SR/val/GFP'

'''Data Ready ==============================================='''
trainDS = FucciDataset(image_dir=train_img_path, label_dir=train_GFP_path, transform=train_transform)
valDS =  FucciDataset(image_dir=val_img_path, label_dir=val_GFP_path, transform=val_transform)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(valDS)} examples in the val set...")
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
    # 获取对应索引的图像和标签
    image1 = images[i] # 假设 image 的形状为 (C, H, W)，C为通道数，H为高度，W为宽度
    label1 = labels[i] # 假设 label 的形状为 (C, H, W)
    label1 = label1.numpy()*255
    #print(type(image1))
    #print(type(label1))
    #print(image1.shape)
    #print(label1.shape)
    cv2.imwrite(f"/home/ccy/cellcycle/test/image{i}.png", image1.permute(1,2,0).numpy())
    cv2.imwrite(f"/home/ccy/cellcycle/test/label{i}.png", label1)

'''Model Selection ==============================================='''
net = Modified_UNet().to(config.DEVICE)
unet= UNet(3,1).to(config.DEVICE)
unet3plus = UNet_3Plus(3,1).to(config.DEVICE)
model = unet



'''Loss function Selection  ======================================'''
mse = nn.MSELoss()
huberloss = HuberLoss()
logcosh = LogCoshLoss()
bce = nn.BCELoss()
ce = nn.CrossEntropyLoss()
loss = mse


'''Optimizer  ===================================================='''
opt = Adam(model.parameters(), lr=config.INIT_LR,betas=(0.9, 0.999))
scheduler = ReduceLROnPlateau(opt,mode='min', factor=0.1, patience=10, verbose=True, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

trainloss = [] 
validloss = [] 
trainPCC = [] 
validPCC = []
trainPSNR = []
validPSNR = []
trainSSIM = []
validSSIM = []
trainiou =[]
valiou =[]
trainf1 =[]
valf1 = []
best_acc = 0.0
stale = 0

'''Load the record  ==========================================='''
#trainloss, validloss, trainPCC, validPCC, trainiou, valiou ,trainf1, valf1= load_(config.RECORD_PATH)
#model, _ = loadmodel(config.CHECKPOINT_PATHS, model, opt, device= config.DEVICE)

'''Learning rate adjustment manually   ==========================================='''
#new_learning_rate = 0.0001
#for param_group in opt.param_groups:
#    param_group['lr'] = new_learning_rate

'''training ==========================================='''
print("[INFO] training the network...")
logging.basicConfig(filename='trainBCE_dice.log', level=logging.INFO, filemode="a")
progress = tqdm(range(config.NUM_EPOCHS))
gradient_accumulation_steps = 2
torch.cuda.empty_cache()

for epoch in range(config.NUM_EPOCHS):
	model.train()
	total_train_loss =0.0
	train_loss = 0.0
	train_ACC = 0.0
	train_PSNR = 0.0
	train_SSIM = 0.0
	train_f1 =  0.0
	train_iou = 0.0
	
	for batch_idx, (x, y) in enumerate(trainLoader):
		x = x.to(dtype=torch.float32).to(config.DEVICE)  # 移动到指定设备
		y = torch.stack((y, y, y), dim=1)
		y = y.to(dtype=torch.float32).to(config.DEVICE)
		pred = model(x)
		#pred = pred.squeeze(1)
		train_loss = bce_dice_loss(pred,y)
		total_train_loss += train_loss	
  
		#print(np.unique(pred_np))
		#print(np.unique(y_np))

		y_np = y.detach().cpu().numpy()
		pred_np = pred.detach().cpu().numpy()
		train_iou += dice_coefficient(pred_np,y_np,0.5)
		train_f1 += f_score(y_np,pred_np)

		y_np_flatten = y.flatten().detach().cpu().numpy()
		pred_np_flatten = pred.flatten().detach().cpu().numpy()
		corr, _ = pearsonr(pred_np_flatten, y_np_flatten)  # 忽略p值
		train_ACC += corr

		#train_PSNR += peak_signal_noise_ratio(y_np,pred_np,data_range=1.0)
		#train_SSIM += structural_similarity(y_np,pred_np,data_range=1.0)

		train_loss.backward()
		if (batch_idx + 1) % gradient_accumulation_steps == 0:
			opt.step()
			opt.zero_grad()
		#torch.cuda.empty_cache()

	with torch.no_grad():
		model.eval()
		total_val_loss =0.0
		valid_loss = 0.0
		valid_ACC = 0.0
		valid_PSNR = 0.0
		valid_SSIM = 0.0
		val_f1 = 0.0
		val_iou =0.0

		for x, y in valLoader:
			x = x.to(torch.float32).to(config.DEVICE) 
			y = torch.stack((y, y, y), dim=1)
			y = y.to(torch.float32).to(config.DEVICE)
			pred = model(x)
			#pred = pred.squeeze(1)
		
			valid_loss = bce_dice_loss(pred,y)
			total_val_loss += valid_loss
   
			y_np = y.detach().cpu().numpy()
			pred_np = pred.detach().cpu().numpy()
			val_iou += dice_coefficient(pred_np,y_np,0.5)
			val_f1 += f_score(y_np,pred_np)
   
			y_np_flatten = y.flatten().detach().cpu().numpy()
			pred_np_flatten = pred.flatten().detach().cpu().numpy()
			corr, _ = pearsonr(pred_np_flatten, y_np_flatten)  # 忽略p值
			valid_ACC += corr
			
   
			#valid_PSNR += peak_signal_noise_ratio(y_np,pred_np,data_range=1.0)
			#valid_SSIM += structural_similarity(y_np,pred_np,data_range=1.0)
   
	trainloss.append(total_train_loss/trainSteps)
	validloss.append(total_val_loss/valSteps)
	trainPCC.append(train_ACC/trainSteps)
	validPCC.append(valid_ACC/valSteps)
	trainiou.append(train_iou/trainSteps)
	valiou.append(val_iou/valSteps)
	trainf1.append(train_f1/trainSteps)
	valf1.append(val_f1/valSteps)
	#trainPSNR.append(train_PSNR/trainSteps)
	#validPSNR.append(valid_PSNR/valSteps)
	#trainSSIM.append(train_SSIM/trainSteps)
	#validSSIM.append(valid_SSIM/valSteps)
	
	scheduler.step(total_val_loss/valSteps)
 
	if  val_iou/valSteps > best_acc:
		print(f"Best model found at epoch {epoch}, saving model")
		torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, config.CHECKPOINT_PATHS)
		torch.save(model, config.MODEL_PATH)
		best_acc = val_iou/valSteps 
		stale = 0 
	else:
		stale += 1
		if stale >= 20:
			print(f"No improvment 20 consecutive epochs, early stopping")
			break
	
	savestatis(trainloss, validloss, trainPCC, validPCC, trainiou, valiou, trainf1, valf1)
	current_lr = scheduler.optimizer.param_groups[0]['lr']
	#logging.info(f"Epoch {epoch + 1}/{epoch}, Train Loss: {total_train_loss/trainSteps:.4f}, Val Loss: {total_val_loss/valSteps:.4f}, Train Accuracy: {train_ACC/trainSteps:.4f}, Val Accuracy: {valid_ACC/valSteps:.4f}, Train PSNR: {train_PSNR/trainSteps:.4f}, Val PSNR: {valid_PSNR/valSteps:.4f}, Train SSIM: {train_SSIM/trainSteps:.4f}, Val SSIM: {valid_SSIM/valSteps:.4f},Learning Rate: {current_lr}")
	logging.info(f"Epoch {epoch + 1}/{epoch}, Train Loss: {total_train_loss/trainSteps:.4f}, Val Loss: {total_val_loss/valSteps:.4f}, Train Accuracy: {train_ACC/trainSteps:.4f}, Val Accuracy: {valid_ACC/valSteps:.4f}, Train iou: {train_iou/trainSteps:.4f}, Val iou: {val_iou/valSteps:.4f}, Train f1: {train_f1/trainSteps:.4f}, Val f1: {val_f1/valSteps:.4f},Learning Rate: {current_lr}")
	progress.update(1)


'''Print in terminal
tqdm.write(f"Epoch: {epoch}/{100}")
tqdm.write(f"trainingLoss:{avg_train_loss:.4f}  validationLoss:{avg_val_loss:.4f}")
tqdm.write(f"trainingAccuracy:{avg_train_acc:.4f}  validationAccuracy:{avg_val_acc:.4f}")
tqdm.write("Learning rate: {}".format(current_lr))'''