import sys
sys.path.append('/home/ccy/cellcycle')
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from PIL import Image
from transform_alb import *
from transform import *


def prepare_plot(origImage, origMask, predMask):
	
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	
	ax[0].imshow(origImage, cmap="gray")
	ax[1].imshow(origMask,cmap="gray")
	ax[2].imshow(predMask,cmap="gray")
	
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	
	filename = str("C_"+sorted(os.listdir(test_mask_path))[i])
	save_path = os.path.join(save_file,filename)
	figure.tight_layout()
	figure.show()
	figure.savefig(save_path)

def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		image = Image.open(imagePath)
		#image = cv2.imread(imagePath)
		orig = image.copy()
		image = test_transform(image)
		image = image.numpy()
  
		filename = sorted(os.listdir(test_mask_path))[i]
  
		''' 要畫compare'''
		groundTruthPath = os.path.join(test_mask_path,filename)
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_WIDTH,config.INPUT_IMAGE_LENGTH))
		
		image = np.expand_dims(image, 0).astype("float32")
		image = torch.from_numpy(image).to(config.DEVICE)
		predMask = model(image)[0].squeeze()
		predMask = predMask.cpu().numpy()
		predMask = predMask*255
		predMask = predMask.astype(np.uint8)
	
		save_path = os.path.join(save_file,filename)
		plt.imsave(save_path, predMask, cmap = "gray")
  
		prepare_plot(orig, gtMask, predMask)
  

print("[INFO] loading up test image paths...")
file_list = "/home/ccy/cellcycle/Data/FUCCI_SR/test/bf_mito_nuclei"
test_mask_path = "/home/ccy/cellcycle/Data/FUCCI_SR/test/GFP"
save_file = "/home/ccy/cellcycle/Predict_GFP_RFP/output_GFP/prediction"

for i in range(len(os.listdir(file_list))):
    imagePaths = os.path.join(file_list,sorted(os.listdir(file_list))[i])
    net = torch.load(config.MODEL_PATH).to(config.DEVICE)
    make_predictions(net, imagePath=imagePaths)
