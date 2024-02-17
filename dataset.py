import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transform import *
from transform_alb import *
import cv2
import os 
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt

class Mydataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------
        self.images = sorted(os.listdir(str(image_dir)))
        self.labels = sorted(os.listdir(str(label_dir)))
        assert len(self.images) == len(self.labels)
        self.transform =transform 
        
        #self.image_path = image_dir
        #self.label_path = label_dir
        
        self.images_and_label = []
        for i in range(len(self.images)):
            self.images_and_label.append((str(image_dir)+'/'+str(self.images[i]),
                                          str(label_dir)+'/'+str(self.labels[i])))

    def __getitem__(self, index):
        
        image_path, label_path = self.images_and_label[index]
        
        image = Image.open(image_path).convert("L")# PIL: RGB, OpenCV: BGR
        image = np.array(image)  # 將圖片轉換為 NumPy 陣列
        image = image.astype(np.uint8)  # 將圖片數據類型轉換為 uint8
        image = Image.fromarray(image)
        
        label = Image.open(label_path).convert("L")
        label = np.array(label)  # 將圖片轉換為 NumPy 陣列
        label =  label.astype(np.uint8)  # 將圖片數據類型轉換為 uint8
        label = Image.fromarray(label)
        
        if self.transform is not None:
            image, label = self.transform((image,label))
        return image, label
    
    def __len__(self):
        return len(self.images)
    
def label_preprocess(label_path):
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.medianBlur(label,5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    #label = cv2.erode(label,kernel,iteration=1)
    threshold_value = 20  # You can adjust this threshold value as needed
    _, label = cv2.threshold(label, threshold_value, 255, cv2.THRESH_BINARY)
    label = cv2.dilate(label,kernel,iterations=8)
    label = label.astype(np.uint8)
    cv2.imwrite("label.png", label)
    

class FucciDataset(Dataset):
    def __init__(self, image_dir, label_dir,transform=None):
        
        self.images = sorted(os.listdir(str(image_dir)))
        self.labels = sorted(os.listdir(str(label_dir)))
        assert len(self.images) == len(self.labels)
        self.transform =transform 
        
        self.images_and_label = []
        for i in range(len(self.images)):
            self.images_and_label.append((str(image_dir)+'/'+str(self.images[i]),
                                          str(label_dir)+'/'+str(self.labels[i])))

    def __getitem__(self, index):
        
        image_path, label_path = self.images_and_label[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
        image = image.astype(np.uint8)
        print(image.shape)
        cv2.imwrite("image.png", image)
        
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.IMREAD_COLOR)
        label = label.astype(np.uint8)
        print(label.shape)
        cv2.imwrite("label.png", label)
        
        if self.transform is not None:
            transformed_image = self.transform(image=image, mask=label)
            image = transformed_image['image']
            label = transformed_image['mask']
        return image, label
    
    def __len__(self):
        return len(self.images)
    