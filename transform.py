import torch
from torchvision import transforms
import numpy as np
import torchvision.transforms as transforms
import config
from PIL import ImageFilter
import cv2

def specific_normalize_image(image):
    min_value = np.min(image)
    max_value = np.max(image)
    normalized_image = (image - min_value) / (max_value - min_value)
    return normalized_image

def zero_normalize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    normalized_image = (image - mean) / std
    return normalized_image

class ZScoreNormalize(object):
    def __call__(self, tensor):
        # 将Tensor转换为NumPy数组
        tensor_array = tensor.numpy()
        
        # 计算均值和标准差
        mean = tensor_array.mean()
        std = tensor_array.std()
        
        # Z-score标准化
        normalized_tensor_array = (tensor_array - mean) / std
        
        # 将NumPy数组转换为Tensor
        normalized_tensor = torch.from_numpy(normalized_tensor_array)
        
        return normalized_tensor

class RandomTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x, y = sample
        seed = torch.randint(0, 2**32, size=(1,)).item()
        torch.manual_seed(seed)
        x = self.transform(x)
        torch.manual_seed(seed)
        y = self.transform(y)
        return x, y

        #image, label = sample
        #seed = torch.randint(0, 2**32, size=(1,)).item()
        #torch.manual_seed(seed)
        #transformed = self.transform(image=image, mask=label)
        #image = transformed["image"]
        #label = transformed["mask"]
        #return image, label
        
def custom_box_blur(image, kernel_size=3):
    return image.filter(ImageFilter.BoxBlur(kernel_size))

train_transform = transforms.Compose(
    [   
        transforms.Resize((config.INPUT_IMAGE_LENGTH, config.INPUT_IMAGE_WIDTH)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        #transforms.CenterCrop((512,512)),
        transforms.RandomCrop(512, padding=5, pad_if_needed=False, padding_mode='edge'),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [   
        transforms.Resize((config.INPUT_IMAGE_LENGTH, config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor(),
    ]
)

def postprocess_image(x):
    contrast_enhancer = transforms.ColorJitter(contrast=2.0) 
    x = contrast_enhancer(x)
    return x

def threshold_pre_processing(label, threshold):
    # 复制输入数组以防止修改原始数据
    processed_label = label.copy()
    
    # 使用OpenCV的dilate函数膨胀图像
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_label = cv2.dilate(processed_label, kernel)
    
    for i in range(processed_label.shape[0]):
        for j in range(processed_label.shape[1]):
            if processed_label[i, j] >= threshold:
                if np.any(dilated_label[i-3:i+4, j-3:j+4] >= threshold):
                    processed_label[i, j] = 255
                else:
                    processed_label[i, j] = 0
    return processed_label

def threshold_post_processing_torch(label_tensor, threshold):
    # 使用 torch.where 进行阈值处理，将大于等于阈值的像素设置为1，小于阈值的像素设置为0
    processed_label = torch.where(label_tensor >= threshold, torch.ones_like(label_tensor), torch.zeros_like(label_tensor))
    return processed_label