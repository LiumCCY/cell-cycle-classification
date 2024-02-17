import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2


train_transform = A.Compose([
    A.Resize(width=1024, height=1024),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
    A.Rotate(limit=30, p=0.3),
    ToTensorV2()
], is_check_shapes=False)

val_transform = A.Compose([
    A.Resize(width=1024, height=1024),
    ToTensorV2()
])