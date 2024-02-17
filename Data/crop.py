import os 
from PIL import Image
import cv2
import numpy as np

bf_mtio_nuclei = "/home/ccy/cellcycle/Data/FUCCI_uncropped/bf_mito(REAL)_nuclei"
gfp = "/home/ccy/cellcycle/Data/FUCCI_uncropped/GFP_png"
rfp = "/home/ccy/cellcycle/Data/FUCCI_uncropped/RFP_png"

def crop(input_path,output_path_root,save_file_name):
    
    file = os.listdir(input_path)
    print(len(file))
    for i in range(len(file)):
        
        image = cv2.imread(os.path.join(input_path, file[i]))
        if image is None:
            raise Exception("Failed to read image")
        
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        print(image.shape)
        
        if (image.shape == (1024,1024,3)):
            # 定义切割后的图像尺寸
            crop_size = 512
            # 切割图像并保存
            for h in range(2):
                for w in range(2):
                    # 计算切割区域的左上角坐标和右下角坐标
                    x1 = h * crop_size
                    y1 = w * crop_size
                    x2 = x1 + crop_size
                    y2 = y1 + crop_size
                    # 切割图像
                    crop_image = image[y1:y2, x1:x2]
                    # 保存切割后的图像
                    os.chdir(output_path_root)
                    output_path = os.path.join(output_path_root, save_file_name)
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    cv2.imwrite(os.path.join(output_path, "{}_{}_{}.png".format(os.path.splitext(file[i])[0],h,w)),crop_image)
            
        elif(image.shape == (512,512)):
            print(file[i])
        
crop(bf_mtio_nuclei,"/home/ccy/cellcycle/Data/FUCCI_SR","bf_mito_nuclei_cropped")

