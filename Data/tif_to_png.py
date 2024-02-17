import os 
from PIL import Image
import cv2
import numpy as np



bright_field = "/home/ccy/cellcycle/data/FUCCI_RFP_GFP/brightfield"

mito = "/home/ccy/cellcycle/data/FUCCI_RFP_GFP/mito"

n = "/home/ccy/cellcycle/data/FUCCI_RFP_GFP/nuclei"

dye = "/home/ccy/cellcycle/data/FUCCI_RFP_GFP/dye"




def tif_to_png(input_path,output_path_root,save_file_name):
    
    file = os.listdir(input_path)
    print(len(file))
    for i in range(len(file)):
        try:
            image = cv2.imread(os.path.join(input_path, file[i]))
            if image is None:
                raise Exception("Failed to read image")

            # 进行图像处理
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            ##blurred = cv2.blur(image, (3, 3))
            #corrected_image = blurred / 255.0

            # Gamma校正
            #corrected_image = np.power(normalized_image, 0.5)

            # 重新缩放像素值到 [0, 255]
            #corrected_image = (corrected_image * 255).astype(np.uint8)
   
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)
            
            
            os.chdir(output_path_root)
            output_path = os.path.join(output_path_root, save_file_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            cv2.imwrite(os.path.join(output_path, "{}.png".format(os.path.splitext(file[i])[0])),image)
            
        except Exception as e:
            print("Failed to process image {}: {}".format(file[i], e))
        finally:
            # 释放内存并关闭文件
            if image is not None:
                del image

tif_to_png(bright_field, "/home/ccy/cellcycle/data/FUCCI_RFP_GFP", "brightfield_png")
tif_to_png(n, "/home/ccy/cellcycle/data/FUCCI_RFP_GFP", "nuclei_png")
tif_to_png(mito, "/home/ccy/cellcycle/data/FUCCI_RFP_GFP", "mito_png")
tif_to_png(dye, "/home/ccy/cellcycle/data/FUCCI_RFP_GFP", "dye_png")

#tif_to_png(train_mito, "/home/ccy/cellcycle/data/split/train", "mito_png1")
#tif_to_png(val_mito, "/home/ccy/cellcycle/data/split/val", "mito_png1")
#tif_to_png(test_mito, "/home/ccy/cellcycle/data/split/test", "mito_png1")
#
#tif_to_png(train_n, "/home/ccy/cellcycle/data/split/train", "n_png1")
#tif_to_png(val_n, "/home/ccy/cellcycle/data/split/val", "n_png1")
#tif_to_png(test_n, "/home/ccy/cellcycle/data/split/test", "n_png1")
