import os 
from PIL import Image, ImageFilter
import shutil

input_file_list = "/home/ccy/cellcycle/data/FUCCI_RFP_GFP"
output_file_list_mito ="/home/ccy/cellcycle/data/FUCCI_RFP_GFP/mito"
output_file_list_dye ="/home/ccy/cellcycle/data/FUCCI_RFP_GFP/dye"
output_file_list_brightfield ="/home/ccy/cellcycle/data/FUCCI_RFP_GFP/brightfield"
output_file_list_nuclei ="/home/ccy/cellcycle/data/FUCCI_RFP_GFP/nuclei"

for index, filename in enumerate(sorted(os.listdir(input_file_list))):
    input_file_path = os.path.join(input_file_list,filename)
    if index%4 == 0:
        output_path = output_file_list_mito
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        shutil.move(input_file_path, output_path)
    elif index%4 == 1:
        output_path = output_file_list_dye
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        shutil.move(input_file_path, output_path)
    elif index%4 == 2:
        output_path = output_file_list_brightfield
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        shutil.move(input_file_path, output_path)
    elif index%4 == 3:
        output_path = output_file_list_nuclei
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        shutil.move(input_file_path, output_path)
    
