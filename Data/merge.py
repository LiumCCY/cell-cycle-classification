import numpy as np
import os
import cv2

'''===Data Directory==='''
bf = "/home/ccy/cellcycle/Data/FUCCI_RFP_GFP/brightfield_png"
pd_mito = "/home/ccy/cellcycle/Data/FUCCI_RFP_GFP/mito_png"
pd_nuclei = "/home/ccy/cellcycle/Data/FUCCI_RFP_GFP/pd_nuclei_1024"

sorted_bf = sorted(os.listdir(bf))
sorted_mito = sorted(os.listdir(pd_mito))
sorted_nuclei = sorted(os.listdir(pd_nuclei))
                     
for i in range(len(sorted_bf)):
    bf_img = cv2.imread(os.path.join(bf, sorted_bf[i]),cv2.IMREAD_GRAYSCALE)
    pd_mito_img = cv2.imread(os.path.join(pd_mito, sorted_mito[i]),cv2.IMREAD_GRAYSCALE)
    pd_nuclei_img = cv2.imread(os.path.join(pd_nuclei, sorted_nuclei[i]),cv2.IMREAD_GRAYSCALE)
    
    bf_img = np.expand_dims(bf_img, axis=-1)
    pd_mito_img = np.expand_dims(pd_mito_img, axis=-1)
    pd_nuclei_img = np.expand_dims(pd_nuclei_img, axis=-1)
    print(bf_img.shape)
    print(pd_mito_img.shape)
    print(pd_nuclei_img.shape)
    
    three_channel_img = np.concatenate((pd_mito_img,bf_img,pd_nuclei_img),axis=-1)
    print(three_channel_img.shape)
    
    cv2.imwrite(os.path.join("/home/ccy/cellcycle/Data/FUCCI_uncropped/bf_mito_nuclei2",sorted_bf[i]),three_channel_img)
    print('saved scence {}, shape:{}'.format(i,three_channel_img.shape))
    

