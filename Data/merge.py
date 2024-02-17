import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

'''
bf = "/home/ccy/cellcycle/Data/FUCCI_RFP_GFP/brightfield_png"
pd_mito = "/home/ccy/cellcycle/Data/FUCCI_RFP_GFP/mito_png"
pd_nuclei = "/home/ccy/cellcycle/Data/FUCCI_RFP_GFP/pd_nuclei_1024"

sorted_bf = sorted(os.listdir(bf))
sorted_mito = sorted(os.listdir(pd_mito))
sorted_nuclei = sorted(os.listdir(pd_nuclei))

'''
gfp_image = '/home/ccy/cellcycle/Data/FUCCI_uncropped/GFP_png'
rfp_image = '/home/ccy/cellcycle/Data/FUCCI_uncropped/RFP_png'
sorted_gfp = sorted(os.listdir(gfp_image))
sorted_rfp = sorted(os.listdir(rfp_image))


                     
for i in range(len(sorted_rfp)):
    
    '''
    bf_img = cv2.imread(os.path.join(bf, sorted_bf[i]),cv2.IMREAD_GRAYSCALE)
    pd_mito_img = cv2.imread(os.path.join(pd_mito, sorted_mito[i]),cv2.IMREAD_GRAYSCALE)
    pd_nuclei_img = cv2.imread(os.path.join(pd_nuclei, sorted_nuclei[i]),cv2.IMREAD_GRAYSCALE)
    
    bf_img = np.expand_dims(bf_img, axis=-1)
    pd_mito_img = np.expand_dims(pd_mito_img, axis=-1)
    pd_nuclei_img = np.expand_dims(pd_nuclei_img, axis=-1)
    print(bf_img.shape)
    print(pd_mito_img.shape)
    print(pd_nuclei_img.shape)
    merged_img = np.concatenate((pd_mito_img,bf_img,pd_nuclei_img),axis=-1)
    
    '''
    gfp_img = cv2.imread(os.path.join(gfp_image, sorted_gfp[i]),cv2.IMREAD_GRAYSCALE)
    rfp_img = cv2.imread(os.path.join(rfp_image, sorted_rfp[i]),cv2.IMREAD_GRAYSCALE)
    
    # Create binary masks.
    thresh = 50  # You may need to adjust this threshold.
    gfp_mask = cv2.threshold(gfp_img, thresh, 255, cv2.THRESH_BINARY)[1]
    rfp_mask = cv2.threshold(rfp_img, thresh, 255, cv2.THRESH_BINARY)[1]
    
    # Create a kernel for dilation. A 5x5 kernel will affect the 5x5 neighborhood.
    kernel = np.ones((3, 3), np.uint8)
    gfp_mask = cv2.dilate(np.uint8(gfp_mask), kernel, iterations=1)
    rfp_mask = cv2.dilate(np.uint8(rfp_mask), kernel, iterations=1)

    # Create a blank RGB image.
    rgb_image = np.zeros((gfp_img.shape[0], gfp_img.shape[1], 3), dtype=np.uint8)

    # Assign the GFP mask to the green channel and the RFP mask to the red channel.
    rgb_image[:, :, 1] = gfp_mask
    rgb_image[:, :, 2] = rfp_mask

    # Handle overlap - for overlap we set both red and green to create yellow.
    overlap = np.logical_and(gfp_mask == 255, rfp_mask == 255)
    # Dilate the overlap area.
    dilated_overlap = cv2.dilate(np.uint8(overlap), kernel, iterations=1)

    # Assign yellow color to the dilated overlap areas in the RGB image.
    rgb_image[dilated_overlap, 0] = 255  # Red
    rgb_image[dilated_overlap, 1] = 255  # Green
    #rgb_image[dilated_overlap, 2] = 0  # Blue channel set to 0

    
    cv2.imwrite(os.path.join("/home/ccy/cellcycle/Data/FUCCI_uncropped/label_merged",sorted_gfp[i]),rgb_image)
    print('saved scence {}, shape:{}'.format(i,rgb_image.shape))
    

