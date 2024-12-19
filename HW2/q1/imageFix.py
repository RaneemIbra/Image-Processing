# Raneem Ibraheem, 212920896
# Selan Abu Saleh, 212111439

import cv2
import matplotlib.pyplot as plt
import numpy as np

def apply_fix(image, id):
    # to fix the first image we performed a histogram equalization
    if id==1:
        return cv2.equalizeHist(image)

    # to fix the second image we adjusted the brightness and contrast
    elif id==2:
        alpha, beta = 1.35, 25
        return np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    
    # we chose to fix the last image with gamma correction
    elif(id==3):
        gamma = 1.1
        inv_gamma = 1.0 / gamma
        lookup_table = (np.arange(256) ** inv_gamma).clip(0, 255).astype(np.uint8)
        return cv2.LUT(image, lookup_table)

    else:
        raise ValueError(f"Invalid fix_type {id}. Supported values are 1, 2, or 3.")


for i in range(1, 4):
    if i == 1:
        path = f'q1\\{i}.png'
    else:
        path = f'q1\\{i}.jpg'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    fixed_image = apply_fix(image, i)
    plt.imsave(f'{i}_fixed.jpg', fixed_image, cmap='gray', vmin=0, vmax=255)

