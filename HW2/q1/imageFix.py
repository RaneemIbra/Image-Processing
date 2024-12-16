# Raneem Ibraheem, 212920896
# Selan Abu Saleh, 212111439

import cv2
import matplotlib.pyplot as plt
import numpy as np

def apply_fix(image, id):
    # i chose for fixing the first image histogram equalization
    if(id==1):
        equalized_image = cv2.equalizeHist(image)
        return equalized_image

    # i chose for fixing the second image  brightness and contrast stretching with alpha = 1.35 and beta = 25
    if(id==2):
        alpha = 1.35
        beta = 25
        stretched_image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
        return stretched_image
    
    # i chose for fixing the third image  gamma correction
    if(id==3):
        gamma = 1.1
        corrected_image = np.clip(image ** (1 / gamma), 0, 255).astype(np.uint8)
        return corrected_image


for i in range(1, 4):
    if i == 1:
        path = f'q1\\{i}.png'  # First image is assumed to be a PNG file
    else:
        path = f'q1\\{i}.jpg'  # The other images are assumed to be JPG files
    
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    fixed_image = apply_fix(image, i)
    plt.imsave(f'{i}_fixed.jpg', fixed_image, cmap='gray', vmin=0, vmax=255)

