# Raneem Ibraheem, 212920896
# Selan Abu Saleh, 212111439

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
	im = im.astype(np.float64)   
	padded_im = np.pad(im, radius, mode='reflect')
	x, y = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
	gs = np.exp(-(x**2 + y**2) / (2 * (stdSpatial**2)))
	cleanIm = np.zeros_like(im)
    
	for i in range(radius, padded_im.shape[0] - radius):
		for j in range(radius, padded_im.shape[1] - radius):
			window = padded_im[i-radius:i+radius+1, j-radius:j+radius+1]            
			gi = np.exp(-((window - padded_im[i, j])**2) / (2 * (stdIntensity**2)))
			weights = gs * gi
			weights /= np.sum(weights)
			cleanIm[i-radius, j-radius] = np.sum(weights * window)
    
	return np.clip(cleanIm, 0, 255).astype(np.uint8)

# change this to the name of the image you'll try to clean up
original_image_path = 'NoisyGrayImage.png'
image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

clear_image_b = clean_Gaussian_noise_bilateral(image, 3, 10, 25)

plt.subplot(121)
plt.imshow(image, cmap='gray')

plt.subplot(122)
plt.imshow(clear_image_b, cmap='gray')

plt.show()
