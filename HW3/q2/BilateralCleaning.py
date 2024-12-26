# Raneem Ibraheem, 212920896
# Selan Abu Saleh, 212111439

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
	im = im.astype(np.float64)   
	padded_image = np.pad(im, radius, mode='reflect')
	x, y = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
	gaussian_mask_distance = np.exp(-(x**2 + y**2) / (2 * (stdSpatial**2)))
	cleanIm = np.zeros_like(im)
    
	for i in range(radius, padded_image.shape[0] - radius):
		for j in range(radius, padded_image.shape[1] - radius):
			window = padded_image[i-radius:i+radius+1, j-radius:j+radius+1]            
			gaussian_mask_intensity = np.exp(-((window - padded_image[i, j])**2) / (2 * (stdIntensity**2)))
			weights = gaussian_mask_distance * gaussian_mask_intensity
			weights /= np.sum(weights)
			cleanIm[i-radius, j-radius] = np.sum(weights * window)
    
	return np.clip(cleanIm, 0, 255).astype(np.uint8)

# change this to the name of the image you'll try to clean up
original_image_path_taj = 'taj.jpg'
original_image_path_balls = 'NoisyGrayImage.png'
original_image_path_noisy = 'balls.jpg'
image_taj = cv2.imread(original_image_path_taj, cv2.IMREAD_GRAYSCALE)
image_balls = cv2.imread(original_image_path_balls, cv2.IMREAD_GRAYSCALE)
image_noisy = cv2.imread(original_image_path_noisy, cv2.IMREAD_GRAYSCALE)

taj_filtered = clean_Gaussian_noise_bilateral(image_taj, radius=5, stdSpatial=20, stdIntensity=30)
noisy_filtered = clean_Gaussian_noise_bilateral(image_balls, radius=3, stdSpatial=10, stdIntensity=50)
balls_filtered = clean_Gaussian_noise_bilateral(image_noisy, radius=7, stdSpatial=30, stdIntensity=30)

# Displaying the original and filtered images
fig, axes = plt.subplots(3, 2, figsize=(12, 15))
axes[0, 0].imshow(image_taj, cmap='gray')
axes[0, 0].set_title('Original Taj Image')
axes[0, 1].imshow(taj_filtered, cmap='gray')
axes[0, 1].set_title('Filtered Taj Image')

axes[1, 0].imshow(image_balls, cmap='gray')
axes[1, 0].set_title('Original Noisy Gray Image')
axes[1, 1].imshow(noisy_filtered, cmap='gray')
axes[1, 1].set_title('Filtered Noisy Gray Image')

axes[2, 0].imshow(image_noisy, cmap='gray')
axes[2, 0].set_title('Original Balls Image')
axes[2, 1].imshow(balls_filtered, cmap='gray')
axes[2, 1].set_title('Filtered Balls Image')

cv2.imwrite('Filtered_Taj_Image.jpg', taj_filtered)
cv2.imwrite('Filtered_Noisy_Gray_Image.jpg', noisy_filtered)
cv2.imwrite('Filtered_Balls_Image.jpg', balls_filtered)

plt.tight_layout()
plt.show()
