# Raneem Ibraheem, 212920896
# Selan Abu Saleh, 212111439

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# the function that will apply a bilateral filter to the images
def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
	# first convert the image into a float type
	im = im.astype(np.float64)
	# pad the image with reflected pixels
	padded_image = np.pad(im, radius, mode='reflect')
	# here we create a gaussian mask, first we create a 2d grid that will be the window
	x, y = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
	# then we calculate the spatial gaussian mask using euclidean distance
	gaussian_mask_distance = np.exp(-(x**2 + y**2) / (2 * (stdSpatial**2)))
	# now create an empty image that will store the filtered image
	cleanIm = np.zeros_like(im)
    # iterate over the pixels
	for i in range(radius, padded_image.shape[0] - radius):
		for j in range(radius, padded_image.shape[1] - radius):
			# extract a window with the same size as the mask
			window = padded_image[i-radius:i+radius+1, j-radius:j+radius+1]
			# calculate the intensity using the formula given in the HW instructions
			gaussian_mask_intensity = np.exp(-((window - padded_image[i, j])**2) / (2 * (stdIntensity**2)))
			# combine the intensity mask and the spatial mask to compute the final weights
			weights = gaussian_mask_distance * gaussian_mask_intensity
			# now normalize the weights
			weights /= np.sum(weights)
			# assing the result of the sum to the corresponding pixel
			cleanIm[i-radius, j-radius] = np.sum(weights * window)
    # ensure that the pixel values are in a valid range and return the image
	return np.clip(cleanIm, 0, 255).astype(np.uint8)

# change this to the name of the image you'll try to clean up
# first we read the images
original_image_path_taj = 'taj.jpg'
original_image_path_balls = 'NoisyGrayImage.png'
original_image_path_noisy = 'balls.jpg'
image_taj = cv2.imread(original_image_path_taj, cv2.IMREAD_GRAYSCALE)
image_balls = cv2.imread(original_image_path_balls, cv2.IMREAD_GRAYSCALE)
image_noisy = cv2.imread(original_image_path_noisy, cv2.IMREAD_GRAYSCALE)

# then apply the bilateral filter on them (params choice in explained in the pdf)
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
