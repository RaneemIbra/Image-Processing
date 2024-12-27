# Raneem Ibraheem, 212920896
# Selan Abu Saleh, 212111439

import cv2
import numpy as np
import matplotlib.pyplot as plt

# same function from the question before, nothing new in it
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

# read the image then apply the filter on it
image = cv2.imread("broken.jpg", cv2.IMREAD_GRAYSCALE)
bilateral_filtered_image = clean_Gaussian_noise_bilateral(image, radius=5, stdSpatial=20, stdIntensity=30)
# tested other blurring techniques to see which gives the best result
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 1.5)
median_blur = cv2.medianBlur(image, 5)

# the following lines are just to display or store the images, nothing fancy
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.subplot(2, 2, 2)
plt.title("Gaussian Blurred")
plt.imshow(gaussian_blur, cmap="gray")
plt.subplot(2, 2, 3)
plt.title("Median Filtered")
plt.imshow(median_blur, cmap="gray")
plt.subplot(2, 2, 4)
plt.title("Bilateral Filtered")
plt.imshow(bilateral_filtered_image, cmap="gray")
plt.show()

cv2.imwrite("fixed_image_a_bilateral.jpg", bilateral_filtered_image)
cv2.imwrite("fixed_image_a_gaussian.jpg", gaussian_blur)
cv2.imwrite("fixed_image_a_median.jpg", median_blur)

# here we load the images from the npy file
noised_images = np.load("noised_images.npy")
# get the average image by calculating the mean of all the images
average_image = np.mean(noised_images, axis=0).astype(np.uint8)
# then apply the filter to find the clean image
refined_image = cv2.GaussianBlur(average_image, (5, 5), 1.5)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Averaged Image")
plt.imshow(average_image, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Refined Image (Gaussian Blur)")
plt.imshow(refined_image, cmap="gray")
plt.show()

cv2.imwrite("fixed_image_npy.png", refined_image)