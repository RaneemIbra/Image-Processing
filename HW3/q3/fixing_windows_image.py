# Raneem Ibraheem, 212920896
# Selan Abu Saleh, 212111439

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("broken.jpg", cv2.IMREAD_GRAYSCALE)
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 1.5)
median_blur = cv2.medianBlur(image, 5)
bilateral_blur = cv2.bilateralFilter(image, 9, 75, 75)

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
plt.imshow(bilateral_blur, cmap="gray")
plt.show()

cv2.imwrite("fixed_image_a_bilateral.jpg", bilateral_blur)
cv2.imwrite("fixed_image_a_gaussian.jpg", gaussian_blur)
cv2.imwrite("fixed_image_a_median.jpg", median_blur)

noised_images = np.load("noised_images.npy")
average_image = np.mean(noised_images, axis=0).astype(np.uint8)
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