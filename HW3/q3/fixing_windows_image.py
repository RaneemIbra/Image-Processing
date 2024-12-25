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

cv2.imwrite("fixed_image_a.jpg", bilateral_blur)
