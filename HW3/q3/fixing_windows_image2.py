import cv2
import numpy as np
import matplotlib.pyplot as plt

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

cv2.imwrite("fixed_image_b.png", refined_image)
