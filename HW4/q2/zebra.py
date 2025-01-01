# Raneem Ibraheem, 212920896
# Selan Abu Saleh, 212111439

# Please replace the above comments with your names and ID numbers in the same format.

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt

image_path = 'zebra.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Your code goes here
# section a, where we get the fourier transform of the image and then show it
fourier_transform = fft2(image)
shifted_fourier_transform = fftshift(fourier_transform)
fourier_spectrum = np.log(1 + np.abs(shifted_fourier_transform))

plt.figure(figsize=(10,10))
plt.subplot(321)
plt.title('Original Grayscale Image')
plt.imshow(image, cmap='gray')

plt.subplot(322)
plt.title('Fourier Spectrum')
plt.imshow(fourier_spectrum, cmap='gray')

# section b, where we get the image height and width and then padd it
# then we apply the same steps as before
Height, Width = image.shape
padded_image = np.pad(image, ((0, Height), (0, Width)), mode='constant')
padded_fourier_transform = fft2(padded_image)
shifted_padded_fourier_transform = fftshift(padded_fourier_transform)
padded_fourier_spectrum = np.log(1 + np.abs(shifted_padded_fourier_transform))

plt.subplot(323)
plt.title('Fourier Spectrum Zero Padding')
plt.imshow(padded_fourier_spectrum, cmap='gray')

plt.subplot(324)
plt.title('Two Times Larger Grayscale Image')
plt.imshow(padded_image, cmap='gray')

# section c, where we use the fourier transform scale formula to generate an image with size (2*H, 2*W) containing 4 copies of the zebra
scaled_fourier_transform = np.zeros((2*Height, 2*Width), dtype=complex)
scaled_fourier_transform[:Height//2, :Width//2] = shifted_fourier_transform[Height//2:, Width//2:]
scaled_fourier_transform[:Height//2, -Width//2:] = shifted_fourier_transform[Height//2:, :Width//2]
scaled_fourier_transform[-Height//2:, :Width//2] = shifted_fourier_transform[:Height//2, Width//2:]
scaled_fourier_transform[-Height//2:, -Width//2:] = shifted_fourier_transform[:Height//2, :Width//2]

# then we restore the image using the inverse fourier transform
scaled_image = np.abs(ifft2(ifftshift(scaled_fourier_transform)))
scaled_fourier_spectrum = np.log(1 + np.abs(scaled_fourier_transform))

plt.subplot(325)
plt.title('Fourier Spectrum Four Copies')
plt.imshow(scaled_fourier_spectrum, cmap='gray')

plt.subplot(326)
plt.title('Four Copies Grayscale Image')
plt.imshow(scaled_image, cmap='gray')
plt.show()
# plt.savefig('zebra_scaled.png')