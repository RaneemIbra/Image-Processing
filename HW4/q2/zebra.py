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
# first get the width and height of the image
Height, Width = image.shape
# Section A:
# compute the fourier transform of the image and the shifted fourier transform and the fourier spectrum
fourier_transform = fft2(image)
fourier_transform_shifted = fftshift(fourier_transform)
fourier_spectrum = np.log(1 + np.abs(fourier_transform_shifted))

plt.figure(figsize=(10,10))
plt.subplot(321)
plt.title('Original Grayscale Image')
plt.imshow(image, cmap='gray')

plt.subplot(322)
plt.title('Fourier Spectrum')
plt.imshow(fourier_spectrum, cmap='gray')

# Section B:
# zero pad the fourier transform and compute the inverse fourier transform to restore the image
padded_fourier_transform = np.zeros((2*Height, 2*Width), dtype=np.complex64)
centerH, centerW = Height // 2, Width // 2
padded_fourier_transform[Height-centerH : Height+centerH, Width-centerW : Width+centerW] = fourier_transform_shifted

inverse_shifted = ifftshift(padded_fourier_transform)
restored_spatial = ifft2(inverse_shifted)
scaled_image = np.abs(restored_spatial)
scaled_image *= 4

# clip the image to be in the range of 0-255 and convert it to 8-bit
scaled_image_8bit = np.clip(scaled_image, 0, 255).astype(np.uint8)
log_spectrum_padded = np.log(1 + np.abs(padded_fourier_transform))

plt.subplot(323)
plt.title('Fourier Spectrum Zero Padding')
plt.imshow(log_spectrum_padded, cmap='gray')

plt.subplot(324)
plt.title('Two Times Larger Grayscale Image')
plt.imshow(scaled_image_8bit, cmap='gray')

# Section C:
# scale the image by 0.5 and create four copies of it and compute the fourier transform and the fourier spectrum
half_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
built_image = np.tile(half_image, (2, 2))
fourier_transform_4copies = fft2(built_image)
fourier_transform_shifted_4copies = fftshift(fourier_transform_4copies)
fourier_spectrum_4copies = np.log(1 + np.abs(fourier_transform_shifted_4copies))

plt.subplot(325)
plt.title('Fourier Spectrum Four Copies')
plt.imshow(fourier_spectrum_4copies, cmap='gray')

plt.subplot(326)
plt.title('Four Copies Grayscale Image')
plt.imshow(built_image, cmap='gray')
plt.show()
# plt.savefig('zebra_scaled.png')