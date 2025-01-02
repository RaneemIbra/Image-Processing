# Raneem Ibraheem, 212920896
# Selan Abu Saleh, 212111439

# Please replace the above comments with your names and ID numbers in the same format.

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt


def clean_baby(im):
	# Your code goes here
    return im

def clean_windmill(im):
	# Your code goes here
    f = fft2(im)
    fshift = fftshift(f)
    rows, cols = im.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols), dtype=np.uint8)
    mask[crow - 5:crow + 5, :] = 0
    fshift *= mask
    f_ishift = ifftshift(fshift)
    restored = np.abs(ifft2(f_ishift))
    return np.clip(restored, 0, 255).astype(np.uint8)

def clean_watermelon(im):
	# Your code goes here
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(im, -1, kernel)
    return sharpened

def clean_umbrella(im):
	# Your code goes here
    rows, cols = im.shape
    shift = 10
    aligned = np.roll(im, -shift, axis=1)
    restored = (im.astype(np.float32) - aligned.astype(np.float32)) / 2
    return np.clip(restored, 0, 255).astype(np.uint8)

def clean_USAflag(im):
	# Your code goes here
    cleaned = cv2.medianBlur(im, 3)
    return cleaned

def clean_house(im):
	# Your code goes here
    psf = np.zeros_like(im, dtype=np.float32)
    psf[im.shape[0] // 2, :15] = 1
    psf /= psf.sum()
    blurred = fft2(im)
    psf_fft = fft2(psf, s=im.shape)
    restored_fft = blurred / (psf_fft + 1e-3)
    restored = np.abs(ifft2(restored_fft))
    return np.clip(restored, 0, 255).astype(np.uint8)

def clean_bears(im):
	# Your code goes here
	gamma = 1.2
	im_float = im.astype(np.float32) / 255
	enhanced = np.power(im_float, gamma) * 255
	return np.clip(enhanced, 0, 255).astype(np.uint8)
