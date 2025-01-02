# Raneem Ibraheem, 212920896
# Selan Abu Saleh, 212111439

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt

def clean_baby(im):
    # define coordinates for the regions of interest
    regions = [
        np.float32([[6, 20], [6, 130], [111, 130], [111, 20]]),
        np.float32([[78, 162], [133, 245], [245, 160], [147, 117]]),
        np.float32([[181, 4], [121, 50], [176, 120], [249, 70]])
    ]
    dst = np.float32([[0, 0], [0, 255], [255, 255], [255, 0]])
    
    # apply perspective transformation
    transformed_images = [cv2.warpPerspective(im, cv2.getPerspectiveTransform(region, dst), (im.shape[1], im.shape[0])) for region in regions]
    
    # combine images using median
    combined_image = np.median(transformed_images, axis=0).astype(np.uint8)
    
    # apply median blur
    final_image = cv2.medianBlur(combined_image, 7)
    
    return final_image

def clean_windmill(im):
    # perform fourier transform
    f_transform = fftshift(fft2(im))
    
    # zero out specific frequencies
    f_transform[132, 156] = 0
    f_transform[124, 100] = 0
    
    # inverse fourier transform
    cleaned_image = np.abs(ifft2(ifftshift(f_transform)))
    
    return np.uint8(cleaned_image)

def clean_watermelon(im):
    # define high pass filter kernel
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    
    # apply filter
    filtered_image = cv2.filter2D(im, -1, kernel)
    
    return filtered_image

def clean_umbrella(im):
    # fourier transform
    f_transform = fft2(im)
    
    # create mask for deconvolution
    mask = np.zeros_like(im, dtype=np.float32)
    mask[0, 0] = 0.5
    mask[4, 79] = 0.5
    mask = fft2(mask)
    
    # avoid division by zero
    mask[np.abs(mask) < 1e-6] = 1
    
    # deconvolution
    deconvolved = f_transform / mask
    
    # inverse fourier transform
    cleaned_image = np.abs(ifft2(ifftshift(deconvolved)))
    
    return np.uint8(cleaned_image)

def clean_USAflag(im):
    # apply median filter
    cleaned_image = im.copy()
    
    # process first part of the image
    for i in range(1, 92):
        for j in range(160, im.shape[1] - 7):
            pixel_values = im[i, j-7:j+8]
            cleaned_image[i, j] = np.median(pixel_values)
    
    # process second part of the image
    for i in range(92, im.shape[0]):
        for j in range(60, im.shape[1] - 7):
            pixel_values = im[i, j-7:j+8]
            cleaned_image[i, j] = np.median(pixel_values)
    
    return cleaned_image

def clean_house(im):
    # fourier transform
    f_transform = fft2(im)
    
    # create motion blur mask
    mask = np.zeros_like(im, dtype=np.float32)
    mask[0, :10] = 0.1
    mask = fft2(mask)
    
    # avoid division by zero
    mask[np.abs(mask) < 1e-3] = 1
    
    # deconvolution
    deconvolved = f_transform / mask
    
    # inverse fourier transform
    cleaned_image = np.abs(ifft2(ifftshift(deconvolved)))
    
    return cleaned_image

def clean_bears(im):
    # histogram equalization
    equalized_image = cv2.equalizeHist(im)
    
    # adjust brightness and contrast
    alpha, beta = 0.85, -16
    adjusted_image = np.clip(alpha * equalized_image + beta, 0, 255).astype(np.uint8)
    
    return adjusted_image

