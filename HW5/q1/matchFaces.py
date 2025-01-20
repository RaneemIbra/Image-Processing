# Raneem Ibraheem, 212920896
# Selan Abu Saleh, 212111439

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

import warnings
warnings.filterwarnings("ignore")

def scale_down(image, resize_ratio):
	# Your code goes here
	# convert the image to frequency domain
	freq_image = fftshift(fft2(image))
    # calculate cropping indices
	rows, cols = freq_image.shape
	crop_start_row = int((1 - resize_ratio) * rows // 2)
	crop_start_col = int((1 - resize_ratio) * cols // 2)
	crop_end_row = crop_start_row + int(rows * resize_ratio)
	crop_end_col = crop_start_col + int(cols * resize_ratio)
    # crop and scale the frequency coefficients
	cropped_freq = freq_image[crop_start_row:crop_end_row, crop_start_col:crop_end_col] * (resize_ratio ** 2)
    
    # transform back to the spatial domain
	return np.abs(ifft2(ifftshift(cropped_freq)))

def scale_up(image, resize_ratio):
	# Your code goes here
	# convert the image to frequency domain
	freq_image = fftshift(fft2(image))
    # calculate dimensions for padding
	rows, cols = freq_image.shape
	pad_rows = int((resize_ratio - 1) * rows // 2)
	pad_cols = int((resize_ratio - 1) * cols // 2)
    # create a zero-padded frequency array
	padded_freq = np.zeros((rows + 2 * pad_rows, cols + 2 * pad_cols), dtype=complex)
	padded_freq[pad_rows:pad_rows + rows, pad_cols:pad_cols + cols] = freq_image * (resize_ratio ** 2)
    # Transform back to the spatial domain
	return np.abs(ifft2(ifftshift(padded_freq)))

def ncc_2d(image, pattern):
	# Your code goes here
	# get the image and pattern dimensions
    img_height, img_width = image.shape
    pat_height, pat_width = pattern.shape
    # calculate output dimensions for the NCC result
    output_height = img_height - pat_height + 1
    output_width = img_width - pat_width + 1
    # initialize the NCC result array
    ncc_result = np.zeros((output_height, output_width))
    # precompute the mean and normalized pattern
    pattern_mean = np.mean(pattern)
    normalized_pattern = pattern - pattern_mean
    pattern_norm = np.sqrt(np.sum(normalized_pattern ** 2))
    # compute NCC for each window in the image
    for i in range(output_height):
        for j in range(output_width):
            # extract the current window from the image
            window = image[i:i + pat_height, j:j + pat_width]
            # compute mean and normalized window
            window_mean = np.mean(window)
            normalized_window = window - window_mean
            window_norm = np.sqrt(np.sum(normalized_window ** 2))
            # calculate the NCC value
            ncc_result[i, j] = np.sum(normalized_window * normalized_pattern) / (window_norm * pattern_norm)

    return ncc_result

# a function to threshold the NCC values
def Thresholding(NCC, threshold):
	exceeding = NCC > threshold
	indices = np.argwhere(exceeding)
	return indices

def display(image, pattern):
	
	plt.subplot(2, 3, 1)
	plt.title('Image')
	plt.imshow(image, cmap='gray')
		
	plt.subplot(2, 3, 3)
	plt.title('Pattern')
	plt.imshow(pattern, cmap='gray', aspect='equal')
	
	ncc = ncc_2d(image, pattern)
	
	plt.subplot(2, 3, 5)
	plt.title('Normalized Cross-Correlation Heatmap')
	plt.imshow(ncc ** 2, cmap='coolwarm', vmin=0, vmax=1, aspect='auto') 
	
	cbar = plt.colorbar()
	cbar.set_label('NCC Values')
		
	plt.show()

def draw_matches(image, matches, pattern_size):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	for point in matches:
		y, x = point
		top_left = (int(x - pattern_size[1]/2), int(y - pattern_size[0]/2))
		bottom_right = (int(x + pattern_size[1]/2), int(y + pattern_size[0]/2))
		cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 1)
	
	plt.imshow(image, cmap='gray')
	plt.show()
	
	cv2.imwrite(f"{CURR_IMAGE}_result.jpg", image)



CURR_IMAGE = "students"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

############# DEMO #############
display(image, pattern)





############# Students #############

image_scaled = scale_up(image, 1.45)
image_scaled = image_scaled - cv2.GaussianBlur(image_scaled, (15, 15), 7) + 128
pattern_scaled =  scale_down(pattern, 0.8)
pattern_scaled = pattern_scaled - cv2.GaussianBlur(pattern_scaled, (19, 19), 5) + 128

display(image_scaled, pattern_scaled)

ncc = ncc_2d(image_scaled, pattern_scaled)
real_matches = 	Thresholding(ncc, 0.48) * (1 / 1.45)

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:,0] += pattern_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:,1] += pattern_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.

draw_matches(image, real_matches, pattern_scaled.shape)	# if pattern was not scaled, replace this with "pattern"



CURR_IMAGE = "thecrew"
image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

############# Crew #############

image_scaled = scale_up(image, 1.35)
pattern_scaled =  scale_down(pattern, 0.3)

display(image_scaled, pattern_scaled)

ncc = ncc_2d(image_scaled, pattern_scaled)
real_matches = Thresholding(ncc, 0.45) * (1 / 1.35)

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:,0] += pattern_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:,1] += pattern_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.

draw_matches(image, real_matches, pattern_scaled.shape)	# if pattern was not scaled, replace this with "pattern"
