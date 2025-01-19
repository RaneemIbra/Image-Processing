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
	fourierTransform = fftshift(fft2(image))
	# get the image dimensions
	rows, columns = fourierTransform.shape
	# where to start cropping the image
	cropRow = int((rows * (1 - resize_ratio)) // 2)
	cropCol = int((columns * (1 - resize_ratio)) // 2)
	# crop the image
	croppedImage = fourierTransform[cropRow:cropRow + int(rows * resize_ratio), cropCol:cropCol + int(columns * resize_ratio)]
	# fix the values of the cropped image
	croppedImage = croppedImage * resize_ratio ** 2
	# return the cropped image in spatial domain
	return np.abs(ifft2(ifftshift(croppedImage)))

def scale_up(image, resize_ratio):
	# Your code goes here
	# convert the image to frequency domain
	fourierTransform = fftshift(fft2(image))
	# get the image dimensions
	rows, columns = fourierTransform.shape
	additionalRows, additionalColumns = int(rows * (resize_ratio - 1)) // 2, int(columns * (resize_ratio - 1)) // 2
	# create a zero matrix with the new dimensions
	paddedImage = np.zeros(((additionalRows * 2) + rows, (additionalColumns * 2) + columns), dtype=complex)
	# copy the original image to the new matrix
	paddedImage[additionalRows: additionalRows + rows, additionalColumns:additionalColumns + columns] = fourierTransform
	paddedImage = paddedImage * resize_ratio ** 2
	# return the padded image in spatial domain
	return np.abs(np.real(ifft2(ifftshift(paddedImage))))

def ncc_2d(image, pattern):
	# Your code goes here
	# get the image and pattern dimensions
	imgHeight, imgWidth = image.shape
	patternHeight, patternWidth = pattern.shape
	# calculate the output dimensions
	outputHeight, outputWidth = imgHeight - patternHeight + 1, imgWidth - patternWidth + 1
	# initialize the output matrix
	NCCout = np.zeros((outputHeight, outputWidth))
	# calculate the mean of the pattern
	patternMean = np.mean(pattern)
	for i in range(outputHeight):
		for j in range(outputWidth):
			# get the current window
			window = image[i:i + patternHeight, j:j + patternWidth]
			# calculate the mean of the window
			windowMean = np.mean(window)
			# calculate the cross-correlation
			NCCout[i, j] = np.sum((window - windowMean) * (pattern - patternMean)) / (np.sqrt(np.sum((window - windowMean) ** 2)) * np.sqrt(np.sum((pattern - patternMean) ** 2)))
	return NCCout

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
