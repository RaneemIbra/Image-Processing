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
	cropRow = int((rows * (1 - resize_ratio)) // 2), cropCol = int((columns * (1 - resize_ratio)) // 2)
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
	return


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

image_scaled = # Your code goes here. If you choose not to scale the image, just remove it.
pattern_scaled =  # Your code goes here. If you choose not to scale the pattern, just remove it.

display(image_scaled, pattern_scaled)

ncc = # Your code goes here
real_matches = # Your code goes here

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:,0] += pattern_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:,1] += pattern_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.

draw_matches(image, real_matches, pattern_scaled.shape)	# if pattern was not scaled, replace this with "pattern"





############# Crew #############

image_scaled = # Your code goes here. If you choose not to scale the image, just remove it.
pattern_scaled =  # Your code goes here. If you choose not to scale the pattern, just remove it.

display(image_scaled, pattern_scaled)

ncc = # Your code goes here
real_matches = # Your code goes here

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:,0] += pattern_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:,1] += pattern_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.

draw_matches(image, real_matches, pattern_scaled.shape)	# if pattern was not scaled, replace this with "pattern"
