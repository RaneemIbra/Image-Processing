# Raneem Ibraheem, 212920896
# Selan Abu Saleh, 212111439

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_laplacian_pyramid(image, levels, resize_ratio=0.5):
    # Initialize the Gaussian and Laplacian pyramids
    gaussian_pyramid = [np.float32(image)]
    laplacian_pyramid = []
    
    # Build the Gaussian pyramid by downscaling the image
    for _ in range(levels - 1):
        # Reduce the size of the image using bilinear interpolation
        gaussian_blur=cv2.GaussianBlur(gaussian_pyramid[-1],(5,5),0)
        image_blurred = cv2.resize(gaussian_blur, (0, 0), fx=resize_ratio, fy=resize_ratio)
        # Append the downscaled image to the Gaussian pyramid

        gaussian_pyramid.append(image_blurred.astype(np.float32))
    
    # Build the Laplacian pyramid by computing the differences between Gaussian levels
    for i in range(levels - 1):
        # Resize the next level's image to match the current level's size
        resized_image = cv2.resize(gaussian_pyramid[i + 1], (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        # Compute the Laplacian level by subtracting the resized image from the current Gaussian level
        laplacian_level = cv2.subtract(gaussian_pyramid[i], resized_image)
        # Append the Laplacian level to the pyramid
        laplacian_pyramid.append(laplacian_level)
    
    # Add the smallest Gaussian image as the last Laplacian level
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return laplacian_pyramid



def restore_from_pyramid(pyramidList, resize_ratio=2):
	# Initialize reconstruction with the smallest image in the pyramid
    reconstructed_image = pyramidList[-1]
    
    # Reconstruct the image by iteratively adding upscaled images to Laplacian levels
    for level in range(len(pyramidList) - 2, -1, -1):
        # Determine the size of the current pyramid level
        target_height, target_width = pyramidList[level].shape[:2]
        # Upscale the reconstructed image to match the current level's dimensions
        upscaled_image = cv2.resize(reconstructed_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        # Add the Laplacian level to the upscaled image to reconstruct details
        reconstructed_image = cv2.add(upscaled_image, pyramidList[level])
    
    return reconstructed_image


def validate_operation(img):
	pyr = get_laplacian_pyramid(img, 7)
	img_restored = restore_from_pyramid(pyr)

	plt.title(f"MSE is {np.mean((img_restored - img) ** 2)}")
	plt.imshow(img_restored, cmap='gray')

	plt.show()
	

def blend_pyramids(levels):
	blend_pyramid = []  # List to store the blended pyramid levels
	for level in range(levels):
		# Step 1: Create an empty mask with the same shape as the current pyramid level
		mask = np.zeros_like(pyr_apple[level], dtype=np.float32)
		column_width = mask.shape[1]  # Get the width of the current pyramid level
        
		# Step 2: Set the left half of the mask to 1.0 (fully showing apple image)
		mask[:, :column_width // 2 - int((levels - level) * 1.5)] = 1.0

		# Step 3: Define the start point for the blend area
		blend_first = column_width // 2 - int((levels - level) * 1.5)

		# Step 4: Gradually create the blend by filling the transition area with a gradient
		for column in range(blend_first, column_width // 2 + int((levels - level) * 1.5) + 1):
			i = column - blend_first
			mask[:, column] = 0.9 - 0.9 * i / (2 * (levels - level))

		# Step 5: Blend the two pyramids (orange and apple) at the current level using the mask
		blended_level = pyr_orange[level] * (1 - mask) + pyr_apple[level] * mask

		# Step 6: Append the blended level to the list of blended pyramid levels
		blend_pyramid.append(blended_level)
	return blend_pyramid



apple = cv2.imread('apple.jpg')
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)

orange = cv2.imread('orange.jpg')
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)

# validate_operation(apple)
# validate_operation(orange)
validate_operation(apple)
validate_operation(orange)

pyramid_height = 7
pyr_apple = get_laplacian_pyramid(apple, pyramid_height, resize_ratio=0.5)
pyr_orange = get_laplacian_pyramid(orange, pyramid_height, resize_ratio=0.5)



pyr_result = []


# Your code goes here
pyr_result = blend_pyramids( pyramid_height)


final = restore_from_pyramid(pyr_result)
plt.imshow(final, cmap='gray')
plt.show()

cv2.imwrite("result.jpg", final)

