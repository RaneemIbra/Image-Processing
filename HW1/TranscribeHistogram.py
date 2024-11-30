# Raneem Ibraheem, 212920896
# Selan Abu Saleh, 212111439

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

# Input: numpy array of images and number of gray levels to quantize the images down to
# Output: numpy array of images, each with only n_colors gray levels
def quantization(imgs_arr, n_colors=4):
	img_size = imgs_arr[0].shape
	res = []
	
	for img in imgs_arr:
		X = img.reshape(img_size[0] * img_size[1], 1)
		km = KMeans(n_clusters=n_colors)
		km.fit(X)
		
		img_compressed = km.cluster_centers_[km.labels_]
		img_compressed = np.clip(img_compressed.astype('uint8'), 0, 255)

		res.append(img_compressed.reshape(img_size[0], img_size[1]))
	
	return np.array(res)

# Input: A path to a folder and formats of images to read
# Output: numpy array of grayscale versions of images read from input folder, and also a list of their names
def read_dir(folder, formats=(".jpg", ".png")):
	image_arrays = []
	lst = [file for file in os.listdir(folder) if file.endswith(formats)]
	for filename in lst:
		file_path = os.path.join(folder, filename)
		image = cv2.imread(file_path)
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image_arrays.append(gray_image)
	return np.array(image_arrays), lst

# Input: image object (as numpy array) and the index of the wanted bin (between 0 to 9)
# Output: the height of the idx bin in pixels
def get_bar_height(image, idx):
	# Assuming the image is of the same pixel proportions as images supplied in this exercise, the following values will work
	x_pos = 70 + 40 * idx
	y_pos = 274
	while image[y_pos, x_pos] == 1:
		y_pos-=1
	return 274 - y_pos

# Sections c, d
# Remember to uncomment compare_hist before using it!

def compare_hist(src_image, target): 
    windows = np.lib.stride_tricks.sliding_window_view(src_image, (target.shape[0], target.shape[1]))
    visualized_image = cv2.cvtColor(src_image.copy(), cv2.COLOR_GRAY2BGR)
    for y in range(30, 50):
        for x in range(100, 145):
            window = windows[x, y]
            top_left = (y, x)
            bottom_right = (y + target.shape[1], x + target.shape[0])
            cv2.rectangle(visualized_image, top_left, bottom_right, (0, 255, 0), 1)
            cv2.imshow("Sliding Window Progress", visualized_image)
            cv2.waitKey(1)
            window_hist = cv2.calcHist([window], [0], None, [256], [0, 256]).flatten()
            target_hist = cv2.calcHist([target], [0], None, [256], [0, 256]).flatten()
            window_cumsum = np.cumsum(window_hist)
            target_cumsum = np.cumsum(target_hist)
            emd = np.sum(np.abs(window_cumsum - target_cumsum))
            if emd < 260:
                print(f"Match found at x={x}, y={y} with EMD={emd:.2f}")
                cv2.destroyWindow("Sliding Window Progress")
                return True
            
    print("No match found within the corrected threshold.")
    cv2.destroyWindow("Sliding Window Progress")
    return False


# Sections a, b

# Main processing loop
images, names = read_dir('./HW1/data')
numbers, numNames = read_dir('./HW1/numbers')

recognized_number = None

# Extract the digits and sort the indices in descending order
digit_indices = sorted(range(len(numbers)), key=lambda i: int(numNames[i][0]), reverse=True)

# Iterate through the sorted indices
for idx in digit_indices:
    digit = int(numNames[idx][0])  # Extract the digit from the filename
    number = numbers[idx]  # Get the corresponding number image
    
    # Process numbers in descending order
    if compare_hist(images[6], number):
        recognized_number = digit
        break

if recognized_number is not None:
    print(f"Histogram {names[1]} recognized number: {recognized_number}")
else:
    print(f"Histogram {names[1]} did not recognize any number below the threshold.")


cv2.imshow(names[0], images[0])
cv2.imshow(numNames[2], numbers[2])
cv2.waitKey(0)
cv2.destroyAllWindows() 
exit()


# The following print line is what you should use when printing out the final result - the text version of each histogram, basically.

# print(f'Histogram {names[id]} gave {heights}')