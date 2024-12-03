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
    while image[y_pos, x_pos] == 0:
        y_pos -= 1
    return 274 - y_pos

# Sections c, d
# Remember to uncomment compare_hist before using it!

def compare_hist(src_image, target):
    windows = np.lib.stride_tricks.sliding_window_view(src_image,
                                                        (target.shape[0], target.shape[1]))
    target_hist = cv2.calcHist([target], [0], None, [256], [0, 256]).flatten()
    for x in range(30, 50):
        for y in range(100, 145):
            window_hist = cv2.calcHist([windows[y,x]], [0], None, [256], [0, 256]).flatten()             
            emd = get_EMD_Value(window_hist, target_hist)            
            if emd < 260: 
                return True
    return False

def get_EMD_Value(first_histogram, second_histogram):
    first_cumulative_histogram = np.cumsum(first_histogram)
    second_cumulative_histogram = np.cumsum(second_histogram)
    return np.sum(np.abs(first_cumulative_histogram - second_cumulative_histogram))

def get_binary_images(images):
    res = []
    for img in images: 
        _, binary_image = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
        res.append(binary_image)
    return res

def get_heights_list(images):
    image_list = []
    for img in images:
        bar_heights = []
        for bin_id in range(10):
            height = get_bar_height(img, bin_id)
            bar_heights.append(height)
        image_list.append(bar_heights)
    return image_list

def get_highest_bar(images, numbers):
    highest_bar_list = []
    for index, image in enumerate(images):
        recognized_digit = None
        for digit in range(9, -1, -1):
            result = compare_hist(image, numbers[digit])
            if result:
                recognized_digit = digit
                highest_bar_list.append(recognized_digit)
                break
    return highest_bar_list

def calc_heights(max_students, bar_heights):
    res = []
    for i in range(7):
        students_per_bin = []
        for bin_id in range(10): 
            students_per_bin.append(round(max_students[i] * bar_heights[i][bin_id] / max(bar_heights[i])))
        res.append(students_per_bin)
    return res  

# Sections a, b

# Main processing loop
images, names = read_dir('./HW1/data')
numbers, numNames = read_dir('./HW1/numbers')

highest_bar = get_highest_bar(images,numbers)  
quantized_images = quantization(images, n_colors=4)
binary_images = get_binary_images(quantized_images)
bar_heights = get_heights_list(binary_images)
heights = calc_heights(highest_bar, bar_heights)

for i in range(7):
    print(f'Histogram {names[i]} gave {heights[i]}')
 
cv2.waitKey(0)
cv2.destroyAllWindows() 