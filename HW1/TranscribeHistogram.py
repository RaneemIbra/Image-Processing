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
    """
    compares a target histogram to sliding windows histograms in the source image
    arguments:
        source image
        target_image
    returns:
        True if a match is found, False otherwise
    """
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
    """
    calculates the EMD between two histograms
    arguments:
        first histogram
        second histogram
    returns:
        the EMD value
    """
    first_cumulative_histogram = np.cumsum(first_histogram)
    second_cumulative_histogram = np.cumsum(second_histogram)
    return np.sum(np.abs(first_cumulative_histogram - second_cumulative_histogram))

def get_binary_images(images):
    """
    converts grayscale images to binary using thresholding
    arguments:
        array of grayscale images
    returns:
        list of binary images
    """
    binary_images = []
    for img in images: 
        _, binary_image = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
        binary_images.append(binary_image)
    return binary_images

def calculate_all_bar_heights(binary_images):
    """
    calculates the heights of all bars in all histogram images
    arguments:
        list of binary histogram images
    returns:
        list of bar heights for each image
    """
    bar_height_list = []
    for image in binary_images:
        bar_heights = [get_bar_height(image, bin_index) for bin_index in range(10)]
        bar_height_list.append(bar_heights)
    return bar_height_list

def find_highest_bar(images, numbers):
    """
    finds the highest bar for each histogram image
    arguments:
        array of histogram images
        array of number images
    returns:
        list of recognized digits for each histogram
    """
    recognized_digits = []
    for image in images:
        for digit in range(9, -1, -1):
            if compare_hist(image, numbers[digit]):
                recognized_digits.append(digit)
                break
    return recognized_digits

def calculate_students_numbers(max_students, bar_heights):
    """
    calculates the number of students for each bin based on bar heights
    arguments:
        list of maximum student counts for each histogram
        list of bar heights for each histogram
    returns:
        list: List of student numbers per bar for each histogram
    """
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

highest_bars = find_highest_bar  (images,numbers)  
quantized_images = quantization(images, n_colors=4)
binary_images = get_binary_images(quantized_images)
bar_heights = calculate_all_bar_heights(binary_images)
students_bar_heights = calculate_students_numbers(highest_bars, bar_heights)

for i in range(7):
    print(f'Histogram {names[i]} gave {students_bar_heights[i]}')
 
cv2.waitKey(0)
cv2.destroyAllWindows() 