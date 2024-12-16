# Raneem Ibraheem, 212920896
# Selan Abu Saleh, 212111439

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import os
import shutil
import sys

def get_transform(matches, is_affine):
    # Extract source and destination points from matches
    src_points, dst_points = matches[:, 0], matches[:, 1]

    if is_affine:
        T, _ = cv2.estimateAffine2D(dst_points, src_points)
    else:
        T, _ = cv2.findHomography(dst_points, src_points)

    return T

def stitch(img1, img2):
    # Create a mask where img2 has non-black pixels
    mask = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0)  # Non-black pixels in img2

    # Overlay non-black pixels from img2 onto img1
    img1[mask] = img2[mask]

    return img1

# Output size is (w,h)
def inverse_transform_target_image(target_img, original_transform, output_size):
    row, col = original_transform.shape
    
    if row == 2:  # Affine transformation (2x3 matrix)
        
        warped_image = cv2.warpAffine(target_img, original_transform, output_size[::-1], 
                                      flags=2, borderMode=cv2.BORDER_TRANSPARENT)
    elif row == 3:  # Homography transformation (3x3 matrix)
        warped_image = cv2.warpPerspective(target_img, transformation_matrix, output_size[::-1], flags=2,
                                            borderMode=cv2.BORDER_TRANSPARENT)
    return warped_image

# returns list of pieces file names
def prepare_puzzle(puzzle_dir):
	edited = os.path.join(puzzle_dir, 'abs_pieces')
	if os.path.exists(edited):
		shutil.rmtree(edited)
	os.mkdir(edited)
	
	affine = 4 - int("affine" in puzzle_dir)
	
	matches_data = os.path.join(puzzle_dir, 'matches.txt')
	n_images = len(os.listdir(os.path.join(puzzle_dir, 'pieces')))

	matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images-1,affine,2,2)
	
	return matches, affine == 3, n_images


if __name__ == '__main__':
    lst = ['puzzle_affine_1', 'puzzle_affine_2', 'puzzle_homography_1']
    
    for puzzle_dir in lst:
        print(f'Starting {puzzle_dir}')
        
        # Define puzzle directory path
        puzzle = os.path.join('puzzles', puzzle_dir)
        
        # Path to pieces and abs_pieces
        pieces_pth = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')
        
        # Prepare puzzle data (matches, affine flag, and number of pieces)
        matches, is_affine, n_images = prepare_puzzle(puzzle)
        
        # Get path to first piece
        target_img_path = os.path.join(puzzle, 'pieces', 'piece_1.jpg')

        # Read the first piece image
        target_img = cv2.imread(target_img_path)
        sol_file = f'piece_{1}_relative.jpg'
        cv2.imwrite(os.path.join(puzzle, 'abs_pieces', f'piece_{1}_absolute.jpg'), target_img)
       

        # Save the first piece as the "absolute" piece
        #cv2.imwrite(os.path.join(puzzle, 'abs_pieces', 'piece_1_absolute.jpg'), target_img)

        # Initialize final puzzle with the first piece
        final_puzzle = target_img
        first_image_size = target_img.shape[:2]

        # Iterate through the rest of the pieces (starting from the second piece)
        for i in range(1, n_images):
            # Get the path to the current piece
            target_img_path = os.path.join(puzzle, 'pieces', f'piece_{i+1}.jpg')

            # Read the current piece image
            target_img = cv2.imread(target_img_path)

            # Compute the transformation matrix for the current piece
            transformation_matrix = get_transform(matches[i-1], is_affine)

            # Apply inverse transformation to align the current piece
            warped_img = inverse_transform_target_image(target_img, transformation_matrix, first_image_size)

            # Save the warped piece to "abs_pieces" folder
            #cv2.imwrite(os.path.join(puzzle, 'abs_pieces', f'piece_{i+1}_absolute.jpg'), warped_img)
            sol_file = f'piece_{i+1}_relative.jpg'
            cv2.imwrite(os.path.join(puzzle, 'abs_pieces', f'piece_{i+1}_absolute.jpg'), warped_img)


            # Stitch the warped piece to the final puzzle image
            final_puzzle = stitch(final_puzzle, warped_img)

        # Save the final stitched puzzle image as solution.jpg
        sol_file = f'solution.jpg'
        cv2.imwrite(os.path.join(puzzle, sol_file), final_puzzle)
