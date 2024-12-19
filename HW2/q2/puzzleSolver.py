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

# a function to stitch two images together
def stitch(img1, img2):
    # create a binary mask 
    mask = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0)
    # stitch img2 onto img1
    img1[mask] = img2[mask]
    return img1

# applies the inverse of a given transformation matrix to a target image
def inverse_transform_target_image(target_img, original_transform, output_size):
    if original_transform.shape == (2, 3):
        warped_image = cv2.warpAffine(
            target_img, 
            original_transform, 
            (output_size[1], output_size[0]),
            flags=cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_TRANSPARENT
        )
    elif original_transform.shape == (3, 3):
        warped_image = cv2.warpPerspective(
            target_img, 
            original_transform, 
            (output_size[1], output_size[0]),
            flags=cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_TRANSPARENT
        )
    else:
        raise ValueError("invalid transformation matrix")

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
        
        puzzle = os.path.join('puzzles', puzzle_dir)
        
        pieces_pth = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')
        
        matches, is_affine, n_images = prepare_puzzle(puzzle)
        
        # get the first piece and store it in the abs folder
        first_piece_path = os.path.join(puzzle, 'pieces', 'piece_1.jpg')
        target_piece = cv2.imread(first_piece_path)
        solutions_dir = f'piece_{1}_relative.jpg'
        cv2.imwrite(os.path.join(puzzle, 'abs_pieces', f'piece_{1}_absolute.jpg'), target_piece)

        # initialize the solved puzzle with the first piece
        solved_puzzle = target_piece
        first_image_size = target_piece.shape[:2]

        # iterate over the remaining pieces to solve the puzzle
        for i in range(1, n_images):
            first_piece_path = os.path.join(puzzle, 'pieces', f'piece_{i+1}.jpg')
            target_piece = cv2.imread(first_piece_path)
            transform = get_transform(matches[i-1], is_affine)
            inverse_transformed_image = inverse_transform_target_image(target_piece, transform, first_image_size)
            solutions_dir = f'piece_{i+1}_relative.jpg'
            cv2.imwrite(os.path.join(puzzle, 'abs_pieces', f'piece_{i+1}_absolute.jpg'), inverse_transformed_image)
            solved_puzzle = stitch(solved_puzzle, inverse_transformed_image)

        solutions_dir = f'solution.jpg'
        cv2.imwrite(os.path.join(puzzle, solutions_dir), solved_puzzle)
