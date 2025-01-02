# Raneem Ibraheem, 212920896
# Selan Abu Saleh, 212111439

# Please replace the above comments with your names and ID numbers in the same format.

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt


def clean_baby(im):
	# Your code goes here
    denoised = cv2.medianBlur(im, 3)
    
    # 2) Convert to binary mask to separate foreground (baby) from black background
    #    Because it's grayscale, pick a threshold that keeps the baby region
    _, mask = cv2.threshold(denoised, 10, 255, cv2.THRESH_BINARY)
    
    # 3) Find the largest contour or connected region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return denoised  # fallback if no contours found
    
    # 4) Keep bounding box of largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 5) Crop to that bounding box
    cropped = denoised[y:y+h, x:x+w]
    
    # Optionally do additional steps like rotate or further enhance.
    # For example, if the baby is sideways, detect orientation and rotate.

    return cropped

def clean_windmill(im):
    # 1) DFT
    f = fftshift(fft2(im.astype(np.float32)))
    rows, cols = im.shape
    crow, ccol = rows//2, cols//2
    
    # 2) Identify bright “spike” frequencies
    #    For example, threshold the magnitude of f
    magnitude = np.log1p(np.abs(f))
    # We might find local maxima columns away from ccol
    
    # A more “manual” approach is to just look for stripes near the center row:
    # We'll create small notches around each spike
    # Example: we see spikes at ±k in the horizontal freq direction. Let’s guess ±30 from center:
    for offset in [30, -30, 60, -60]:
        notch_radius = 5
        center_x = ccol + offset
        # Make a circular notch
        for r in range(rows):
            for c in range(cols):
                if (r - crow)**2 + (c - center_x)**2 < notch_radius**2:
                    f[r,c] = 0
    
    # 3) Inverse transform
    f_ishift = ifftshift(f)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    
    # 4) Possibly do mild sharpening or contrast enhancement if needed
    return img_back

def clean_watermelon(im):
    # Convert to float
    f32 = im.astype(np.float32)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(f32, (3,3), 0)
    
    # Increase alpha for stronger sharpening
    alpha = 2.5  # more aggressive
    sharpened = cv2.addWeighted(f32, 1 + alpha, blurred, -alpha, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # Optionally apply a second pass if you want it even sharper
    # sharpened_2 = cv2.addWeighted(sharpened, 1 + alpha, 
    #                              cv2.GaussianBlur(sharpened, (3,3), 0), -alpha, 0)
    
    return sharpened


def clean_umbrella(im):
    # Convert to float
    f32 = im.astype(np.float32)
    
    # 1) Estimate shift using phaseCorrelate on the left half vs. right half
    #    This is a bit of a trick, but let’s just do a more direct approach:
    rows, cols = f32.shape
    # We'll break into two halves (left half, right half)
    left_half  = f32[:, :cols//2]
    right_half = f32[:, cols//2:]
    
    # Pad them so they’re the same size for phase correlation
    shift, response = cv2.phaseCorrelate(left_half, right_half)
    # shift is (shift_x, shift_y), but we’re correlating horizontally, so shift_y might be ~0
    # shift_x might be negative or positive.  We need to interpret carefully.
    
    # 2) Construct a shifted version of the entire image in the opposite direction
    #    For example, if shift_x > 0, we shift left, etc.
    M = np.float32([
        [1, 0, -shift[0]],  # negative because we want to undo the shift
        [0, 1, -shift[1]]
    ])
    shifted_im = cv2.warpAffine(f32, M, (cols, rows), flags=cv2.INTER_LINEAR)
    
    # 3) Combine the two to approximate original
    #    From the equation: I_noisy = 0.5*(I_orig + shift(I_orig))
    #    So roughly: I_orig ≈ 2*I_noisy - shift(I_noisy).
    #    But let’s do a simpler approach: I_orig = im + shifted_im
    out = f32 + shifted_im  # or 2*im - shifted_im, depending on your math sign
    
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    # Possibly do a small median blur to remove any boundary artifacts
    cleaned = cv2.medianBlur(out, 3)
    
    return cleaned


def clean_USAflag(im):
    # Convert to float
    f32 = im.astype(np.float32)
    
    # 1) We can do a morphological approach to find the scribbles. For example:
    #    a) Use a small median filter to smooth normal stripes
    smooth = cv2.medianBlur(f32, 3)
    diff = cv2.absdiff(f32, smooth)
    
    # 2) Threshold the difference
    _, mask = cv2.threshold(diff.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)
    
    # 3) We might refine the mask by removing small noise with morphological ops
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=2)
    
    # 4) Inpaint
    inpainted = cv2.inpaint(im, mask, 3, cv2.INPAINT_TELEA)
    
    return inpainted


def clean_house(im):
    # Convert to float
    g = im.astype(np.float32)
    rows, cols = g.shape
    
    # (Optional) Attempt to guess blur angle/length from the radon transform
    # or by analyzing the frequency domain. Below is a manual guess:
    length = 20
    angle = 0  # degrees, you can try 0, 5, 10, etc. or find automatically
    
    # Create a motion blur PSF
    psf = np.zeros((rows, cols), np.float32)
    
    # We’ll just draw a line in psf to represent the blur
    def make_motion_psf(size, angle_deg, length_px):
        # For simplicity, we’ll keep it big. But a real approach might pad smaller PSF
        psf_small = np.zeros((size, size), np.float32)
        cv2.line(psf_small,
                 (size//2 - length_px//2, size//2),
                 (size//2 + length_px//2, size//2),
                 1,
                 1)
        # Rotate
        M = cv2.getRotationMatrix2D((size//2, size//2), angle_deg, 1.0)
        psf_rot = cv2.warpAffine(psf_small, M, (size, size))
        # Normalize
        psf_rot /= psf_rot.sum() if psf_rot.sum() != 0 else 1
        return psf_rot
    
    small_size = 2*max(rows, cols)  # big enough to avoid wrap
    psf_small = make_motion_psf(small_size, angle, length)
    
    # Center-crop to our image size
    psf = psf_small[ (small_size-rows)//2 : (small_size+rows)//2,
                     (small_size-cols)//2 : (small_size+cols)//2 ]
    
    # FFT of image, FFT of PSF
    G = fftshift(fft2(g))
    H = fftshift(fft2(psf))
    
    # Wiener filtering
    K = 0.01
    magnitude_H2 = np.abs(H)**2
    # F_restored = G * (conj(H) / (|H|^2 + K))
    F_restored = G * (np.conj(H) / (magnitude_H2 + K))
    
    # Inverse transform
    out_shift = ifftshift(F_restored)
    out_spatial = ifft2(out_shift)
    out_spatial = np.abs(out_spatial)
    out_spatial = np.clip(out_spatial, 0, 255).astype(np.uint8)
    
    # Possibly denoise if it’s grainy
    de_noised = cv2.fastNlMeansDenoising(out_spatial, None, 7, 7, 21)
    
    return de_noised


def clean_bears(im):
    # 1) Histogram Equalization
    #    If 8-bit grayscale, do:
    eq = cv2.equalizeHist(im)
    
    # 2) Adaptive (CLAHE) can be better:
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # eq = clahe.apply(im)
    
    # 3) Mild gamma correction.  Because eq might already lighten it,
    #    pick a gamma > 1 to brighten or < 1 to darken. Let’s try gamma=0.8
    gamma = 0.8
    eq_f = eq.astype(np.float32)/255.0
    corrected = np.power(eq_f, gamma)
    corrected = (corrected*255).clip(0, 255).astype(np.uint8)
    
    # 4) Unsharp mask to bring out details
    blur = cv2.GaussianBlur(corrected, (3,3), 0)
    alpha = 1.0
    sharpened = cv2.addWeighted(corrected, 1 + alpha, blur, -alpha, 0)
    
    return sharpened

