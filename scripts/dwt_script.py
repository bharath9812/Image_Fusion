import pywt

def channel_transform_advanced(ch1, ch2, level=2):
    coeffs1 = pywt.wavedec2(ch1, 'db5', mode='periodization', level=level)
    coeffs2 = pywt.wavedec2(ch2, 'db5', mode='periodization', level=level)
    
    fused_coeffs = []
    for c1, c2 in zip(coeffs1, coeffs2):
        if isinstance(c1, tuple):
            # Fuse the detail coefficients using a more sophisticated rule, like taking the maximum
            fused_coeffs.append(tuple((np.maximum(detail1, detail2) for detail1, detail2 in zip(c1, c2))))
        else:
            # Average the approximation coefficients
            fused_coeffs.append((c1 + c2) / 2)
    
    return pywt.waverec2(fused_coeffs, 'db5', mode='periodization')

# You would need to adjust the rest of your fusion function to use this new channel transform function.
import pywt
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def wavelet_decompose(image, wavelet='db5', level=2):
    """ Decompose an image into wavelet coefficients up to the specified level. """
    coeffs = pywt.wavedec2(image, wavelet, mode='periodization', level=level)
    return coeffs

def wavelet_reconstruct(coeffs, wavelet='db5'):
    """ Reconstruct an image from wavelet coefficients. """
    return pywt.waverec2(coeffs, wavelet, mode='periodization')


def pad_coefficients(c1, c2):
    # Calculate padding amounts
    pad_rows = abs(c1.shape[0] - c2.shape[0])
    pad_cols = abs(c1.shape[1] - c2.shape[1])
    
    # Apply padding to the smaller array
    if c1.shape[0] < c2.shape[0]:
        c1 = np.pad(c1, ((0, pad_rows), (0, 0)), 'symmetric')
    else:
        c2 = np.pad(c2, ((0, pad_rows), (0, 0)), 'symmetric')
    
    if c1.shape[1] < c2.shape[1]:
        c1 = np.pad(c1, ((0, 0), (0, pad_cols)), 'symmetric')
    else:
        c2 = np.pad(c2, ((0, 0), (0, pad_cols)), 'symmetric')
    
    return c1, c2

def fuse_coefficients(coeffs1, coeffs2, method='average'):
    """ Fuse the wavelet coefficients from two images. """
    fused_coeffs = []
    for i, (c1, c2) in enumerate(zip(coeffs1, coeffs2)):
        if isinstance(c1, tuple):  # Detail coefficients
            temp = []
            for detail1, detail2 in zip(c1, c2):
                detail1, detail2 = pad_coefficients(detail1, detail2)
                if method == 'average':
                    temp.append((detail1 + detail2) / 2)
                elif method == 'min':
                    temp.append(np.minimum(detail1, detail2))
                elif method == 'max':
                    temp.append(np.maximum(detail1, detail2))
            fused_coeffs.append(tuple(temp))
        else:  # Approximation coefficients
            c1, c2 = pad_coefficients(c1, c2)
            if method == 'average':
                fused_coeffs.append((c1 + c2) / 2)
            elif method == 'min':
                fused_coeffs.append(np.minimum(c1, c2))
            elif method == 'max':
                fused_coeffs.append(np.maximum(c1, c2))
    return fused_coeffs



def fusion_process(img1, img2, wavelet='db5', level=2, fusion_method='average'):
    img1, img2 = resize_for_wavelet(img1, img2, level=2)
    """ Perform the fusion of two images. """
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Decompose both images
    coeffs1 = wavelet_decompose(img1_gray, wavelet, level)
    coeffs2 = wavelet_decompose(img2_gray, wavelet, level)
    
    # Fuse coefficients
    fused_coeffs = fuse_coefficients(coeffs1, coeffs2, fusion_method)
    
    # Reconstruct the image based on the fused coefficients
    fused_image = wavelet_reconstruct(fused_coeffs, wavelet)
    
    # Normalize the pixel values
    fused_image = np.clip(fused_image, 0, 255)
    fused_image = fused_image.astype(np.uint8)
    
    # Quality metrics
    sim1 = ssim(img1_gray, fused_image)
    sim2 = ssim(img2_gray, fused_image)
    
    return fused_image

def resize_for_wavelet(image1, image2, level):
    """ Resize both images to the same dimensions suitable for wavelet decomposition. """
    factor = 2**level
    new_width = min(image1.shape[1], image2.shape[1]) // factor * factor
    new_height = min(image1.shape[0], image2.shape[0]) // factor * factor
    
    resized_image1 = cv2.resize(image1, (new_width, new_height), interpolation=cv2.INTER_AREA)
    resized_image2 = cv2.resize(image2, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_image1, resized_image2

