import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift

plus_image = cv2.imread('plus2.jpg', cv2.IMREAD_GRAYSCALE)
heart_image = cv2.imread('heart.jpg', cv2.IMREAD_GRAYSCALE)


plus_image_size = plus_image.shape[::-1]
heart_image_size = heart_image.shape[::-1]
synthetic_image = np.zeros(plus_image_size, dtype=np.uint8)

center_x = plus_image_size[0] // 2
center_y = plus_image_size[1] // 2

crosshair_width = 3
crosshair_length = 30
gray_level_value = 255

# Drawing horizontal crosshair
synthetic_image[center_y - crosshair_width // 2:center_y + crosshair_width // 2 + 1, 
                center_x - crosshair_length // 2:center_x + crosshair_length // 2 + 1] = gray_level_value

synthetic_image[center_y - crosshair_length // 2:center_y + crosshair_length // 2 + 1, 
                center_x - crosshair_width // 2:center_x + crosshair_width // 2 + 1] = gray_level_value

synthetic_image_resized = cv2.resize(synthetic_image, plus_image_size)
cv2.imwrite('synthetic_image_2.jpg', synthetic_image_resized)

synthetic_fft = fft2(synthetic_image_resized)
synthetic_fft_resized = cv2.resize(heart_image, heart_image_size)

plus_fft = fft2(plus_image)
heart_fft = fft2(heart_image)

blurring_function = np.divide(synthetic_fft, plus_fft)
blurring_function_magnitude = np.abs(blurring_function)
blurring_function_magnitude_resized = cv2.resize(blurring_function_magnitude, (heart_image.shape[1], heart_image.shape[0]))

cv2.imwrite('synthetic_image_2.jpg', blurring_function_magnitude_resized)
cv2.imshow('Blurring Function_resized', blurring_function_magnitude_resized)
cv2.waitKey(0)

blurring_function_log_magnitude = np.log1p(blurring_function_magnitude_resized)
cv2.imshow('Blurring Function', blurring_function_log_magnitude.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


def inverse_filter(blurred_fft, H_uv):
    return np.abs(ifft2((blurred_fft / H_uv)))

def pseudo_inverse_filter(blurred_fft, H_uv, epsilon=1e-10):
    return np.abs(ifft2(fftshift(blurred_fft / (H_uv + epsilon))))

def wiener_filter(blurred_fft, H_uv, noise_power=0.1):
    return np.abs(ifft2(fftshift((np.conj(H_uv) / (np.abs(H_uv) ** 2 + noise_power)) * blurred_fft)))

restored_inverse = inverse_filter(fftshift(heart_fft), fftshift(blurring_function_magnitude_resized))
restored_pseudo_inverse = pseudo_inverse_filter(fftshift(heart_fft), fftshift(blurring_function_magnitude_resized))
restored_wiener = wiener_filter(fftshift(heart_fft), fftshift(blurring_function_magnitude_resized))


restored_inverse = cv2.normalize(restored_inverse, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
restored_pseudo_inverse = cv2.normalize(restored_pseudo_inverse, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
restored_wiener = cv2.normalize(restored_wiener, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

cv2.imshow('Restored Image (Inverse Filter)', restored_inverse.astype(np.uint8))
cv2.imshow('Restored Image (Pseudo-inverse Filter)', restored_pseudo_inverse.astype(np.uint8))
cv2.imshow('Restored Image (Wiener Filter)', restored_wiener.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
