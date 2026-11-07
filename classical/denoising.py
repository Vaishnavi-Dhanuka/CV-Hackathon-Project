import cv2
import numpy as np


def bilateral_denoise(image, sigma=15, d=7):
    """Bilateral filter. sigma ~ noise level in gray levels (0-255) you want removed."""
    sigma = max(1.0, float(sigma))
    return cv2.bilateralFilter(image, d, sigmaColor=2.0 * sigma, sigmaSpace=5.0)


def guided_denoise(image, sigma=15, radius=4):
    """Guided filter with the image as its own guide. eps = sigma^2 (gray-level units)."""
    sigma = max(1.0, float(sigma))
    return cv2.ximgproc.guidedFilter(image, image, int(radius), sigma * sigma)


def denoise(image, method="guided", sigma=15):
    """Main entry point. uint8 RGB in/out. method: 'guided' | 'bilateral' | 'none'."""
    if method == "guided":
        return guided_denoise(image, sigma)
    if method == "bilateral":
        return bilateral_denoise(image, sigma)
    return image