import base64
import random
import cv2
import numpy as np


def img_to_base64(filepath):
    with open(filepath, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
def rand(num_bits=30):
    print("Random input bit message")
    random_str = ''.join([str(random.randint(0, 1)) for _ in range(num_bits)])
    return random_str

def calculate_similarity_percentage(str1, str2):
    if len(str1) == 0:
        return "Original copyright watermark unknown"
    elif len(str1) != len(str2):
        return "Input and output watermark lengths differ"
    total_length = len(str1)
    same_count = sum(1 for x, y in zip(str1, str2) if x == y)
    similarity_percentage = (same_count / total_length) * 100
    return f"{similarity_percentage}%"

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR NumPy image to RGB using OpenCV.
    
    Args:
        img_bgr (np.ndarray): Input image in BGR format (H,W,3)
    
    Returns:
        np.ndarray: Output image in RGB format (H,W,3), same dtype as input
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB NumPy image to BGR using OpenCV.

    Args:
        img_rgb (np.ndarray): Input image in RGB format (H,W,3)

    Returns:
        np.ndarray: Output image in BGR format (H,W,3), same dtype as input
    """
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)