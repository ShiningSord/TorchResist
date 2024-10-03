import matplotlib.pyplot as plt
import numpy as np
import cv2

def writeMaskPNG(image : np.ndarray, threshold = 0.5, write = False, filename=None):
    image[image >= threshold] = 1
    image[image < threshold] = 0
    image = (image * 255).astype(np.uint8)
    if write:
        cv2.imwrite(f"{filename}.png", image)
    else:
        plt.imshow(image)
        
        
def writePNG(image : np.ndarray, write = False, filename=None):
    if write:
        cv2.imwrite(f"{filename}.png", image)
    else:
        plt.axis("off")
        plt.imshow(image)