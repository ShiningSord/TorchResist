# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from simulator import AbbeSim
from tqdm import tqdm

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process images with AbbeSim.")
    parser.add_argument("--mask", required=True, type=str, help="Path to the folder containing input mask images.")
    parser.add_argument("--outpath", required=True, type=str, help="Path to the output directory.")
    parser.add_argument("--resolution", required=True, type=float, help="Pixel size resolution for processing.")
    return parser.parse_args()

def load_images_in_batches(folder_path, batch_size, device):
    """
    Load images in batches from the specified folder.

    Args:
        folder_path (str): Path to the folder containing images.
        batch_size (int): Number of images per batch.
        device (torch.device): Device to load images onto.

    Yields:
        tuple: Batch of image filenames and corresponding tensors.
    """
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    image_files.sort()
    image_files = image_files[:50]

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_images = []

        for file in batch_files:
            image_path = os.path.join(folder_path, file)
            image_np = np.array(Image.open(image_path).convert('L'), dtype=np.float32)
            image_tensor = torch.from_numpy(image_np).to(device) / 255.0
            batch_images.append(image_tensor)

        batch_tensor = torch.stack(batch_images)
        yield (batch_files, batch_tensor)

def save_image(image_array, output_path, filename):
    """
    Save a single image using plt.imsave.

    Args:
        image_array (np.ndarray): Image array to save.
        output_path (str): Path to save the image.
        filename (str): Filename for the saved image.
    """
    plt.imsave(os.path.join(output_path, filename), image_array, cmap='gray')

def save_numpy(array, output_path, filename):
    """
    Save a numpy array to the specified path.

    Args:
        array (np.ndarray): Numpy array to save.
        output_path (str): Path to save the array.
        filename (str): Filename for the saved numpy file.
    """
    np.save(os.path.join(output_path, filename), array)

def main():
    args = parse_arguments()

    # Set paths and parameters
    mask_path = args.mask
    output_path = args.outpath
    resolution = args.resolution
    batch_size = 16
    sigma = 0.05

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize simulation and gradient
    sim = AbbeSim(None, resolution, sigma, defocus=None, batch=True, par=False)

    # Create output directories
    image_output_path = os.path.join(output_path, "images")
    numpy_output_path = os.path.join(output_path, "numpys")
    os.makedirs(image_output_path, exist_ok=True)
    os.makedirs(numpy_output_path, exist_ok=True)

    # Process images
    for batch_files, masks in tqdm(load_images_in_batches(mask_path, batch_size, device), desc="Processing batches"):
        intensity = sim(masks)
        intensity_np = intensity.cpu().numpy()

        for i, filename in enumerate(batch_files):
            # Save 
            save_image(intensity_np[i], image_output_path, filename)
            save_numpy(intensity_np[i], numpy_output_path, filename.replace('.png', '.npy'))

if __name__ == "__main__":
    main()
 