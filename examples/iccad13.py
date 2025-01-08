# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from simulator import LithoSim
from tqdm import tqdm

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process images with ICCAD13 Simulator.")
    parser.add_argument("--mask", required=True, type=str, help="Path to the folder containing input mask images.")
    parser.add_argument("--outpath", required=True, type=str, help="Path to the output directory.")
    parser.add_argument("--config", required=True, type=str, help="Path to the config file.")
    return parser.parse_args()

def load_images_one_by_one(folder_path, device):
    """
    Load images one by one from the specified folder.

    Args:
        folder_path (str): Path to the folder containing images.
        device (torch.device): Device to load images onto.

    Yields:
        tuple: Image filename and corresponding tensor.
    """
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    image_files.sort()

    for file in image_files:
        image_path = os.path.join(folder_path, file)
        image_np = np.array(Image.open(image_path).convert('L'), dtype=np.float32)
        image_tensor = torch.from_numpy(image_np).to(device) / 255.0
        yield (file, image_tensor)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize simulation and gradient
    sim = LithoSim(args.config)

    # Create output directories
    image_output_path = os.path.join(output_path, "images")
    numpy_output_path = os.path.join(output_path, "numpys")
    os.makedirs(image_output_path, exist_ok=True)
    os.makedirs(numpy_output_path, exist_ok=True)

    # Process images
    for filename, mask in tqdm(load_images_one_by_one(mask_path, device), desc="Processing batches"):
        intensity = sim.sim(mask)[0]
        intensity_np = intensity.cpu().numpy()

        save_image(intensity_np, image_output_path, filename)
        save_numpy(intensity_np, numpy_output_path, filename.replace('.png', '.npy'))

if __name__ == "__main__":
    main()
 