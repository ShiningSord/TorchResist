import os
import math
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

from simulator import get_fuilt_simulator, get_iccad13_simulator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Simulate resist process with lithography models.")
    parser.add_argument(
        "--lithomodel", type=str, choices=["ICCAD13", "FUILT"], default="FUILT",
        help="Lithography model to use ('ICCAD13' or 'FUILT'). Default is 'FUILT'."
    )
    parser.add_argument(
        "--lithoresults", type=str, required=True,
        help="Path to save the lithography results (must end with .npy)."
    )
    parser.add_argument(
        "--outpath", type=str, required=True,
        help="Path to save the processed output image."
    )
    parser.add_argument(
        "--resolution", type=float, default=1.0,
        help="Resolution for the simulation. Default is 1.0."
    )
    return parser.parse_args()

def save_image(args):
    """Helper function to save an image using plt.imsave."""
    image, file_path = args
    plt.imsave(file_path, image, cmap='gray')


def save_grayscale_images(array, output_path):
    """
    Save grayscale images from a 3D numpy array to the specified path using multiprocessing.

    Parameters:
        array (numpy.ndarray): Input array of shape (B, H, W).
        output_path (str): Path to save the images.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    num_images = array.shape[0]
    
    # Prepare arguments for multiprocessing
    args = [
        (array[i], os.path.join(output_path, f"resist{i:06d}.png"))
        for i in range(num_images)
    ]

    # Use multiprocessing to save images
    with Pool() as pool:
        pool.map(save_image, args)



def resist(lithomodel, resolution, lithoresults):
    """Perform resist simulation based on the selected lithography model."""
    if lithomodel == "FUILT":
        simulator = get_fuilt_simulator().cuda()
    elif lithomodel == "ICCAD13":
        simulator = get_iccad13_simulator().cuda()
    else:
        raise ValueError(f"Invalid lithography model selected: {lithomodel}")

    aerial_image = np.load(lithoresults)
    aerial_image = torch.from_numpy(aerial_image).to(simulator.device)

    batch_size = 40
    results = []

    with torch.no_grad():
        for i in tqdm(range(math.ceil(aerial_image.shape[0] / batch_size))):
            start = i * batch_size
            end = min((i + 1) * batch_size, aerial_image.shape[0])
            sub_res = simulator.forward(aerial_image[start:end], dx=resolution)
            sub_res = (sub_res > simulator.threshold).cpu().numpy().astype(bool)
            results.append(sub_res)

    return np.concatenate(results, axis=0)


def main():
    """Main function to handle the simulation and save results."""
    args = parse_args()

    # Perform resist simulation
    result = resist(args.lithomodel, args.resolution, args.lithoresults)

    # Ensure output directories exist
    images_dir = os.path.join(args.outpath, "images")
    numpys_dir = os.path.join(args.outpath, "numpys")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(numpys_dir, exist_ok=True)

    # Save the results
    result_path = os.path.join(numpys_dir, "resist.npy")
    np.save(result_path, result)
    print(f"Results saved to {result_path}")
    
    if True:
        save_grayscale_images(result, images_dir)
        print(f"Images saved to {images_dir}")


if __name__ == "__main__":
    main()
