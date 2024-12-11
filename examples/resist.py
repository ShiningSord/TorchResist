import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
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
        help="Path to save the lithography results."
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


def save_grayscale_images(array, output_path, all_name_list):
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
        (array[i], os.path.join(output_path, all_name_list[i].replace('.npy', '.png')))
        for i in range(num_images)
    ]

    # Use multiprocessing to save images
    with Pool() as pool:
        pool.map(save_image, args)


def load_npy_batches(folder_path, batch_size):
    """
    Iteratively load .npy files from a folder in batches.

    Parameters:
        folder_path (str): Path to the folder containing .npy files.
        batch_size (int): Number of .npy files to include in each batch.

    Yields:
        tuple: (name_list, array) where
            - name_list (list of str): List of file names in the batch.
            - array (numpy.ndarray): Batch array of shape (B, H, W).
    """
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    file_list.sort()  # Ensure files are loaded in a consistent order

    for i in range(0, len(file_list), batch_size):
        batch_names = file_list[i:i + batch_size]
        batch_arrays = [np.load(os.path.join(folder_path, file_name)) for file_name in batch_names]
        yield batch_names, np.stack(batch_arrays)



def resist(lithomodel, resolution, lithoresults, numpy_dir):
    """Perform resist simulation based on the selected lithography model and save results."""
    if lithomodel == "FUILT":
        simulator = get_fuilt_simulator().cuda()
    elif lithomodel == "ICCAD13":
        simulator = get_iccad13_simulator().cuda()
    else:
        raise ValueError(f"Invalid lithography model selected: {lithomodel}")

    batch_size = 2
    results = []
    all_name_list = []

    with torch.no_grad():
        for name_list, aerial_image_batch in tqdm(load_npy_batches(lithoresults, batch_size), desc="Processing batches"):
            aerial_image_batch = torch.from_numpy(aerial_image_batch).to(simulator.device)
            sub_res = simulator.forward(aerial_image_batch, dx=resolution)
            sub_res = (sub_res > simulator.threshold).cpu().numpy().astype(bool)
            results.append(sub_res)
            all_name_list.extend(name_list)

            # Save each result in the batch to the specified directory
            for name, result in zip(name_list, sub_res):
                file_path = os.path.join(numpy_dir, name)
                np.save(file_path, result)

    print("Simulation results saved successfully.")
    return all_name_list, np.concatenate(results, axis=0)





def main():
    """Main function to handle the simulation and save results."""
    args = parse_args()
    
    # Ensure output directories exist
    images_dir = os.path.join(args.outpath, "images")
    numpys_dir = os.path.join(args.outpath, "numpys")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(numpys_dir, exist_ok=True)

    # Perform resist simulation
    all_name_list, result = resist(args.lithomodel, args.resolution, args.lithoresults, numpys_dir)
    
    if True:
        save_grayscale_images(result, images_dir, all_name_list)
        print(f"Images saved to {images_dir}")


if __name__ == "__main__":
    main()
