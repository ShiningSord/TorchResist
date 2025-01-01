from PIL import Image
import numpy as np
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def process_image(file_info):
    """
    Process a single binary image: downsample it and save the result.
    
    Args:
        file_info (tuple): (input_file_path, output_file_path, downsample_ratio)
    """
    input_file_path, output_file_path, downsample_ratio = file_info
    
    # Load image
    image = Image.open(input_file_path).convert('1')  # Convert to binary (1-bit)
    image_array = np.array(image, dtype=np.uint8)

    # Downsample with block mean
    height, width = image_array.shape
    down_height = int(np.ceil(height / downsample_ratio))
    down_width = int(np.ceil(width / downsample_ratio))

    padded_height = down_height * downsample_ratio
    padded_width = down_width * downsample_ratio

    # Pad image with zeros
    padded_image = np.zeros((padded_height, padded_width), dtype=np.uint8)
    padded_image[:height, :width] = image_array

    # Downsample using mean pooling
    downsampled_image = padded_image.reshape(
        down_height, downsample_ratio, down_width, downsample_ratio
    ).mean(axis=(1, 3)) > 0.5  # Thresholding to keep binary nature

    # Convert back to binary image
    downsampled_image = (downsampled_image * 255).astype(np.uint8)
    downsampled_pil_image = Image.fromarray(downsampled_image)

    # Save the processed image
    downsampled_pil_image.save(output_file_path)
    return f"Processed: {output_file_path}"


def downsample_binary_images_parallel(input_folder, output_folder, downsample_ratio=14):
    """
    Downsample binary images in the input folder using multiprocessing and save them to the output folder.

    Args:
        input_folder (str): Path to the folder containing binary images.
        output_folder (str): Path to the folder to save the downsampled images.
        downsample_ratio (int): Ratio for downsampling (e.g., 14 for 1:14).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Collect file paths
    tasks = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png') or file_name.endswith('.jpg'):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)
            tasks.append((input_file_path, output_file_path, downsample_ratio))

    # Use multiprocessing to process images
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_image, tasks)
        for result in tqdm(results):
            # print(result)
            pass


# Example usage:
if __name__ == "__main__":
    input_folder = "./data/MetalSet/1nm"  # Replace with your input folder path
    output_folder = "./data/MetalSet/7nm"  # Replace with your output folder path

    downsample_binary_images_parallel(input_folder, output_folder, downsample_ratio=7)
