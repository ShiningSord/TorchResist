import os
import numpy as np

def compute_average_l0_distance(folder1, folder2):
    """
    Compute the average L0 distance between identically named numpy arrays in two folders.

    Parameters:
        folder1 (str): Path to the first folder containing numpy arrays.
        folder2 (str): Path to the second folder containing numpy arrays.

    Returns:
        float: The average L0 distance between the arrays.
    """
    # List all files in the two folders
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # Find the common files in both folders
    common_files = files1.intersection(files2)

    if not common_files:
        print("No common files found between the two folders.")
        return 0.0

    total_l0_distance = 0
    count = 0

    # Iterate through each common file
    for filename in common_files:
        # Load the numpy arrays
        array1 = np.load(os.path.join(folder1, filename))
        array2 = np.load(os.path.join(folder2, filename))

        # Ensure both arrays have the same shape
        if array1.shape != array2.shape:
            print(f"Skipping {filename} due to shape mismatch: {array1.shape} vs {array2.shape}")
            continue

        # Compute the L0 distance (count of differing elements)
        l0_distance = np.sum(array1 != array2)
        total_l0_distance += l0_distance
        count += 1

    if count == 0:
        print("No valid common files to compare.")
        return 0.0

    # Compute and return the average L0 distance
    average_l0_distance = total_l0_distance / count
    return average_l0_distance

# Example usage
if __name__ == "__main__":
    folder1 = "path_to_folder1"  # Replace with the actual path to the first folder
    folder2 = "path_to_folder2"  # Replace with the actual path to the second folder

    average_l0 = compute_average_l0_distance(folder1, folder2)
    print(f"The average L0 distance is: {average_l0}")