import os
import sys
from tqdm import tqdm
from multiprocessing import Pool

def process_file(filename, source_dir):
    if filename.startswith('cell') and filename.endswith('.png'):
        # Extract the numeric part of the filename
        index = int(filename[4:-4])  # Extract the number after 'cell' and before '.png'
        
        # Generate a new filename in the format 'cell000000.png'
        new_filename = f'cell{index:06}.png'
        
        # Construct the full old and new file paths
        old_file_path = os.path.join(source_dir, filename)
        new_file_path = os.path.join(source_dir, new_filename)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        # print(f'Renamed: {old_file_path} to {new_file_path}')

def rename_images_multiprocess(source_dir):
    # List all files in the source directory
    files = os.listdir(source_dir)

    # Use a multiprocessing Pool to process files in parallel
    with Pool() as pool:
        pool.starmap(process_file, [(filename, source_dir) for filename in tqdm(files)])

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python rename_images.py <source folder path>")
        sys.exit(1)

    # Get the source folder path from the command line arguments
    source = sys.argv[1]

    # Call the function to rename images using multiprocessing
    rename_images_multiprocess(source)