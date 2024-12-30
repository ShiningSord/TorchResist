#!/bin/sh
TARGET_DIR="./data/MetalSet/1nm/images" # where you want to save mask data

# Data paths
ARCHIVE_FILE="$1"  # path of initial dataset lithodata.tar.gz
SAVE_DIR="./lithodata"  # path of unzipped data
EXTRACT_DIR="./lithodata/target" 

# unzip lithodata.tar.gz
mkdir -p "$SAVE_DIR"
tar xvfz "$ARCHIVE_FILE" -C "$SAVE_DIR" --strip-components=1 MetalSet/target

# Save to TARGET_DIR and rename files 
mkdir -p "$TARGET_DIR"
cp -r "$EXTRACT_DIR/"* "$TARGET_DIR"
chmod +x tools/rename_images.py
python3 tools/rename_images.py "$TARGET_DIR" 


