#!/bin/sh
TARGET_DIR="./data/Dataset1/1nm/images" # where you want to save mask data

# Data paths
ARCHIVE_FILE="$1"  # path of initial dataset lithodata.tar.gz
SAVE_DIR="./lithodata"  # path of unzipped data
EXTRACT_DIR="./lithodata/MetalSet/target" 

# unzip lithodata.tar.gz
# mkdir -p "$SAVE_DIR"
# tar xvfz "$ARCHIVE_FILE" -C "$SAVE_DIR"

# Save to TARGET_DIR and rename files 
mkdir -p "$TARGET_DIR"
cp -r "$EXTRACT_DIR/"* "$TARGET_DIR"
chmod +x tools/rename_images.py
python3 tools/rename_images.py "$TARGET_DIR" 


# python3 tools/rename_images.py /research/d5/gds/zxwang22/storage/resist/cells/png/1nm 


