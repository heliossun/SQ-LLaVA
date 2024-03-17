#!/bin/bash

# Define the path to the text file containing the image names
IMAGE_LIST="/home/gs4288/PycharmProjects/Visual-self-QA/playground/data/eval/MME/MME_Benchmark_release_version/artwork/images/image_list.txt"

# Define the target directory where you want to copy the images
TARGET_DIR="/home/gs4288/PycharmProjects/Visual-self-QA/playground/data/eval/MME/MME_Benchmark_release_version/artwork/images"

# Check if the target directory exists, if not, create it
if [ ! -d "$TARGET_DIR" ]; then
  mkdir -p "$TARGET_DIR"
fi

# Read each line in the image list file
while IFS= read -r image_name; do
  # Copy each image to the target directory
  # Add full path before "$image_name" if the images are not in the current directory
  cp /home/gs4288/PycharmProjects/Visual-self-QA/playground/data/eval/toy_dataset/$image_name "$TARGET_DIR"
done < "$IMAGE_LIST"

echo "All images have been copied to $TARGET_DIR."
