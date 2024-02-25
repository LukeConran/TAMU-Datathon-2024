

import os
import shutil

# Path to the dataset directory containing PNG files
dataset_dir = '/path/to/your/dataset'

# Path to the new folder where you want to move the files
new_folder = '/path/to/your/new/folder'

# Ensure the new folder exists, create it if it doesn't
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# Initialize a counter to keep track of the number of files processed
counter = 0

# Iterate over files in the dataset directory
for filename in os.listdir(dataset_dir):
    if filename.endswith(".png"):  # Filter PNG files
        counter += 1
        if counter % 50 == 0:  # Move every fiftieth file
            src = os.path.join(dataset_dir, filename)
            dst = os.path.join(new_folder, filename)
            shutil.move(src, dst)
            print(f"Moved {filename} to {new_folder}")

print("File moving completed.")