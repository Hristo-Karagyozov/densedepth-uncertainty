import os
import csv

rgb_dir = "rgb"
depth_dir = "depth"
csv_path = "nyu2_train.csv"

# Get sorted lists of RGB and depth files
rgb_files = sorted(os.listdir(rgb_dir))
depth_files = sorted(os.listdir(depth_dir))

# Write to CSV
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["rgb", "depth"])
    for rgb_file, depth_file in zip(rgb_files, depth_files):
        writer.writerow([os.path.join(rgb_dir, rgb_file), os.path.join(depth_dir, depth_file)])