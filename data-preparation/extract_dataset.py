#!/usr/bin/env python3
# extract_nyuv2_hdf5storage.py

import os
import argparse
import numpy as np
import hdf5storage
from PIL import Image
from tqdm import tqdm

def main(mat_path, out_root="nyu2_processed"):
    # --- Load the .mat ---------------------------------------------------
    data = hdf5storage.loadmat(mat_path)
    # data['images'] shape → (480, 640, 3, 1449) uint8
    # data['depths'] shape → (480, 640, 1449) float32
    imgs  = data['images']
    deps  = data['depths']
    N = imgs.shape[3]
    print(f"Found {N} frames (should be 1449)")

    # --- Prepare output folders ------------------------------------------
    rgb_dir   = os.path.join(out_root, 'rgb')
    depth_dir = os.path.join(out_root, 'depth')
    os.makedirs(rgb_dir,   exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # --- Iterate and save ------------------------------------------------
    for i in tqdm(range(N), desc="Saving frames"):
        # RGB is already H×W×3 uint8
        rgb = imgs[:, :, :, i]

        # Depth is in meters as float32; normalize to 0–255 for PNG
        depth = deps[:, :, i].astype(np.float32)
        dmin, dmax = np.nanmin(depth), np.nanmax(depth)
        depth_png = ((depth - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)

        Image.fromarray(rgb).save(os.path.join(rgb_dir,   f"{i:04d}.png"))
        Image.fromarray(depth_png).save(os.path.join(depth_dir, f"{i:04d}.png"))

    print(f"Extraction complete. Files in '{out_root}/'")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extract 1449 RGB–depth pairs from NYU-Depth V2 v7.3 .mat"
    )
    p.add_argument('--mat', required=True,
                   help="Path to nyu_depth_v2_labeled.mat")
    p.add_argument('--out', default="nyu2_processed",
                   help="Output directory for rgb/ and depth/")
    args = p.parse_args()
    main(args.mat, args.out)
