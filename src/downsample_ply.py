import open3d as o3d
import os

# --- CONFIG ---
INPUT_FILE = "aisim_ns_dataset_lidar/lidar_merged.ply"
OUTPUT_FILE = "aisim_ns_dataset_lidar/lidar_optimized.ply"
VOXEL_SIZE = 0.1  # 0.1 means 10cm. Increase to 0.2 or 0.3 if file is still too big.
# --------------

def downsample():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    print(f"Loading {INPUT_FILE} (This may take a minute for 53M points)...")
    pcd = o3d.io.read_point_cloud(INPUT_FILE)
    
    original_count = len(pcd.points)
    print(f"Original Points: {original_count:,}")

    print(f"Downsampling with voxel size {VOXEL_SIZE}m...")
    # This averages all points inside a 10cm box into a single point
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    
    new_count = len(downsampled_pcd.points)
    print(f"New Points:      {new_count:,}")
    
    reduction = (1 - (new_count / original_count)) * 100
    print(f"Reduction:       {reduction:.1f}% removed")

    print(f"Saving to {OUTPUT_FILE}...")
    o3d.io.write_point_cloud(OUTPUT_FILE, downsampled_pcd, write_ascii=False)
    print("Done! Use this new file for training.")

if __name__ == "__main__":
    downsample()