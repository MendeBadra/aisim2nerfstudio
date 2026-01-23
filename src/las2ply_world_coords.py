import laspy
import numpy as np
import open3d as o3d
import json
import os
import glob

# --- CONFIGURATION ---
DATASET_FOLDER = 'aisim_ns_dataset_lidar'
LIDAR_FOLDER = f"{DATASET_FOLDER}/ego_lidar_sensor_las"
TRANSFORMS_FILE = f"{DATASET_FOLDER}/transforms.json"
OUTPUT_FILENAME = f"{DATASET_FOLDER}/lidar_world_aligned.ply"
VOXEL_SIZE = 0.1  # Downsample grid size (meters)

# Toggle this True if your final road looks vertical/sideways
FIX_ORIENTATION = True
# ---------------------

def get_lidar_files(folder):
    """Finds and sorts all .las/.laz files in the directory."""
    files = glob.glob(os.path.join(folder, "*.las"))
    files += glob.glob(os.path.join(folder, "*.laz"))
    return sorted(files)

def load_transforms(json_path):
    """Loads the Nerfstudio JSON and sorts frames by filename."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Sort to ensure Frame 0 matches Lidar File 0
    frames = data['frames']
    frames.sort(key=lambda x: x['file_path'])
    return frames

def apply_matrix_transform(points, matrix):
    """
    Applies a 4x4 transformation matrix to an array of 3D points.
    Input: (N, 3) array, 4x4 matrix
    Output: (N, 3) array
    """
    # Convert [x,y,z] to [x,y,z,1] (Homogeneous coordinates)
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack((points, ones))
    
    # Multiply: (Matrix * Point_Transpose).Transpose
    transformed_hom = (matrix @ points_hom.T).T
    
    # Return only [x,y,z]
    return transformed_hom[:, :3]

def process_single_frame(las_path, transform_matrix):
    """
    Reads one Lidar file and moves its points to World Coordinates
    using the car's position (transform_matrix).
    """
    try:
        with laspy.open(las_path) as f:
            # Skip empty files (common in simulation warm-up)
            if f.header.point_count == 0:
                return None

            las = f.read()
            # 1. Get Local Points (Relative to Car)
            local_points = np.vstack((las.x, las.y, las.z)).transpose()

            # 2. (Optional) Fix Lidar Axis Orientation
            # Rotates Lidar (X-forward) to Camera (Z-back) conventions if needed
            if FIX_ORIENTATION:
                rotation_fix = np.array([
                    [0, -1, 0, 0], 
                    [0, 0, 1, 0], 
                    [1, 0, 0, 0], 
                    [0, 0, 0, 1]
                ])
                local_points = apply_matrix_transform(local_points, rotation_fix)

            # 3. Move to World Space (Local -> Global)
            world_points = apply_matrix_transform(local_points, transform_matrix)
            return world_points

    except Exception as e:
        print(f"Warning: Failed to read {las_path}: {e}")
        return None

def save_point_cloud(points_list, output_path):
    """Merges, downsamples, and saves the final result."""
    if not points_list:
        print("Error: No points found to save.")
        return

    print("Merging chunks...")
    merged_points = np.vstack(points_list)
    print(f" -> Total Raw Points: {len(merged_points):,}")

    print(f"Downsampling (Voxel Size: {VOXEL_SIZE}m)...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    
    # Optimize: Remove duplicates to save memory
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f" -> Optimized Points: {len(pcd.points):,}")

    print(f"Saving to {output_path}...")
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)
    print("Success.")

def main():
    print("--- Starting Lidar Alignment ---")
    
    # 1. Load Data
    las_files = get_lidar_files(LIDAR_FOLDER)
    frames = load_transforms(TRANSFORMS_FILE)
    
    # Match the counts (prevent index errors)
    count = min(len(las_files), len(frames))
    print(f"Processing {count} aligned frames...")

    all_world_points = []

    # 2. Process Loop
    for i in range(count):
        # Extract the pose matrix for this specific frame
        matrix = np.array(frames[i]['transform_matrix'])
        
        # Process the file
        points = process_single_frame(las_files[i], matrix)
        
        if points is not None:
            all_world_points.append(points)
            
        # Progress Bar
        if i % 50 == 0:
            print(f"Processed {i}/{count}...")

    # 3. Save
    save_point_cloud(all_world_points, OUTPUT_FILENAME)

if __name__ == "__main__":
    main()