import laspy
import numpy as np
import open3d as o3d
import glob
import os

# --- CONFIGURATION ---
INPUT_FOLDER = "aisim_ns_dataset_lidar\ego_lidar_sensor_las"   # Folder containing your .las files
OUTPUT_FILENAME = "aisim_ns_dataset_lidar/lidar_merged.ply"
USE_INTENSITY_AS_COLOR = True # Set to False if you want pure white points
# ---------------------

def get_files():
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.las"))
    files += glob.glob(os.path.join(INPUT_FOLDER, "*.laz"))
    return sorted(files)

def process_lidar():
    files = get_files()
    if not files:
        print(f"No .las/.laz files found in {INPUT_FOLDER}")
        return

    all_points = []
    all_colors = []

    print(f"Found {len(files)} Lidar files. Processing...")

    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        
        try:
            with laspy.open(filepath) as f:
                # 1. CHECK POINT COUNT BEFORE READING
                if f.header.point_count == 0:
                    print(f"[{i+1}/{len(files)}] Skipping {filename} (Empty: 0 points)")
                    continue

                # Read the data
                las = f.read()
                
                # 2. Extract XYZ
                points = np.vstack((las.x, las.y, las.z)).transpose()
                all_points.append(points)

                # 3. Extract Color
                if USE_INTENSITY_AS_COLOR:
                    if hasattr(las, 'intensity') and len(las.intensity) > 0:
                        intensity = np.array(las.intensity, dtype=np.float32)
                        
                        # Normalize
                        max_val = np.percentile(intensity, 99) 
                        if max_val > 0:
                            intensity /= max_val
                        intensity = np.clip(intensity, 0, 1)
                        
                        colors = np.stack([intensity, intensity, intensity], axis=1)
                        all_colors.append(colors)
                    
                    elif hasattr(las, 'red') and len(las.red) > 0:
                        red = np.array(las.red)
                        green = np.array(las.green)
                        blue = np.array(las.blue)
                        scale = 65535.0 if np.max(red) > 255 else 255.0
                        colors = np.vstack((red, green, blue)).transpose() / scale
                        all_colors.append(colors)
                    else:
                        white = np.ones_like(points)
                        all_colors.append(white)
                else:
                    white = np.ones_like(points)
                    all_colors.append(white)
        
        # Only print if valid points were actually processed
            if i % 10 == 0:
                print(f"[{i+1}/{len(files)}] Processed {filename}...")

        except Exception as e:
            print(f"   -> Error reading {filename}: {e}")

    # --- MERGE ---
    print("Merging point clouds...")
    if not all_points:
        print("Error: No valid points found in any file.")
        return

    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)

    print(f"Total Points: {len(merged_points)}")

    # --- SAVE ---
    print(f"Saving to {OUTPUT_FILENAME}...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    pcd.colors = o3d.utility.Vector3dVector(merged_colors)
    o3d.io.write_point_cloud(OUTPUT_FILENAME, pcd, write_ascii=False)
    
    print("Success!")

if __name__ == "__main__":
    process_lidar()

# --- Test config
# INPUT_FOLDER = "depth"

# las = laspy.read(INPUT_FOLDER + '/lidar_sensor_00002.las')
# print(np.unique(las.classification))
