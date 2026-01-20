import laspy
import numpy as np
import open3d as o3d
import os

# --- CONFIGURATION ---
INPUT_LAS = "depth\lidar_sensor_00002.las"  # Change this to your file
OUTPUT_PLY = "lidar_nerfstudio_test.ply"
# ---------------------

def convert_las_to_ply():
    if not os.path.exists(INPUT_LAS):
        print(f"Error: Could not find {INPUT_LAS}")
        return

    print(f"Loading {INPUT_LAS}...")
    
    # 1. Read LAS file
    las = laspy.read(INPUT_LAS)
    
    # 2. Extract Coordinates
    # aiSim usually exports standard XYZ. 
    # NOTE: If your coordinates are massive (e.g., GPS/UTM), 
    # you might need to shift them to 0,0,0 here.
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    print(f" -> Found {len(points)} points.")

    # 3. Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 4. Extract Colors (Optional)
    # If aiSim exported intensity or RGB, we can try to use it.
    try:
        if hasattr(las, 'red'):
            # LAS colors are often 16-bit (0-65535), convert to 0-1 float
            colors = np.vstack((las.red, las.green, las.blue)).transpose()
            if np.max(colors) > 255:
                colors = colors / 65535.0
            else:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
            print(" -> Colors found and processed.")
        elif hasattr(las, 'intensity'):
            # Use intensity as a grayscale color
            intensity = np.array(las.intensity)
            intensity = intensity / np.max(intensity) # Normalize 0-1
            # Stack to make R=G=B=Intensity
            colors = np.stack([intensity, intensity, intensity], axis=1)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            print(" -> Intensity found, converting to grayscale.")
    except Exception as e:
        print(f"Warning: Could not process colors: {e}")

    # 5. Save as PLY
    print(f"Saving to {OUTPUT_PLY}...")
    o3d.io.write_point_cloud(OUTPUT_PLY, pcd, write_ascii=False) # Binary is smaller/faster
    print("Done!")

if __name__ == "__main__":
    convert_las_to_ply()