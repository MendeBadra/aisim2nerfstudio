import open3d as o3d
import sys
import os

# usage: python view_ply.py your_file.ply
DATASET_FOLDER = "aisim_ns_dataset_lidar"
file_name = "lidar_world_aligned.ply" # Change this if not running from command line

filename = os.path.join(DATASET_FOLDER, file_name)

if len(sys.argv) > 1:
    filename = sys.argv[1]

if not os.path.exists(filename):
    print(f"Error: {filename} not found.")
    sys.exit()

print(f"Loading {filename}...")
pcd = o3d.io.read_point_cloud(filename)
print(f"Points: {len(pcd.points):,}")

# Visualize
# We add a coordinate frame (X=Red, Y=Green, Z=Blue) at 0,0,0 to help you see orientation
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

print("Opening viewer... (Press 'H' in window for help, 'Q' to close)")
o3d.visualization.draw_geometries([pcd, axes])