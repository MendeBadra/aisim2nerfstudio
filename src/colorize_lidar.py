import open3d as o3d
import numpy as np
import json
import cv2
import os

# --- CONFIGURATION ---
PLY_FILE = "lidar_world_aligned.ply"  # The uncolored point cloud
JSON_FILE = "transforms.json"         # Nerfstudio transforms
IMAGE_DIR = "."                       # Root dir where image paths in JSON are relative to
OUTPUT_FILE = "lidar_colored.ply"

# Optimization: Don't use every single frame (too slow). 
# Using every 10th frame is usually enough to color the whole map.
FRAME_STEP = 10 
# ---------------------

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_camera_matrix(json_data, frame_idx):
    """
    Returns the World-to-Camera matrix (Inverse of pose).
    Also handles coordinate conversion from OpenGL (Nerfstudio) to OpenCV.
    """
    frame = json_data['frames'][frame_idx]
    c2w = np.array(frame['transform_matrix']) # Camera to World
    
    # 1. Invert to get World to Camera
    w2c = np.linalg.inv(c2w)
    
    # 2. Nerfstudio is OpenGL convention (-Z forward, +Y Up).
    #    OpenCV projection needs +Z forward, +Y Down.
    #    We need to flip Y and Z axes.
    #    Correction Matrix: 1, 0, 0; 0, -1, 0; 0, 0, -1
    fix_rot = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    # Apply correction: New_W2C = Fix @ Old_W2C
    w2c_cv = fix_rot @ w2c
    return w2c_cv

def main():
    print(f"Loading Point Cloud {PLY_FILE}...")
    pcd = o3d.io.read_point_cloud(PLY_FILE)
    points = np.asarray(pcd.points)
    
    # Initialize colors to Grey (0.5)
    colors = np.full(points.shape, 0.5)
    
    # Add a column of 1s for matrix multiplication (N, 4)
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    
    print(f"Loading JSON {JSON_FILE}...")
    meta = load_json(JSON_FILE)
    
    # Get global intrinsics (assuming all cams are same)
    fl_x = meta.get('fl_x')
    fl_y = meta.get('fl_y')
    cx = meta.get('cx')
    cy = meta.get('cy')
    w = meta.get('w')
    h = meta.get('h')
    
    frames = meta['frames']
    print(f"Projecting colors from {len(frames):,} frames (Step={FRAME_STEP})...")
    
    for i in range(0, len(frames), FRAME_STEP):
        frame = frames[i]
        
        # 1. Load Image
        img_path = os.path.join(IMAGE_DIR, frame['file_path'])
        if not os.path.exists(img_path):
            continue
            
        # Read image (OpenCV reads as BGR, convert to RGB)
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img.shape
        
        # 2. Project Points
        # Convert World Points -> Camera Space
        w2c = get_camera_matrix(meta, i)
        
        # Matrix Multiply: (4x4 @ 4xN).T -> Nx4
        # We process ALL 5 million points at once using vector math (fast!)
        pts_cam = (w2c @ points_hom.T).T
        
        # Extract X, Y, Z
        xyz = pts_cam[:, :3]
        Z = xyz[:, 2]
        
        # 3. Filter: Keep only points In Front of camera (Z > 0)
        # We use a small epsilon 0.1 to avoid division by zero
        valid_z_mask = Z > 0.1
        
        # 4. Perspective Divide (X/Z, Y/Z)
        # Apply intrinsics: u = fx * (x/z) + cx
        u = (xyz[:, 0] * fl_x / Z) + cx
        v = (xyz[:, 1] * fl_y / Z) + cy
        
        # 5. Filter: Keep only points inside image bounds
        valid_u = (u >= 0) & (u < img_w - 1)
        valid_v = (v >= 0) & (v < img_h - 1)
        
        # Combine masks
        final_mask = valid_z_mask & valid_u & valid_v
        
        if np.count_nonzero(final_mask) == 0:
            continue
            
        # 6. Sample Colors
        # Get integer indices
        u_idx = u[final_mask].astype(int)
        v_idx = v[final_mask].astype(int)
        
        # Read colors from image array [row, col] -> [v, u]
        new_colors = img[v_idx, u_idx]
        
        # Normalize 0-255 -> 0.0-1.0
        new_colors = new_colors.astype(float) / 255.0
        
        # Update the global color array
        colors[final_mask] = new_colors
        
        if i % 100 == 0:
            print(f"Processed frame {i}/{len(frames)}...")

    print("Saving colored point cloud...")
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(OUTPUT_FILE, pcd, write_ascii=False)
    print("Done! Open in CloudCompare to verify.")

if __name__ == "__main__":
    main()