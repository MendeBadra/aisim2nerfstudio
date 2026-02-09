import open3d as o3d
import json
import numpy as np

# --- CONFIGURATION ---
PLY_FILE = "lidar_world_aligned.ply"
JSON_FILE = "transforms.json"
FRUSTUM_SCALE = 1.0  # Size of the camera pyramid in the viewer
SKIP_FRAMES = 10     # Only draw every Nth frame to avoid clutter
# ---------------------

def get_camera_frustum(img_w, img_h, K, W2C, color=[1, 0, 0], scale=1.0):
    """
    Creates a line set representing the camera frustum (pyramid).
    """
    # 1. Define Camera center and Image Plane corners in Camera Space
    # Nerfstudio/OpenGL Convention: -Z is forward, +Y is up.
    # We draw the frustum pointing along -Z.
    
    # Inverse Intrinsics to unproject pixels
    K_inv = np.linalg.inv(K)
    
    # Corners of the image plane at Z=1 (Normalized Device Coordinates)
    # BL, BR, TR, TL
    corners_pix = np.array([
        [0, img_h, 1],
        [img_w, img_h, 1],
        [img_w, 0, 1],
        [0, 0, 1]
    ]).T # 3x4
    
    # Project to Camera Space (multiply by inv K)
    corners_cam = K_inv @ corners_pix
    
    # Normalize to be at distance 'scale'
    corners_cam = corners_cam * scale
    
    # Make them 4D (homogenous)
    corners_cam = np.vstack((corners_cam, np.ones((1, 4)))) # 4x4
    center_cam = np.array([[0], [0], [0], [1]])             # 4x1
    
    # 2. Transform to World Space (using C2W matrix)
    # Note: JSON usually provides C2W directly.
    # If W2C was passed, invert it. Here we assume input is C2W.
    C2W = W2C 
    
    corners_world = (C2W @ corners_cam)[:3, :].T  # 4x3
    center_world = (C2W @ center_cam)[:3, 0]      # 1x3
    
    # 3. Create Lines (Connect center to corners, and corners to each other)
    points = np.vstack((center_world, corners_world)) # 5 points total
    
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4], # Center to corners
        [1, 2], [2, 3], [3, 4], [4, 1]  # Image plane rectangle
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set

def main():
    # 1. Load Point Cloud
    print(f"Loading {PLY_FILE}...")
    pcd = o3d.io.read_point_cloud(PLY_FILE)
    
    # 2. Load JSON
    with open(JSON_FILE, 'r') as f:
        meta = json.load(f)
        
    frames = meta['frames']
    fl_x = meta.get('fl_x', 500)
    fl_y = meta.get('fl_y', 500)
    cx = meta.get('cx', 320)
    cy = meta.get('cy', 240)
    w = meta.get('w', 640)
    h = meta.get('h', 480)
    
    # Build Intrinsic Matrix K
    K = np.array([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ])

    geometries = [pcd]
    
    print(f"Generating frustums for {len(frames)} frames...")
    
    # 3. Create Frustums
    for i in range(0, len(frames), SKIP_FRAMES):
        frame = frames[i]
        c2w = np.array(frame['transform_matrix'])
        
        # Color code based on camera name
        fname = frame['file_path']
        if "pinhole_duplicate0" in fname:
            col = [1, 0, 0] # Red (Right?)
        elif "pinhole_duplicate1" in fname:
            col = [0, 1, 0] # Green (Back?)
        elif "pinhole_duplicate2" in fname:
            col = [0, 0, 1] # Blue (Left?)
        else:
            col = [1, 1, 0] # Yellow (Front/Center)
            
        frustum = get_camera_frustum(w, h, K, c2w, color=col, scale=FRUSTUM_SCALE)
        geometries.append(frustum)

    # 4. Add Coordinate Frame (Origin)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0,0,0])
    geometries.append(axes)

    print("Visualizing... (Mouse to rotate)")
    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    main()