"""
Colorize LAS/LAZ lidar point clouds by projecting points into camera images.

This script is adapted from lidar projection workflows and is designed for datasets
with Nerfstudio-style transform file (camera intrinsics + per-frame camera poses).

Example:
    python src/colorize_las_pointcloud.py \
      --las-path frame_with_depth/lidar_sensor_00002.las \
      --transforms aisim_ns_dataset/transforms.json \
      --images-root aisim_ns_dataset \
      --frame-index 2 \
      --coord-convention nerfstudio \
      --output frame_with_depth/lidar_sensor_00002_colorized.ply
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import laspy
import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt

import pyvista as pv

def pv_plot(points: np.ndarray):
    pl = pv.Plotter()
    pl.add_points(points, color="black", point_size=2, render_points_as_spheres=True)
    pl.add_axes()        # orientation widget
    pl.add_axes_at_origin(labels_off=False, line_width=5)
    pl.show_grid()       # world grid
    pl.show()

# undo of the permutation defined in def nerfstudio_conversion(T_matrix):
# Nerfstudio | Meaning | AISIM axis
#    +X      | right   |   -Y    
#    +Y      |   up    |    Z
#    +Z      | backward|   -X
T_permutation = np.array([ 
            [0, -1, 0, 0],      
            [0,  0, 1, 0],
            [-1, 0, 0, 0],
            [0,  0, 0, 1]
            ])

# from src.calculation_for_transformsfile import calculate_transform_matrix

# LIDAR_FILE = "../calibrations/lidar_sensor.json"
# VEHICLE_SENSOR_FILE = "../data/2025-12-04_18-22-25/ego/vehicle_sensor/vehicle_sensor_00000.json"
# SENSOR_TYPE = "lidar_sensor" # for the json 

# CONVENTION = "nerfstudio"

@dataclass
class CameraView:
    image: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    T_w2c: np.ndarray


def apply_transform_to_points(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to Nx3 points."""
    points_h = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)])
    transformed_h = (T @ points_h.T).T
    return transformed_h[:, :3] / transformed_h[:, 3:4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colorize LAS points from camera images")
    parser.add_argument("--las-path", type=Path, required=True, help="Input LAS/LAZ file")
    parser.add_argument(
        "--transform",
        type=Path,
        required=True,
        help="Single Nerfstudio-style transforms.json file containing all the camera streams",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="Root folder used to resolve frame file_path entries from transforms",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame index to use (e.g. 2 -> *_00002.jpg). If omitted, inferred from LAS filename.",
    )
    parser.add_argument(
        "--depth-threshold",
        type=float,
        default=0.1,
        help="Minimum positive depth to consider a projected point valid",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Maximum depth to consider for projection. If omitted, no upper bound is applied.",
    )
    parser.add_argument(
        "--debug-stats",
        action="store_true",
        help="Print per-camera projection statistics.",
    )
    parser.add_argument(
        "--lidar-calibration-file",
        type=Path,
        default=None,
        help="Calibration JSON containing lidar mounting data (used to compute lidar->world).",
    )
    parser.add_argument(
        "--vehicle-sensor-file",
        type=Path,
        default=None,
        help="vehicle_sensor_XXXXX.json file used to compute lidar->world at the matching frame.",
    )
    parser.add_argument(
        "--lidar-sensor-type",
        type=str,
        default="lidar_sensor",
        help="Sensor key inside calibration file for LiDAR mounting data.",
    )
    parser.add_argument(
        "--lidar-to-world-convention",
        choices=["nerfstudio", "aisim"],
        default="nerfstudio",
        help="Convention used when computing lidar->world transform.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file (.ply, .las, or .laz)",
    )
    return parser.parse_args()

# NOTE: Not tested
def infer_frame_index(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        return 0
    return int(match.group(1))

# NOTE: perhaps better not to use this


def resolve_image_path(transforms_path: Path, images_root: Path, file_path: str) -> Path:
    file_path_obj = Path(file_path)

    candidates = []
    if file_path_obj.is_absolute():
        candidates.append(file_path_obj)
    candidates.append(images_root / file_path_obj)
    candidates.append(images_root / file_path_obj.name)
    candidates.append(transforms_path.parent / file_path_obj)
    candidates.append(transforms_path.parent / file_path_obj.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve image path '{file_path}' from transforms '{transforms_path}'. "
        f"Tried under '{images_root}' and '{transforms_path.parent}'."
    )


def select_frames(frames: Iterable[dict], frame_index: int) -> list[dict]:
    suffix = f"_{frame_index:05d}"
    matches = []
    for frame in frames:
        frame_name = Path(frame["file_path"]).stem
        if frame_name.endswith(suffix):
            matches.append(frame)

    if not matches:
        raise ValueError(f"No frames matching index {frame_index} found.")
    return matches

# tested
def load_camera_views(transforms_path: Path, images_root: Path, frame_index: int) -> list[CameraView]:
    # undo of the permutation defined in def nerfstudio_conversion(T_matrix):
    # Nerfstudio | Meaning | AISIM axis
    #    +X      | right   |   -Y    
    #    +Y      |   up    |    Z
    #    +Z      | backward|   -X

    # T_permutation = np.array([ 
    #         [0, -1, 0, 0],      
    #         [0,  0, 1, 0],
    #         [-1, 0, 0, 0],
    #         [0,  0, 0, 1]
    #         ])


    with transforms_path.open("r", encoding="utf-8") as f:
        tf = json.load(f)

    frame_dicts = select_frames(tf["frames"], frame_index)
    # print(f"{frame_dicts=}")
    views = []
    for frame in frame_dicts:
        image_path = resolve_image_path(transforms_path, images_root, frame["file_path"])
        # print(image_path)
        image = np.array(Image.open(image_path).convert("RGB"))
        T_c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        # permute the rows
        # NOTE: check the note in calculation_for_transformsfile def nerfstudio
        T_c2w = T_c2w @ T_permutation
        # print(T_c2w)
        T_w2c = np.linalg.inv(T_c2w)
        # print(f"{frame['file_path']=} , \n {image_path=}")
        views.append(CameraView(
            image=image,
            fx=float(tf["fl_x"]),
            fy=float(tf["fl_y"]),
            cx=float(tf["cx"]),
            cy=float(tf["cy"]),
            k1=float(tf.get("k1", 0.0)),
            k2=float(tf.get("k2", 0.0)),
            p1=float(tf.get("p1", 0.0)),
            p2=float(tf.get("p2", 0.0)),
            T_w2c=T_w2c,
        ))
    return views



def project_points(
    points: np.ndarray,
    camera: CameraView,
    depth_threshold: float,
    max_depth: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard pinhole camera projection.
    
    This implementation follows the user's coordinate system where:
    - Depth is points[:, 0]
    - Horizontal projection on Y (index 1)
    - Vertical projection on Z (index 2)
    """
    
    depth = points[:, 0]
    valid_mask = depth > depth_threshold
    
    if max_depth is not None:
        valid_mask &= depth <= max_depth

    # Avoid division by zero
    valid_mask &= (depth != 0)
    
    # Normalized coordinates
    # u = cx - fx * (y / depth)
    # v = cy - fy * (z / depth)
    u = camera.cx - camera.fx * (points[:, 1] / depth)
    v = camera.cy - camera.fy * (points[:, 2] / depth)

    h, w = camera.image.shape[:2]
    x_pix = np.floor(u).astype(np.int32)
    y_pix = np.floor(v).astype(np.int32)

    valid = (
        valid_mask
        & (x_pix >= 0)
        & (x_pix < w)
        & (y_pix >= 0)
        & (y_pix < h)
    )
    return valid, depth, x_pix, y_pix

def colorize_points(
    points_lidar: np.ndarray,
    cameras: list[CameraView],
    T_lidar_to_world: np.ndarray,
    depth_threshold: float,
    max_depth: float | None,
    debug_stats: bool = False,
) -> np.ndarray:
    """
    Colorize points by projecting them into multiple camera views.
    Includes occlusion pruning and "white-out" logic for distant points.
    """
    n_points = points_lidar.shape[0]
    point_rgb = np.full((n_points, 3), 255, dtype=np.uint8)
    point_depth = np.full(n_points, 9000.0, dtype=np.float32)

    points_world = apply_transform_to_points(points_lidar, T_lidar_to_world)
    # DEBUG
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_world)
    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd, coord_frame])
    pv_plot(points_world)
    
    for cam_index, cam in enumerate(cameras):
        # 1. Transform points from Lidar space to this Camera's space
        # plt.imshow(cam.image)
        # plt.title(f"{cam_index=}")
        # plt.show()
        # T_lidar_to_camera = cam.T_w2c @ 
        points_camera = apply_transform_to_points(points_world, cam.T_w2c)
        # DEBUG
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_camera)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, coord_frame])

        # 2. Project 3D camera points to 2D pixel coordinates
        valid, depths, x_pix, y_pix = project_points(
            points_camera,
            cam,
            depth_threshold,
            max_depth,
        )

        idx = np.flatnonzero(valid)
        if debug_stats:
            print(f"[DEBUG] camera[{cam_index}] valid projections: {idx.size}/{n_points}")

        if idx.size == 0:
            continue

        # occlusion mapping for this camera view
        h, w = cam.image.shape[:2]
        pixel_point_map = np.zeros((h, w), dtype=np.uint32)
        pixel_point_map_set = np.zeros((h, w), dtype=bool)

        # We process the valid points for this camera
        for i in idx:
            d = depths[i]
            x = x_pix[i]
            y = y_pix[i]

            # Logic from Barebones snippet:
            # 1. Update point's global best depth if this camera sees it closer
            if point_depth[i] > d:
                point_depth[i] = d

                # 2. Update pixel-to-point mapping for this image
                # Criteria: pixel not set OR current mapped point is further OR close enough in depth
                has_previous = pixel_point_map_set[y, x]
                old_idx = pixel_point_map[y, x]
                
                if not has_previous or \
                   (point_depth[old_idx] > d) or \
                   (abs(point_depth[old_idx] - d) < 0.5):
                    
                    pixel_point_map[y, x] = i
                    pixel_point_map_set[y, x] = True
                    point_rgb[i] = cam.image[y, x]

                    # 3. "White out" distant points that were previously mapped to this pixel
                    # but are now found to be occluded by something much closer
                    if has_previous and \
                       (point_depth[old_idx] > d) and \
                       (abs(point_depth[old_idx] - d) > 0.5):
                        point_rgb[old_idx] = [255, 255, 255]

    return point_rgb


def save_ply(points_xyz: np.ndarray, rgb: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points_xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points_xyz, rgb):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

# NOTE: Perhaps not use this
def save_las(source_las: laspy.LasData, rgb: np.ndarray, output_path: Path) -> None:
    out = source_las
    dims = set(out.point_format.dimension_names)

    if not {"red", "green", "blue"}.issubset(dims):
        raise ValueError(
            "Input LAS point format does not support RGB dimensions. "
            "Use a .ply output or convert LAS point format to support color first."
        )

    out.red = (rgb[:, 0].astype(np.uint16) * 257)
    out.green = (rgb[:, 1].astype(np.uint16) * 257)
    out.blue = (rgb[:, 2].astype(np.uint16) * 257)
    out.write(output_path)


def main() -> None:
    args = parse_args()

    if args.frame_index is None:
        frame_index = infer_frame_index(args.las_path)
        print(f"[INFO] Inferred frame index: {frame_index}")
    else:
        frame_index = args.frame_index

    las = laspy.read(args.las_path)
    points_xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    print(f"[INFO] Loaded {points_xyz.shape[0]} points from {args.las_path}")

    points_lidar = points_xyz
    T_lidar_to_world = np.eye(4, dtype=np.float64)
    
    if args.lidar_calibration_file is not None and args.vehicle_sensor_file is not None:
        from calculation_for_transformsfile import calculate_transform_matrix

        T_lidar_to_world = calculate_transform_matrix(
            str(args.lidar_calibration_file),
            str(args.vehicle_sensor_file),
            args.lidar_sensor_type,
            args.lidar_to_world_convention,
        )
        # NOTE: I've found this bug after quite a while
        # Turned out the calculate_transform_matrix returns in the nerfstudio format
        # Thus need to convert back to aiSim
        # NOTE: if args.lidar_to_world_convention == "nerfstudio"
        if args.lidar_to_world_convention == "nerfstudio":
            T_lidar_to_world = T_permutation @ T_lidar_to_world 
        
    elif args.lidar_calibration_file is not None or args.vehicle_sensor_file is not None:
        raise ValueError(
            "Both --lidar-calibration-file and --vehicle-sensor-file must be provided together."
        )

    cameras = load_camera_views(args.transform, args.images_root, frame_index)
    print(f"[INFO] Loaded {len(cameras)} camera views from {args.transform}")

    rgb = colorize_points(
        points_lidar,
        cameras,
        T_lidar_to_world,
        args.depth_threshold,
        args.max_depth,
        debug_stats=args.debug_stats,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.output.suffix.lower()
    if suffix == ".ply":
        save_ply(points_lidar, rgb, args.output)
    elif suffix in {".las", ".laz"}:
        save_las(las, rgb, args.output)
    else:
        raise ValueError("Output extension must be one of: .ply, .las, .laz")

    # Load a ply point cloud, print it, and render it 
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(args.output)

    camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Black Background Point Cloud", width=1024, height=768)

    # Add your geometries
    vis.add_geometry(pcd)
    vis.add_geometry(camera)

    # Get rendering options and set background to black [R, G, B]
    render_option = vis.get_render_option()
    render_option.background_color = [0.0, 0.0, 0.0] 

    # Run and destroy when closed
    vis.run()
    vis.destroy_window()

    colored_count = int(np.sum(np.any(rgb != 255, axis=1)))
    print(f"[INFO] Saved colorized cloud to {args.output}")
    print(f"[INFO] Colored points: {colored_count}/{rgb.shape[0]}")


if __name__ == "__main__":
    main()
