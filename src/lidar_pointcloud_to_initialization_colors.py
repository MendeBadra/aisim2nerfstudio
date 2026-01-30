# The input: 
#   a single lidar pointcloud
#   the configuration file for the sensor
#   camera parameters and camera images
# The output:
#   correctly colorized lidar point cloud for 3DGS training.

# The strategy. 
# To tranform the pointcloud's origin to, at first, the ego (body) space and then to each of the camera space.
# Then do some projection with the opencv.projectPoints function and obtain coloring (hopefully).

import laspy
import json
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt

from pathlib import Path

from calculation_for_transformsfile import get_sensor_position_rotation
from calculation_for_transformsfile import calculate_pom_deg

CAMERA_CALIBRATION_FILE = "C:\\aiSim\\aiMotive\\aisim_gui-5.7.0\\data\\calibrations\\mend_front_back_2side_pinhole_lidar_sensor_top.json"
SAMPLE_FOLDER = "frame_with_depth" 
CAMERA_TYPE = "pinhole" # forward
# CAMERA_TYPE = "pinhole_duplicate0"

# CAMERA_TYPE = "pinhole_duplicate1"
# CAMERA_TYPE = "pinhole_duplicate2"
# CAMERA_TYPE = "lidar_sensor"

LIDAR = "lidar_sensor"



def get_sensor_pom(camera_calibration_file: Path, camera_type: str = CAMERA_TYPE):
    sensor_pos, sensor_rot = get_sensor_position_rotation(camera_calibration_file, camera_type)
    yaw, pitch, roll = sensor_rot['yaw'], sensor_rot['pitch'], sensor_rot['roll']
    T_sensor_pom = calculate_pom_deg(sensor_pos, yaw, pitch, roll)

    return T_sensor_pom


def get_ego_points(points: np.ndarray, sensor_pom:np.ndarray) -> np.ndarray:
    return sensor_pom @ points

def main():
    # test main
    sample_folder = Path(SAMPLE_FOLDER)
    # print(Path('.').absolute())
    camera_calibration_file = Path(CAMERA_CALIBRATION_FILE)
    las_file: Path = sample_folder / "lidar_sensor_00002.las"
    las = laspy.read(las_file)

    points = np.vstack((las.x, las.y, las.z)).T

    T_sensor_pom = get_sensor_pom(camera_calibration_file, LIDAR)
    ego_points = get_ego_points(points, T_sensor_pom)
    print(ego_points)

# def main():
#     sample_folder = Path(CAMERA_TYPE)

#     las_file: Path = sample_folder / "lidar_sensor_00002.las"
#     camera_calibration_file = Path(CAMERA_CALIBRATION_FILE)

#     las = laspy.read(las_file)

#     points = np.vstack((las.x, las.y, las.z)).T

    
if __name__ == "__main__":
    main()