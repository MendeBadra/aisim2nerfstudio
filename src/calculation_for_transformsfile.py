"""
This script writes the transforms.json
"""


import json
from pathlib import Path
import numpy as np
import math
from tqdm import tqdm

# aiSim uses column-major matrix, where the translation vector is the 4th column (index 3).
# We will use a standard numpy (row-major) convention for readability, 
# and ensure the final translation is in the 4th row (index 3).
CAMERA_TYPE = "pinhole" # forward
# CAMERA_TYPE = "pinhole_duplicate0"

# CAMERA_TYPE = "pinhole_duplicate1"
# CAMERA_TYPE = "pinhole_duplicate2"
GPS_VEHICLE_SENSOR_DATASET = "2025-12-04_18-22-25"

CAMERA_CALIBRATION_FILE = "C:/aiSim/aiMotive/aisim_gui-5.7.0/data/calibrations/mend_front_back_2side_pinhole.json"
# OUTPUT_DIR = "outputs/test0" # 2025.12.17 The fix needed
OUTPUT_DIR = "outputs"
TEST_NUM = 12
# CAMERA_TYPE = "pinhole_duplicate0"

def get_intrinsic_params(camera_calibration_file, camera_type = CAMERA_TYPE):
    """
    Reads nested camera parameters from the "camera calibration" profiles that is found under: C:\aiSim\aiMotive\aisim_gui-5.7.0\data\calibrations folder

    Args:
        input_data (dict): The dictionary loaded from the source JSON file.

    Returns:
        dict: The restructured camera configuration.
    """
    with open(camera_calibration_file, 'r') as f:
        camera_calibration = json.load(f)
    # 1. Navigate to the core camera configuration block
    try:
        camera_config = camera_calibration['sensors'][camera_type]['camera_config']
        dist_params = camera_config['distortion_parameters']
        dist_coeffs = dist_params['distortion_coefficients']
    except KeyError as e:
        print(f"Error: Missing expected key in the JSON structure: {e}")
        return None

    # 2. Extract and map the values
    
    k1 = dist_coeffs[0] if len(dist_coeffs) > 0 else 0.0
    k2 = dist_coeffs[1] if len(dist_coeffs) > 1 else 0.0
    p1 = dist_coeffs[2] if len(dist_coeffs) > 2 else 0.0
    p2 = dist_coeffs[3] if len(dist_coeffs) > 3 else 0.0
    k3 = dist_coeffs[4] if len(dist_coeffs) > 4 else 0.0 # k3 apparently not used in nerfstudio
    # k4 = 0.0 # Not present in the 5-element input array

    output = {
        # General parameters
        "camera_model": "OPENCV" if camera_config['model'] == "OpenCVPinhole" else "Undefined", # NERFSTUDIO: camera model type [OPENCV, OPENCV_FISHEYE]
        "w": camera_config['width'],
        "h": camera_config['height'],
        
        # Intrinsic parameters (focal length and principal point)
        "fl_x": dist_params['focal_length'][0],
        "fl_y": dist_params['focal_length'][1],
        "cx": dist_params['principal_point'][0],
        "cy": dist_params['principal_point'][1],
        
        # Distortion coefficients
        "k1": k1,
        "k2": k2,
        "k3": k3,
        # "k4": k4,
        "p1": p1,
        "p2": p2,
    }
    
    return output

def get_sensor_position_rotation(camera_calibration_file):
     with open(camera_calibration_file, 'r') as f:
        input_data = json.load(f)
    # 1. Navigate to the core camera configuration block
        try:
            camera_config = input_data['sensors'][CAMERA_TYPE]['camera_config']
            relative_sensor_position = camera_config['position'] # this is body space position. Is not the absolute position
            relative_sensor_rotation = camera_config['rotation']
            
        except KeyError as e:
            print(f"Error: Missing expected key in the JSON structure: {e}")
            return None
        
        return relative_sensor_position, relative_sensor_rotation

def get_rt_transform(file_path):
    with open(file_path, 'r') as f:
        vehicle_sensor_data = json.load(f)

    return vehicle_sensor_data["ego_motion"]["rt_transform"]     

def euler_zyx_to_matrix(yaw_rad, pitch_rad, roll_rad):
    """
    Converts Euler ZYX (Yaw, Pitch, Roll) angles to a 4x4 rotation matrix.
    Note: See the aiSim documentation Coordinate transforms page. This is Python version of the Cpp code in the documentation
    """
    sin_z, cos_z = math.sin(yaw_rad), math.cos(yaw_rad)
    sin_y, cos_y = math.sin(pitch_rad), math.cos(pitch_rad)
    sin_x, cos_x = math.sin(roll_rad), math.cos(roll_rad)

    # Rz (Rotation around Z-axis)
    # GLM matrix definition is column-major, we transpose to row-major for numpy storage.
    # The GLM code shows the *columns* of the matrix. We use row-major for numpy:
    # Rz = [[cos_z, sin_z, 0], [-sin_z, cos_z, 0], [0, 0, 1]] (Transposed from GLM column vectors)
    Rz = np.array([
        [cos_z, -sin_z, 0.0],
        [sin_z, cos_z, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Ry (Rotation around Y-axis)
    # Ry = [[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]] (Transposed from GLM column vectors)
    Ry = np.array([
        [cos_y, 0.0, sin_y],
        [0.0, 1.0, 0.0],
        [-sin_y, 0.0, cos_y]
    ])

    # Rx (Rotation around X-axis)
    # Rx = [[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]] (Transposed from GLM column vectors)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_x, -sin_x],
        [0.0, sin_x, cos_x]
    ])

    # The final rotation matrix R is calculated as Rz * Ry * Rx
    # R = Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx
    
    # Create the 4x4 homogeneous matrix with a 0 translation vector initially
    M = np.identity(4)
    M[:3, :3] = R
    
    return M

def calculate_pom_deg(position, yaw_deg, pitch_deg, roll_deg):
    """
    Calculates the Position and Orientation Matrix (POM) in degrees.
    :param position: [x, y, z] vector of the sensor's position (e.g., [1.8, 0, 1.5]).
    :param yaw_deg: Yaw angle in degrees.
    :param pitch_deg: Pitch angle in degrees.
    :param roll_deg: Roll angle in degrees.
    :return: 4x4 homogeneous transformation matrix (row-major).
    """
    yaw_rad, pitch_rad, roll_rad = (
        math.radians(yaw_deg), 
        math.radians(pitch_deg), 
        math.radians(roll_deg)
    )
    
    # Get the 4x4 rotation matrix
    pom = euler_zyx_to_matrix(yaw_rad, pitch_rad, roll_rad)
    
    pom[:3, 3] = position # last column
    
    return pom

def reshape_rt_transform(rt_transform_array):
    """
    Loads the Ego-POM from the 16-element rt_transform array.
    The documentation implies the array is already structured as the Ego POM.
    """
    ego_pom = np.array(rt_transform_array).reshape((4, 4), order='F')
    return ego_pom


# def create_nerfstudio_conversion_matrix():
#     """
#     Creates the 4x4 matrix to convert from a typical (X-Right, Y-Down, Z-Forward)
#     camera convention to the Nerfstudio/OpenGL (X-Right, Y-Up, Z-Back) convention.
#     This is a 180-degree rotation around the X-axis.
#     """
#     T_conversion = np.identity(4)
#     # Flip Y and Z axes
#     T_conversion[1, 1] = -1.0
#     T_conversion[2, 2] = -1.0
#     return T_conversion

# def create_nerfstudio_conversion_matrix():
#     """
#     Creates the 4x4 matrix to convert from the aiSim Robotics convention 
#     (X-Forward, Y-Left, Z-Up) to the Nerfstudio/OpenGL convention 
#     (X-Right, Y-Up, Z-Back).
#     """
#     # We construct the matrix columns based on where we want the source axes to go.
#     # Source X (Forward) -> Target -Z (Back is +Z, so Forward looks down -Z)
#     # Source Y (Left)    -> Target -X (Right is +X, so Left is -X)
#     # Source Z (Up)      -> Target +Y (Up is +Y)
    
#     T_conversion = np.zeros((4, 4))
    
#     # Column 0: Source X (Forward) lands in Target Z (with -1 sign)
#     T_conversion[2, 0] = -1.0 
    
#     # Column 1: Source Y (Left) lands in Target X (with -1 sign)
#     T_conversion[0, 1] = -1.0 
    
#     # Column 2: Source Z (Up) lands in Target Y (with +1 sign)
#     T_conversion[1, 2] = 1.0 
    
#     # Homogeneous component
#     T_conversion[3, 3] = 1.0
    
#     return T_conversion

# def create_nerfstudio_conversion_matrix(): # test 4
#     """
#     Revised Conversion Matrix.
#     Previous Issue: Camera was looking "Outside" (Right/Radial) instead of Forward.
#     Fix: Rotate 90 degrees to align Source Y (Left/Tangent) with Target Look.
    
#     Mapping:
#     - Source Y (was Left)  -> Target -Z (New Look Direction)
#     - Source X (was Fwd)   -> Target +X (New Right Direction)
#     - Source Z (Up)        -> Target +Y (Up)
#     """
#     T_conversion = np.zeros((4, 4))
    
#     # 1. Map Source X to Nerfstudio Right (+X)
#     # This assumes the "Old Forward" is actually pointing Right (Outside)
#     T_conversion[0, 0] = 1.0 
    
#     # 2. Map Source Y to Nerfstudio Back (+Z)
#     # Nerfstudio looks down -Z. So if we map Y to -Z (value -1), 
#     # the Camera will look along +Y. 
#     # Wait, we want to look along the tangent.
#     # If the previous code looked "Outside" (North), and we want "Tangent" (West),
#     # we need to rotate the mapping.
    
#     # Let's try mapping Source Y (Left) to Look (-Z).
#     # If Source Y is West, Camera will look West.
#     T_conversion[2, 1] = -1.0 
    
#     # 3. Map Source Z to Nerfstudio Up (+Y)
#     T_conversion[1, 2] = 1.0 
    
#     # 4. Homogeneous
#     T_conversion[3, 3] = 1.0
    
#     return T_conversion

# def create_nerfstudio_conversion_matrix(): # test 5
#     """
#     Corrects the alignment from aiSim Sensor Space to Nerfstudio Camera Space.
    
#     Source (aiSim Sensor):
#     - X: Forward
#     - Y: Left
#     - Z: Up
    
#     Target (Nerfstudio Camera):
#     - Z: Back (so -Z is Look/Forward)
#     - X: Right
#     - Y: Up
#     """
#     T = np.zeros((4, 4))
    
#     # 1. Forward Mapping:
#     # We want Sensor Forward (+X) to be Camera Look (-Z).
#     # So Source X maps to Target -Z.
#     T[2, 0] = -1.0 
    
#     # 2. Left Mapping:
#     # We want Sensor Left (+Y) to be Camera Left (-X).
#     # Since Target X is Right, Source Y maps to Target -X.
#     T[0, 1] = -1.0
    
#     # 3. Up Mapping:
#     # We want Sensor Up (+Z) to be Camera Up (+Y).
#     T[1, 2] = 1.0
    
#     # 4. Homogeneous
#     T[3, 3] = 1.0
    
#     return T

def nerfstudio_conversion(T_matrix):
    T_converted = np.zeros((4, 4))
    T_rotation = T_matrix[:3, :3]
    T_permutation = np.array([
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0]
        ])
    
    T_rotation_converted = T_rotation @ T_permutation
    position_vec = T_matrix[:3, 3]
    position_vec_converted = position_vec# @ T_permutation
    T_converted[:3, :3] = T_rotation_converted
    T_converted[:3, 3] = position_vec_converted
    T_converted[3, 3] = 1.0
    # print(T_converted)
    return T_converted

def calculate_ns_transform_matrix(camera_calibration_file, vehicle_sensor_file):
    """transform matrix like in the nerfstudio transforms.json
    NOTE: This is the main function that uses all of the functions defined."""
    # camera_calibration_file = "C:/aiSim/aiMotive/aisim_gui-5.7.0/data/calibrations/mend_front_back_2side_pinhole.json"
    sensor_pos, sensor_rot = get_sensor_position_rotation(camera_calibration_file)
    # print(pos, rot)

    # are these in degrees
    yaw, pitch, roll = sensor_rot['yaw'], sensor_rot['pitch'], sensor_rot['roll']
    
    # 
    rt_transform_arr = get_rt_transform(vehicle_sensor_file)
    T_sensor_pom = calculate_pom_deg(sensor_pos, yaw, pitch, roll)
    T_ego_pom = reshape_rt_transform(rt_transform_arr)
    # print(f"T_ego_pom = \n {T_ego_pom}")
    T_w2c = T_ego_pom @ T_sensor_pom
    T_converted = nerfstudio_conversion(T_w2c)

    return T_converted # T for the ns representation

# TEST - for a single example
# def main():
#     print(TEST_NUM)
#     camera_calibration_file = "C:/aiSim/aiMotive/aisim_gui-5.7.0/data/calibrations/mend_front_back_2side_pinhole.json" # the calibration file
#     vehicle_sensor_file = "./data/2025-12-04_18-22-25/ego/vehicle_sensor/vehicle_sensor_00000.json" # single example file
#     print(calculate_ns_transform_matrix(camera_calibration_file, vehicle_sensor_file))

#The Main loop
def main():
    # DEV - for a big folder
    output_dir = Path(OUTPUT_DIR)
    output_file_path = output_dir / f"transforms_{CAMERA_TYPE}_test{TEST_NUM}.json"
    print(f"[INFO] Output path is {str(output_file_path)}")
    
    intrinsic_params = get_intrinsic_params(CAMERA_CALIBRATION_FILE, CAMERA_TYPE)

    vehicle_sensor_files_path = Path("./data/2025-12-04_18-22-25/ego/vehicle_sensor") # for the whole car
    vehicle_sensor_files = list(vehicle_sensor_files_path.glob("vehicle_sensor*.json"))
    # vehicle_sensor_files = vehicle_sensor_files[:1]
    print(f"[INFO] We've got {len(vehicle_sensor_files)} files found for vehicle_sensor*.json")
    frames = []

    for vehicle_sensor_file in tqdm(vehicle_sensor_files):
        stem = vehicle_sensor_file.stem # filename without extension
        prefix = "vehicle_sensor_"
        if stem.startswith(prefix):
            id_str = stem.replace(prefix, "")
        else:
            id_str = "N/A"

        T_matrix = calculate_ns_transform_matrix(CAMERA_CALIBRATION_FILE, vehicle_sensor_file)
        frame = {}
        frame['file_path'] = f"images/{CAMERA_TYPE}_{id_str}.jpg" # supposing that we've converted all .tga images to .jpg
        frame['mask_path'] = f"masks/mask_{CAMERA_TYPE}_{id_str}.jpg" # mask path
        frame["transform_matrix"] = T_matrix.tolist()
        frame["colmap_im_id"] = int(id_str)
        frames.append(frame)

    transforms = intrinsic_params
    transforms["frames"] = frames
    transforms["applied_transform"] = [
        [
            1.0,
            0.0,
            0.0,
            0.0
        ],
        [
            0.0,
            0.0,
            1.0,
            0.0
        ],
        [
            -0.0,
            -1.0,
            -0.0,
            -0.0
        ]
    ]
    
    with open(output_file_path, 'w') as f:
        json.dump(transforms, f, indent=4)
    print("Done!")


if __name__ == "__main__":
    main()
