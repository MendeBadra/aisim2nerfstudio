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
CAMERA_TYPE = "pinhole"
# CAMERA_TYPE = "pinhole_duplicate0" 

# CAMERA_TYPE = "pinhole_duplicate1"
# CAMERA_TYPE = "pinhole_duplicate2"
GPS_VEHICLE_SENSOR_DATASET = "2025-12-04_18-22-25"

CAMERA_CALIBRATION_FILE = "C:/aiSim/aiMotive/aisim_gui-5.7.0/data/calibrations/mend_front_back_2side_pinhole.json"
# OUTPUT_DIR = "outputs/test0" # 2025.12.17 The fix needed
OUTPUT_DIR = "outputs"
TEST_NUM = 1
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
        # "k3": k3,
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
    
    # The GLM code sets pom[3] (the 4th column in column-major, or 4th row in row-major) 
    # to the position vector [x, y, z, 1.0].
    # In row-major numpy, this is the last row (index 3).
    pom[3, :3] = position
    
    return pom

def reshape_rt_transform(rt_transform_array):
    """
    Loads the Ego-POM from the 16-element rt_transform array.
    The documentation implies the array is already structured as the Ego POM.
    """
    # Reshape the 16-element array into a 4x4 matrix. 
    # Based on the documentation's structure (Forward, Left, Up, Position), 
    # the data is likely stored in a row-major format, or the code handles a 
    # column-major array by transposing during reshaping.
    # We will assume row-major based on the observed position vector in the 4th block.
    
    # The matrix provided in your prompt:
    # [
    #  R00, R01, R02, 0.0,
    #  R10, R11, R12, 0.0,
    #  R20, R21, R22, 0.0,
    #  Tx, Ty, Tz, 1.0
    # ]
    ego_pom = np.array(rt_transform_array).reshape((4, 4), order='F')
    return ego_pom

# ... (Your previous calculate_pom_deg and load_ego_pom_from_rt_transform functions)

def create_nerfstudio_conversion_matrix():
    """
    Creates the 4x4 matrix to convert from a typical (X-Right, Y-Down, Z-Forward)
    camera convention to the Nerfstudio/OpenGL (X-Right, Y-Up, Z-Back) convention.
    This is a 180-degree rotation around the X-axis.
    """
    T_conversion = np.identity(4)
    # Flip Y and Z axes
    T_conversion[1, 1] = -1.0
    T_conversion[2, 2] = -1.0
    return T_conversion

def calculate_ns_transform_matrix(camera_calibration_file, vehicle_sensor_file):
    """transform matrix like in the nerfstudio transforms.json
    NOTE: This is the main function that uses all of the functions defined."""
    # camera_calibration_file = "C:/aiSim/aiMotive/aisim_gui-5.7.0/data/calibrations/mend_front_back_2side_pinhole.json"
    sensor_pos, sensor_rot = get_sensor_position_rotation(camera_calibration_file)
    # print(pos, rot)
    yaw, pitch, roll = sensor_rot['yaw'], sensor_rot['pitch'], sensor_rot['roll']
    
    # 
    rt_transform_arr = get_rt_transform(vehicle_sensor_file)

    T_sensor_pom = calculate_pom_deg(sensor_pos, yaw, pitch, roll)
    T_ego_pom = reshape_rt_transform(rt_transform_arr)
    # print(f"T_ego_pom = \n {T_ego_pom}")
    T_w2c = T_ego_pom @ T_sensor_pom
    T_conversion = create_nerfstudio_conversion_matrix()
    
    # Final matrix: T_w2c @ T_conversion
    T_c2w_ns = T_w2c @ T_conversion
    
    return T_c2w_ns  # T for the ns representation

    
# def main():
    # TEST - for a single example
    # camera_calibration_file = "C:/aiSim/aiMotive/aisim_gui-5.7.0/data/calibrations/mend_front_back_2side_pinhole.json" # the calibration file
    # vehicle_sensor_file = "C:/Users/Labor/Documents/MendeFolder/quick_poly_crop/2025-12-04_18-22-25/ego/vehicle_sensor/vehicle_sensor_00000.json" # single example file
    # print(calculate_ns_transform_matrix(camera_calibration_file, vehicle_sensor_file))

def main():
    # DEV - for a big folder
    output_dir = Path(OUTPUT_DIR)
    output_file_path = output_dir / f"transforms_{CAMERA_TYPE}_test{TEST_NUM}.json"
    print(f"[INFO] Output path is {str(output_file_path)}")
    
    intrinsic_params = get_intrinsic_params(CAMERA_CALIBRATION_FILE)

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
