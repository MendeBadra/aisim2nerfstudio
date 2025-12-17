"""This file is to join 4 camera_type's transforms.json output into a single json file"""

import json
from pathlib import Path

INPUT_DIR = "outputs"
OUTPUT_DIR = "outputs"
TEST_NUM = 1

# List your input files here
input_file_names = [
   f"transforms_pinhole_test{TEST_NUM}.json",
    f"transforms_pinhole_duplicate0_test{TEST_NUM}.json",
    f"transforms_pinhole_duplicate1_test{TEST_NUM}.json",
    f"transforms_pinhole_duplicate2_test{TEST_NUM}.json"
]

input_files = [Path(INPUT_DIR) / input_file_name for input_file_name in input_file_names]
output_file = Path(OUTPUT_DIR) / "combined_transforms.json"

print(input_files)
# This will hold our final data
master_data = {}
combined_frames = []

print(f"Merging {len(input_files)} files...")

for index, file_path in enumerate(input_files):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # If it's the first file, save the metadata (camera angle, intrinsics, etc.)
        if index == 0:
            master_data = data
            combined_frames = data['frames']
        else:
            # For subsequent files, just extend the list of frames
            if "frames" in data:
                combined_frames.extend(data['frames'])
                
        print(f"Loaded {len(data.get('frames', []))} frames from {file_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file {file_path}")

# Assign the combined list back to the master dictionary
master_data['frames'] = combined_frames

# OPTIONAL: Re-index IDs to ensure they are unique
# (Uncomment the lines below if you have duplicate IDs across files)
for i, frame in enumerate(master_data['frames']):
    frame['colmap_im_id'] = i

# Save to new file
with open(output_file, 'w') as f:
    json.dump(master_data, f, indent=4)

print(f"Success! Saved {len(master_data['frames'])} total frames to {output_file}")