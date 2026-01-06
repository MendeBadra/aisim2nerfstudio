"""This file is to join 4 camera_type's transforms.json output into a single json file"""

import json
from pathlib import Path

INPUT_DIR = "outputs"
OUTPUT_DIR = "outputs"
TEST_NUM = 12

# --- CONFIGURATION ---
# Set K here. 1 = every frame, 2 = every 2nd frame, 10 = every 10th frame, etc.
FRAME_STEP = 20
# ---------------------

# List your input files here
input_file_names = [
    f"transforms_pinhole_test{TEST_NUM}.json",
    f"transforms_pinhole_duplicate0_test{TEST_NUM}.json",
    f"transforms_pinhole_duplicate1_test{TEST_NUM}.json",
    f"transforms_pinhole_duplicate2_test{TEST_NUM}.json"
]

input_files = [Path(INPUT_DIR) / input_file_name for input_file_name in input_file_names]

# Update output filename to include the step count (e.g., ..._k5.json)
output_file = Path(OUTPUT_DIR) / f"combined_transforms_test{TEST_NUM}_k{FRAME_STEP}.json"

print(input_files)

master_data = {}
combined_frames = []

print(f"Merging {len(input_files)} files with step size {FRAME_STEP}...")

for index, file_path in enumerate(input_files):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            # Get the raw frames from the file
            raw_frames = data.get('frames', [])
            
            # SLICING MAGIC: This grabs every k-th frame
            # Syntax: list[start:stop:step]
            sampled_frames = raw_frames[::FRAME_STEP]

        # If it's the first file, save the metadata
        if index == 0:
            master_data = data
            combined_frames = sampled_frames
        else:
            # For subsequent files, extend with the sampled list
            combined_frames.extend(sampled_frames)
                
        print(f"Loaded {len(raw_frames)} frames from {file_path} -> kept {len(sampled_frames)}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file {file_path}")

# Assign the combined list back to the master dictionary
master_data['frames'] = combined_frames

# IMPORTANT: Re-index IDs to ensure they are unique and sequential
# Since we skipped frames, the original IDs might have gaps.
for i, frame in enumerate(master_data['frames']):
    frame['colmap_im_id'] = i

# Save to new file
with open(output_file, 'w') as f:
    json.dump(master_data, f, indent=4)

print(f"Success! Saved {len(master_data['frames'])} total frames to {output_file}")