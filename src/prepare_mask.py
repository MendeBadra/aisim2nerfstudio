"""A script to produce masks for the captured frames in nerfstudio's format and to quote from documentation of nerfstudio:
"
The following mask requirements must be met:
- Must be 1 channel with only black and white pixels
- Must be the same resolution as the training image
- Black corresponds to regions to ignore
- If used, all images must have a mask
"

This script has 2 options "generate_white" or "copy_existing"
"generate_white" means that we don't ignore anything
"copy_existing" is used when I have already have masks that I used photopea to create. 
You could obtain the mask this way and copy over all the relevant image transforms.json["frames"] field to mask not relevant details.
I, for example, used this feature of nerfstudio to mask out front and back of a vehicle from the camera.   
"""


import os
import shutil
from typing import Optional
from PIL import Image
import numpy as np

# --- Configuration ---
CONFIG = {
    "num_frames": 1840,
    "output_dir": "aisim_ns_dataset/masks",
    
    # Mode: 'generate_white' OR 'copy_existing'
    "mode": "generate_white", 

    # Parameters for 'generate_white' mode
    "resolution_reference_file": "pinhole_00000_edited.tga", 
    
    # Parameters for 'copy_existing' mode
    "source_file_to_copy": "mask_pinhole_duplicate0.jpg",

    # Naming Scheme
    "camera_type": "pinhole_duplicate2",
    "filename_prefix": "mask", 
    "filename_extension": ".jpg",
    
    # Padding: Ensures '00000' (5 digits) instead of '0000' (4 digits)
    "min_padding": 5 
}

def get_resolution_from_image(filepath: str) -> tuple[int, int]:
    """Reads dimensions (width, height) from an image file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Reference file not found: {filepath}")
    
    with Image.open(filepath) as img:
        return img.size

def create_white_mask(width: int, height: int, save_path: str) -> None:
    """Creates a single-channel (L) all-white image and saves it."""
    white_array = np.full((height, width), 255, dtype=np.uint8)
    image = Image.fromarray(white_array, mode='L')
    image.save(save_path)

def generate_sequence(source_file: str, destination_dir: str, count: int, prefix: str, ext: str, min_pad: int) -> None:
    """
    Copies a source file 'count' times into destination_dir using sequential naming.
    """
    os.makedirs(destination_dir, exist_ok=True)
    
    # Calculate padding: Use the larger of min_padding OR the digits needed for count
    # Example: If count is 1840, len is 4. max(5, 4) = 5. Result: 00000
    pad_width = max(min_pad, len(str(count)))
    
    # Construct base name: "mask_pinhole_duplicate1_"
    base_name = f"{prefix}_{CONFIG['camera_type']}_"

    print(f"Generating {count} frames in '{destination_dir}'...")
    print(f"Format: {base_name}{'0'*pad_width}{ext}")

    for i in range(count):
        # zfill(5) turns 0 -> '00000', 10 -> '00010'
        filename = f"{base_name}{str(i).zfill(pad_width)}{ext}"
        dest_path = os.path.join(destination_dir, filename)
        
        try:
            shutil.copy2(source_file, dest_path)
        except OSError as e:
            print(f"Failed to copy frame {i}: {e}")
            break
            
        if i > 0 and i % (count // 10) == 0:
            print(f" -> Progress: {i}/{count}")

    print(f"Completed. {count} files created.")

def main():
    mode = CONFIG["mode"]
    temp_source = "temp_source_mask" + CONFIG["filename_extension"]
    
    try:
        if mode == "generate_white":
            # 1. Determine resolution
            ref_file = CONFIG["resolution_reference_file"]
            w, h = get_resolution_from_image(ref_file)
            print(f"Detected resolution: {w}x{h}")

            # 2. Create a temporary master mask
            create_white_mask(w, h, temp_source)
            source_file = temp_source

        elif mode == "copy_existing":
            source_file = CONFIG["source_file_to_copy"]
            if not os.path.exists(source_file):
                raise FileNotFoundError(f"Source file to copy not found: {source_file}")

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 3. Replicate the file
        generate_sequence(
            source_file=source_file, 
            destination_dir=CONFIG["output_dir"], 
            count=CONFIG["num_frames"], 
            prefix=CONFIG["filename_prefix"],
            ext=CONFIG["filename_extension"],
            min_pad=CONFIG["min_padding"]
        )

    except Exception as e:
        print(f"Error: {e}")

    finally:
        if mode == "generate_white" and os.path.exists(temp_source):
            os.remove(temp_source)

if __name__ == "__main__":
    main()