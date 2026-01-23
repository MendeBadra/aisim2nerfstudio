import os
import shutil
import argparse
import re
from typing import List, Dict, Optional

class CameraRig:
    """
    Represents the specific 4-camera setup for the aiSim dataset.
    Handles the logic of grouping individual files into synchronized temporal frames.
    """
    def __init__(self):
        # The specific camera prefixes defined in your convention
        self.camera_names = [
            "pinhole",
            "pinhole_duplicate0",
            "pinhole_duplicate1",
            "pinhole_duplicate2"
        ]
        
        # Regex to parse: {camera_type}_{00000}.jpg
        # Captures: (camera_name), (frame_number)
        self.filename_pattern = re.compile(r"(.+)_(\d{5})\.jpg$")

    def is_valid_camera(self, name: str) -> bool:
        return name in self.camera_names

    def group_files_by_frame(self, file_list: List[str]) -> Dict[int, Dict[str, str]]:
        """
        Scans a list of filenames and groups them into atomic frames.
        Returns: { frame_id: { 'pinhole': 'filename', 'pinhole_duplicate0': 'filename', ... } }
        """
        frames = {}

        for filename in file_list:
            match = self.filename_pattern.match(filename)
            if not match:
                continue

            cam_name, frame_str = match.groups()
            frame_id = int(frame_str)

            if not self.is_valid_camera(cam_name):
                continue

            if frame_id not in frames:
                frames[frame_id] = {}
            
            frames[frame_id][cam_name] = filename

        return frames

    def filter_complete_frames(self, frames: Dict[int, Dict[str, str]]) -> Dict[int, Dict[str, str]]:
        """
        Optional: Validation step to ensure every frame has exactly 4 cameras.
        If a frame is missing a camera (dropped packet), we usually want to skip it.
        """
        complete_frames = {}
        expected_count = len(self.camera_names)

        for frame_id, cam_files in frames.items():
            if len(cam_files) == expected_count:
                complete_frames[frame_id] = cam_files
            else:
                missing = set(self.camera_names) - set(cam_files.keys())
                print(f"[Warning] Frame {frame_id} is incomplete. Missing: {missing}. Skipping.")
        
        return complete_frames


def process_dataset(args):
    rig = CameraRig()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    # 1. Scan Directory
    print(f"Scanning {args.input_dir}...")
    all_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".jpg")])
    
    # 2. Group into Frames
    # We work with 'Frames' (temporal moments), not raw images, to keep sync.
    frames = rig.group_files_by_frame(all_files)
    
    # 3. Filter for Completeness (Optional but recommended)
    valid_frames = rig.filter_complete_frames(frames)
    sorted_frame_ids = sorted(valid_frames.keys())
    
    total_frames_available = len(sorted_frame_ids)
    print(f"Found {total_frames_available} complete frames ({total_frames_available * 4} images).")

    # 4. Apply N-th Selection
    selected_frame_ids = sorted_frame_ids[::args.nth]
    
    # 5. Apply Max Count Limit
    # Logic: If user wants 9 images, we have 4 cameras. 9 // 4 = 2 full frames.
    # We discard the extra 1 image because incomplete frames break 3DGS/NeRF.
    if args.max_images is not None:
        max_frames = args.max_images // 4
        remainder = args.max_images % 4
        
        if remainder != 0:
            print(f"[Info] You requested {args.max_images} images.")
            print(f"       Aligning to camera rig size (4). processing {max_frames} frames ({max_frames*4} images).")
            print(f"       {remainder} requested images will be ignored to maintain synchronization.")
        
        selected_frame_ids = selected_frame_ids[:max_frames]

    # Summary of job
    files_to_copy = []
    for fid in selected_frame_ids:
        files_to_copy.extend(valid_frames[fid].values())

    print(f"\n--- Operation Summary ---")
    print(f"Input Directory:  {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Step (N-th):      {args.nth}")
    print(f"Frames Selected:  {len(selected_frame_ids)}")
    print(f"Images to Copy:   {len(files_to_copy)}")
    print(f"-------------------------")

    if args.dry_run:
        print("Dry run enabled. No files were copied.")
        return

    # 6. Execute Copy
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i, filename in enumerate(files_to_copy):
        src = os.path.join(args.input_dir, filename)
        dst = os.path.join(args.output_dir, filename)
        
        shutil.copy2(src, dst)
        
        if i % 100 == 0:
            print(f"Copied {i}/{len(files_to_copy)}...", end='\r')
            
    print(f"\nDone! Processed {len(files_to_copy)} images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample a multi-camera dataset by N-th frame.")
    
    parser.add_argument("--input_dir", type=str, required=True, help="Path to source images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to destination folder")
    parser.add_argument("--nth", type=int, default=1, help="Take every N-th frame (e.g. 10 for 10x speedup)")
    parser.add_argument("--max_images", type=int, default=None, help="Stop after copying this many total images")
    parser.add_argument("--dry_run", action="store_true", help="Print stats without copying files")

    args = parser.parse_args()
    
    process_dataset(args)