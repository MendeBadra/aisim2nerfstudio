"""This script converts .tga files to .jpg"""


import argparse
import os
from PIL import Image

def convert_tga_to_jpg(source_path, output_path, quality=95):
    """
    Converts TGA files from the source directory to JPG files in the output directory.

    Args:
        source_path (str): The directory containing the .tga files.
        output_path (str): The directory to save the .jpg files.
        quality (int): The JPEG compression quality (0-100).
    """
    if not os.path.isdir(source_path):
        print(f"Error: Source directory not found: {source_path}")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Starting conversion from '{source_path}' to '{output_path}'...")
    
    converted_count = 0
    
    # Walk through all files in the source directory
    for item in os.listdir(source_path):
        if item.lower().endswith('.tga'):
            tga_filename = item
            tga_filepath = os.path.join(source_path, tga_filename)
            
            # Create the new JPG filename by replacing the extension
            jpg_filename = os.path.splitext(tga_filename)[0] + '.jpg'
            jpg_filepath = os.path.join(output_path, jpg_filename)

            try:
                # Open the TGA image
                img = Image.open(tga_filepath)

                if img.mode == "RGBA":
                    img = img.convert("RGB")
                
                # Save the image as JPG
                # The 'quality' parameter controls compression, 95 is high quality.
                # The 'optimize=True' parameter helps reduce file size slightly.
                img.save(jpg_filepath, 'JPEG', quality=quality, optimize=True)
                
                print(f"  Converted: {tga_filename} -> {jpg_filename}")
                converted_count += 1
            except FileNotFoundError:
                print(f"  Skipping: File not found {tga_filepath}")
            except Exception as e:
                print(f"  Error converting {tga_filename}: {e}")

    print(f"\nConversion complete. Total files converted: {converted_count}")

def main():
    """
    Sets up the argument parser and calls the conversion function.
    """
    parser = argparse.ArgumentParser(
        description="Convert all TGA images in a source directory to JPG images in an output directory."
    )
    
    # Define the required positional arguments
    parser.add_argument(
        'source',
        type=str,
        help="The path to the directory containing .tga files."
    )
    parser.add_argument(
        'output',
        type=str,
        help="The path to the directory where .jpg files will be saved."
    )
    
    # Optional argument for JPEG quality
    parser.add_argument(
        '-q', '--quality',
        type=int,
        default=95,
        choices=range(1, 101),
        metavar='[1-100]',
        help="JPEG quality setting (1-100). Default is 95."
    )

    args = parser.parse_args()
    
    # Call the conversion function with the parsed arguments
    convert_tga_to_jpg(args.source, args.output, args.quality)

if __name__ == "__main__":
    main()