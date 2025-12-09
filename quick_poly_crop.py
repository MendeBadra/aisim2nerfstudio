from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image
import numpy as np

# We try to cover the image with 
# ----------------------------
# 1️⃣ Load the edited image
# ----------------------------

# GPS_ONLY_DATASET = "2025-12-04_16-02-46"
GPS_VEHICLE_SENSOR_DATASET = "2025-12-04_18-22-25"
EDITED = "_editedjpg"
# Choose one!
PINHOLE_TYPE = "pinhole"
# PINHOLE_TYPE = "pinhole_duplicate0"

# edited_path = Path("pinhole_duplicate0_00000_edited.tga") # Choose this for different
edited_path = Path(f"{PINHOLE_TYPE}_00000_edited.tga")
edited_img = Image.open(edited_path).convert("RGBA")
edited_array = np.array(edited_img)

mask = edited_array[:, :, 3] != 0 

plt.imshow(mask)
plt.show()

# input_folder = Path("ego/pinhole/color")

# input_folder = Path(f"{2025-12-04_16-02-46}/ego/pinhole_duplicate0/color") # 
# NOTE: Different folder

input_folder = Path(f"{GPS_VEHICLE_SENSOR_DATASET}/ego/{PINHOLE_TYPE}/color")
output_folder = Path(f"./{GPS_VEHICLE_SENSOR_DATASET + EDITED}/{PINHOLE_TYPE}") 
output_folder.mkdir(exist_ok=True)

# we essentially retain the same mask for the edited image for each image in the folder
for img_path in tqdm(input_folder.glob("*.tga")):
    img = Image.open(img_path).convert("RGBA")
    img_array = np.array(img)

    if img_array.shape != edited_array.shape:
        raise ValueError(f"Image {img_path} has different dimensions.")

    result_array = np.zeros_like(img_array)
    result_array[mask] = img_array[mask]

    result_img = Image.fromarray(result_array)
    result_img.save(output_folder / (img_path.stem + ".jpg"))

print("Batch processing complete!")
print(f"Output saved to: {output_folder}")
