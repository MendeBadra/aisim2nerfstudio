# aisim2nerfstudio

Convert **aiSim autonomous driving data** into **Nerfstudio** format for NeRF-based 3D reconstruction. The example output of camera and LiDAR poses in `frame_with_depth/transforms.json`.
This is a living and active repository for taking in aiSim sensor JSON data and outputs a `transforms.json` file that's compatible for `nerfstudio` program. 

## Setup (Windows)

1. **Create conda environment**
   Make sure you have **conda** installed. Then, from your project root where `environment.yml` is located, run:

   ```bash
   conda env create -f environment.yml
   ```

2. **Activate environment and setup paths**

   ```bat
   conda activate nerfstudio1
   cd Documents\MendeFolder\quick_poly_crop

   :: Setup Visual Studio environment for compilation
   call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

   :: Set CUDA path
   set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
   ```

3. **Dataset folder**
   Ensure the `frame_with_depth` folder (from `aisim_ns_dataset_lidar`) is placed in the project root. It should contain:

   ```
   transforms.json
   RGB frames
   LiDAR frames
   ```

4. **Run Nerfstudio training**
   Example command for training with your MCMC strategy:

   ```bash
   ns-train splatfacto-big \
     --pipeline.model.strategy mcmc \
     --data aisim_ns_dataset \
     --load-dir outputs\aisim_ns_dataset\splatfacto\2025-12-18_194002\nerfstudio_models
   ```

---

### Example Input/Output (TODO)

* **Input:** `frame_with_depth/transforms.json` + RGB/LiDAR frames
* **Output:** Trained NeRF/SplatFacto model in Nerfstudio `outputs/` folder, ready for visualization

[pic]()


## Coordinate Transformation

aiSim → Nerfstudio axes:

* **aiSim World:** X-East, Y-North, Z-Up
* **aiSim Body:** X-Forward, Y-Left, Z-Up
* **Nerfstudio:** +X Right, +Y Up, -Z Forward

Applied transformation matrix:
```
T_trans = [
  0   0  -1   0
 -1   0   0   0
  0   1   0   0
  0   0   0   1
]
```
---

## Status

* Camera poses reasonably aligned in Nerfstudio
* Output quality needs improvement → further research ongoing
* LiDAR included but not fully utilized yet

---

## Folder Structure

```
aisim2nerfstudio/
├─ frame_with_depth/     # RGB + LiDAR + transforms.json
├─ scripts/             # Conversion scripts
└─ README.md
```

---

## Next Steps

* LiDAR integration
