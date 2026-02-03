Adversarial Illumination Parameter Sweeps

This repository contains two Blender Python scripts used to search for camera-level adversarial illumination parameters that degrade stop sign detection by a vision model.

The scripts generate physically grounded synthetic data, apply a flashing light attack, and evaluate each configuration using YOLOv11. Each run produces rendered frames, integrated exposure images, and a scalar objective score that measures detection confidence.

Both scripts assume a Blender scene that contains:

*   A stop sign
*   A vehicle or camera
*   A controllable flashlight or emissive light source
*   A sun light for ambient illumination
*   The modifier and material names referenced in the scripts

All results are written to disk. No command line arguments are required.

Script 1: full_sweep_temporal.py

This script performs a two-stage sweep:

1.  **Coarse sweep** over all spatial, illumination, and temporal parameters
2.  **Temporal micro sweep** using only the best-performing coarse samples

Spatial parameters are frozen during the micro stage. Only timing and flashing behavior are refined.

Search strategy

*   **Coarse:** Halton sampling across all parameters
*   **Micro:** Percentile-based narrowing using top-K coarse samples
*   **Optimization target:** Minimize YOLO stop sign detection score

Parameters explored

**Spatial:**

*   Flashlight X position
*   Flashlight Z height
*   Car Y distance
*   Flashlight yaw angle

**Illumination:**

*   Sun irradiance
*   Flashlight emission strength

**Temporal:**

*   Camera FPS
*   Shutter time
*   Flash frequency
*   Duty cycle
*   Phase offset

Output

A new experiment folder is created for each run. Inside you will find:

*   `sweep_index__coarse.csv`
*   `sweep_index__micro.csv`
*   One folder per sample containing:
    *   `params.json`
    *   `frames.csv`
    *   Integrated EXR frames
    *   Subsample EXR frames
    *   `objective.txt` (if YOLO is available)

Script 2: full_sweep_spatial_illumination_fps.py

This script performs a four-stage hierarchical sweep:

1.  Coarse sweep
2.  Spatial micro sweep
3.  Illumination micro sweep
4.  Camera micro sweep

Each stage narrows only one group of parameters while freezing all others.

Stage logic

| Stage | Parameters refined | Parameters frozen |
| :--- | :--- | :--- |
| Coarse | All | None |
| Spatial micro | Position and orientation | Illumination, camera, timing |
| Illumination micro | Sun and flash power | Spatial and camera |
| Camera micro | FPS only | All others |

Parameter groups

**Spatial:**

*   `flashlight_x`
*   `flashlight_z`
*   `car_y`
*   `incidence_z_deg`

**Illumination:**

*   `sun_wm2`
*   `flash_strength`

**Camera:**

*   `fps`

**Temporal (frozen in micro stages):**

*   `shutter_time_ms`
*   `flash_freq_hz`
*   `duty_cycle`
*   `phase`

Output

The output directory contains:

*   `sweep_index__coarse.csv`
*   `sweep_index__spatial.csv`
*   `sweep_index__illumination.csv`
*   `sweep_index__camera.csv`
*   Per-sample folders with:
    *   params
    *   frames
    *   EXR images
    *   YOLO objective scores
*   `autopilot_log.txt` with all print output

Requirements

*   Blender with Cycles
*   GPU recommended (CUDA, Optix, HIP, Metal, or OneAPI)
*   Python access inside Blender
*   Internet access for YOLO auto-download
*   Ultralytics package (auto-installed if missing)

How to Run

1.  Open Blender
2.  Load the scene file
3.  Open the script in the Text Editor
4.  Press Run

The sweep starts automatically. All results are saved under:

`~/Desktop/MultiEV/experiments/parameter_sweeps/`

Notes

*   Blender has no visible terminal. All output is written to: `autopilot_log.txt`
*   The scripts do not modify your scene permanently.
*   All transformations are applied per-sample and reset automatically.
*   YOLO scoring is optional. If YOLO is not available, frames are still generated.
