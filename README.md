# ISS Speed Calculation Project

**Team:** astroduo  
**Members:** Christopher and Raphael (Year 9 Chancellors)  
**Challenge:** [Astro Pi Mission Space Lab](https://astro-pi.org/mission-space-lab)

## What this program does (high level)
1) Take photos with the Pi camera about every 15s for ~10 minutes (keeps time for shutdown).  
2) Figure out the time gap between photos (EXIF if present; otherwise our own capture timestamps; otherwise the configured interval).  
3) Find matching dots between each photo pair using **SIFT** (1000 features) + CLAHE + BFMatcher (L2) + **RANSAC**.  
4) Put ALL inlier matches from every pair into one big list; drop any pair with < 50 matches; then drop the bottom 95% of speeds (keep top 5%).  
5) Turn the good matches into per-match speeds, average them, and write the final speed.  
6) Save one number to `result.txt` (≤5 sig figs), all kept matches to `data.csv`, and dump per-image EXIF tags to `exif_data.csv`.

## Current pipeline (main.py)
- Capture loop: ~10 minutes total (600s) with 15s between photos; images saved flat as `image_XXX.jpg`.  
- Timing: prefers EXIF `datetime_original`; falls back to recorded capture timestamps; otherwise uses the configured interval.  
- Features: SIFT keypoints/descriptors (max 1000) with CLAHE pre-processing; BFMatcher with L2; RANSAC homography (threshold 8, min matches 20) for inliers.  
- Filters: drop pairs with < 50 inlier matches; then drop the bottom 95% of speeds (keep top 5%).  
- Stats/output: final mean speed written to `result.txt`; filtered match rows (speed, pixel_distance, time_difference, gsd_used, pair_image_name) written to `data.csv`; per-image EXIF tags written to `exif_data.csv`.  
- GSD: 12,648 cm/pixel.  
- Final speed format: ≤5 significant figures.

## Usage
From the repo root:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 astro-pi-replay run main.py          # normal
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 astro-pi-replay run main.py fast     # fast mode (60s run)
```
(If running on-device with the real camera, just `python3 main.py`.)

Outputs:
- `result.txt` — one line with the mean speed (km/s, ≤5 sig figs)
- `data.csv` — filtered match diagnostics
- `exif_data.csv` — one row per image with all EXIF tags available
- Captured images — `image_001.jpg`, `image_002.jpg`, … in the project root

## Dependencies
- Python 3.11+
- `opencv-contrib-python` (for SIFT/RANSAC)
- `numpy`
- `exif`
- `picamzero` (camera interface on Astro Pi / replay)

## Notes / limitations
- Needs EXIF timestamps or fallback to the captured timestamps; we guard against missing EXIF.  
- Filters drop weak pairs (<50 matches) and outliers (std-dev).  
- Replay on macOS may require limiting OpenBLAS threads; set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1` when running.  
- Images are stored flat (no subfolders).

## Repo layout (key files)
```
ISS_speed/
├── main.py
├── README.md
├── .gitignore
└── (runtime outputs) result.txt, data.csv, image_XXX.jpg
```

## License
This project is part of the Astro Pi Mission Space Lab challenge. See the
[Astro Pi website](https://astro-pi.org/) for challenge guidelines and terms.
