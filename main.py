# main.py — Astro Pi ISS Speed Challenge (submission-safe timing + graceful shutdown)
# AstroDu (Christopher and Raphael) Here's what this script does:
# 1) Take a bunch of pics with the Pi cam (about every 15s) for ~10 minutes.
# 2) Read timestamps (EXIF first, otherwise our own) to know how long between pics.
# 3) Find cool matching dots with SIFT + CLAHE + RANSAC to ditch bad matches.
# 4) Toss matches into one giant list, ignore pairs with too few matches.
# 5) Drop the slowest 95% (keep the speedy top 5%), then average the speeds.
# 6) Save the final speed in result.txt (5 sig figs) and all kept matches in data.csv.

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

from exif import Image
from datetime import datetime
from pathlib import Path
import cv2
import math
import time
import csv
import numpy as np
from picamzero import Camera
from statistics import mean


# -----------------------
# Mission timing settings
# -----------------------
# Mission timing (single set)
DEFAULT_MISSION_DURATION = 120       # seconds (total runtime)
DEFAULT_SHUTDOWN_MARGIN = 10         # seconds reserved for filtering + writing files
DEFAULT_TIME_BETWEEN_IMAGES = 5      # seconds between shots

# Processing / model settings
GSD_CM_PER_PIXEL = 12648        # cm per pixel
MAX_FEATURES = 1000
RANSAC_THRESHOLD = 8
RANSAC_MIN_MATCHES = 20

# Filters
MINIMUM_MATCHES_CONFIG = {"enabled": True, "minimum_matches": 50}
PERCENTILE_KEEP_FRACTION = 0.05  # keep top 5% speeds (drop bottom 95%)


# -----------------------
# EXIF time helpers
# -----------------------
def get_time_from_exif(image_path: str):
    """Return datetime from EXIF datetime_original, or None if missing/unreadable."""
    # Try EXIF first; if it fails, caller will try other stuff
    try:
        with open(image_path, "rb") as f:
            img = Image(f)
        time_str = img.get("datetime_original")
        if not time_str:
            return None
        return datetime.strptime(time_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


def get_exif_dict(image_path: str):
    """Grab all EXIF tags for one image as a dict of strings (kid-proof!)."""
    try:
        with open(image_path, "rb") as f:
            img = Image(f)
            tags = img.list_all()
            exif_data = {}
            for tag in tags:
                try:
                    val = img.get(tag)
                    exif_data[tag] = str(val)
                except Exception:
                    exif_data[tag] = ""
            return exif_data
    except Exception:
        return {}


def get_time_difference_seconds(image_1: str, image_2: str, capture_epoch: dict, fallback_seconds: float):
    """
    Compute dt in seconds using EXIF when available.
    If EXIF missing, use recorded capture_epoch timestamps.
    Never returns <= 0. (No divide-by-zero disasters!)
    """
    # 1) Prefer EXIF
    t1 = get_time_from_exif(image_1)
    t2 = get_time_from_exif(image_2)
    if t1 is not None and t2 is not None:
        dt = (t2 - t1).total_seconds()
        if dt > 0:
            return dt

    # 2) Fallback to our own capture timestamps
    e1 = capture_epoch.get(image_1)
    e2 = capture_epoch.get(image_2)
    if e1 is not None and e2 is not None:
        dt = e2 - e1
        if dt > 0:
            return dt

    # 3) Last resort: configured interval
    return max(float(fallback_seconds), 0.001)


# -----------------------
# Vision helpers (SIFT + CLAHE)
# -----------------------
def convert_to_cv_gray(image_1: str, image_2: str):
    """Load images as grayscale with CLAHE. Returns (img1, img2) or (None, None) if load fails."""
    # CLAHE makes dark space pics pop a bit so SIFT can find more dots
    img1 = cv2.imread(image_1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return None, None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img1 = clahe.apply(img1)
    img2 = clahe.apply(img2)
    return img1, img2


def calculate_features_sift(img1_gray, img2_gray, max_features=1000):
    """Detect SIFT features and descriptors."""
    # SIFT time! Find up to 1000 shiny points per image
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    return kp1, kp2, des1, des2


def calculate_matches_l2(des1, des2):
    """Match SIFT descriptors using L2 distance."""
    # Brute-force match with crossCheck to keep only mutual BFFs
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    return matches


def apply_ransac(keypoints_1, keypoints_2, matches, ransac_threshold=5, min_matches=10):
    """Filter matches via homography RANSAC; returns inlier matches."""
    # RANSAC = kick out weird outliers; keep the good squad
    if matches is None or len(matches) < min_matches:
        return matches or [], None

    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
    if mask is None:
        return matches, None

    mask = mask.ravel().tolist()
    inliers = [matches[i] for i in range(len(matches)) if mask[i] == 1]
    return inliers, H


def calculate_match_speeds(keypoints_1, keypoints_2, matches, time_difference_s, gsd_cm_per_pixel, pair_name):
    """Compute speed for each inlier match; speed in km/s."""
    # Convert pixel jumps -> km/s using GSD and dt
    out = []
    if not matches:
        return out

    dt = max(float(time_difference_s), 0.001)

    for m in matches:
        (x1, y1) = keypoints_1[m.queryIdx].pt
        (x2, y2) = keypoints_2[m.trainIdx].pt
        pixel_distance = math.hypot(x2 - x1, y2 - y1)

        distance_km = (pixel_distance * gsd_cm_per_pixel) / 100000.0  # cm -> km
        speed_km_s = distance_km / dt

        out.append(
            {
                "speed": speed_km_s,
                "pixel_distance": pixel_distance,
                "time_difference": dt,
                "gsd_used": gsd_cm_per_pixel,
                "pair_image_name": pair_name,
            }
        )

    return out


# -----------------------
# Filtering + output
# -----------------------
def apply_filters(all_match_data, minimum_matches_config, keep_top_fraction):
    filtered = list(all_match_data)

    # Filter 1: Minimum matches per pair (skip weak pairs)
    if minimum_matches_config.get("enabled", False):
        pair_counts = {}
        for m in filtered:
            pair_counts[m["pair_image_name"]] = pair_counts.get(m["pair_image_name"], 0) + 1

        min_n = int(minimum_matches_config.get("minimum_matches", 0))
        valid_pairs = {p for p, c in pair_counts.items() if c >= min_n}
        filtered = [m for m in filtered if m["pair_image_name"] in valid_pairs]

    # Filter 2: Percentile keep (drop bottom speeds, keep the speedsters)
    if filtered and keep_top_fraction > 0:
        filtered = sorted(filtered, key=lambda m: m["speed"])
        keep_count = max(1, int(math.ceil(len(filtered) * keep_top_fraction)))
        filtered = filtered[-keep_count:]

    return filtered


def write_result_to_file(final_speed_km_s, path: Path):
    """Write one number to result.txt (up to 5 significant figures, no trailing newline)."""
    with open(path, "w") as f:
        f.write(f"{final_speed_km_s:.5g}")


def write_data_to_csv(match_data, path: Path):
    """Optional: write diagnostics. Safe even if empty."""
    # CSV gets every kept match so we can nerd out later
    fieldnames = ["speed", "pixel_distance", "time_difference", "gsd_used", "pair_image_name"]
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in match_data:
            writer.writerow(row)


def write_exif_to_csv(exif_map, path: Path):
    """Write all EXIF goodies per image into one CSV (one row per image)."""
    if not exif_map:
        return
    # Collect all possible keys across images
    keys = set()
    for _, exif in exif_map.items():
        keys.update(exif.keys())
    fieldnames = ["image_name"] + sorted(keys)

    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for image_path, exif in exif_map.items():
            row = {"image_name": Path(image_path).name}
            for k in keys:
                row[k] = exif.get(k, "")
            writer.writerow(row)


# -----------------------
# Pair processing
# -----------------------
def process_image_pair(
    image_1_path: str,
    image_2_path: str,
    pair_name: str,
    capture_epoch: dict,
    fallback_dt_s: float,
):
    """Return (match_data_list, inlier_count). Never raises."""
    dt_s = get_time_difference_seconds(image_1_path, image_2_path, capture_epoch, fallback_dt_s)
    # Load, feature, match, RANSAC, compute speeds — all in one go

    img1, img2 = convert_to_cv_gray(image_1_path, image_2_path)
    if img1 is None or img2 is None:
        return [], 0

    kp1, kp2, des1, des2 = calculate_features_sift(img1, img2, max_features=MAX_FEATURES)
    if des1 is None or des2 is None or kp1 is None or kp2 is None:
        return [], 0

    try:
        matches = calculate_matches_l2(des1, des2)
    except Exception:
        return [], 0

    inliers, _H = apply_ransac(kp1, kp2, matches, ransac_threshold=RANSAC_THRESHOLD, min_matches=RANSAC_MIN_MATCHES)

    if len(inliers) < MINIMUM_MATCHES_CONFIG.get("minimum_matches", 0):
        return [], len(inliers)

    data = calculate_match_speeds(
        kp1,
        kp2,
        inliers,
        dt_s,
        GSD_CM_PER_PIXEL,
        pair_name,
    )
    return data, len(inliers)


# -----------------------
# Main program
# -----------------------
def main():
    import sys

    BASE_DIR = Path(__file__).parent
    result_path = BASE_DIR / "result.txt"
    data_path = BASE_DIR / "data.csv"
    exif_path = BASE_DIR / "exif_data.csv"

    # Pick timing settings (fast mode if user passes "fast")
    use_fast = len(sys.argv) > 1 and sys.argv[1].lower() == "fast"
    mission_duration = FAST_MISSION_DURATION if use_fast else DEFAULT_MISSION_DURATION
    shutdown_margin = FAST_SHUTDOWN_MARGIN if use_fast else DEFAULT_SHUTDOWN_MARGIN
    time_between_images = FAST_TIME_BETWEEN_IMAGES if use_fast else DEFAULT_TIME_BETWEEN_IMAGES
    capture_duration = mission_duration - shutdown_margin
    print(f"Mode: {'FAST' if use_fast else 'NORMAL'} | mission={mission_duration}s, gap={time_between_images}s, shutdown={shutdown_margin}s")

    mission_start = time.time()
    mission_end = mission_start + mission_duration
    capture_end = mission_start + capture_duration

    print("Initializing camera...")  # turn on the space cam!
    cam = Camera()

    # Track capture times for robust dt fallback (epoch seconds)
    capture_epoch = {}
    # Stash EXIF per image so we can dump them later
    exif_map = {}

    all_match_data = []

    # Capture the first image immediately (counts inside 10 minutes)
    image_number = 1
    image_1 = str(BASE_DIR / "image_001.jpg")
    print(f"Capturing image {image_number:03d} ...")
    cam.take_photo(image_1)
    capture_epoch[image_1] = time.time()
    exif_map[image_1] = get_exif_dict(image_1)

    # Main capture+process loop (ends early to leave shutdown margin)
    while time.time() < capture_end:
        # Sleep until next capture slot, but never overrun capture_end
        remaining_to_capture_end = capture_end - time.time()
        if remaining_to_capture_end <= 0:
            break
        time.sleep(min(time_between_images, remaining_to_capture_end))

        if time.time() >= capture_end:
            break

        image_number += 1
        image_2 = str(BASE_DIR / f"image_{image_number:03d}.jpg")

        print(f"Capturing image {image_number:03d} ...")
        cam.take_photo(image_2)
        capture_epoch[image_2] = time.time()
        exif_map[image_2] = get_exif_dict(image_2)

        pair_name = f"image_{image_number-1:03d}_to_image_{image_number:03d}"
        print(f"Processing pair: {pair_name}")

        match_data, inlier_count = process_image_pair(
            image_1_path=image_1,
            image_2_path=image_2,
            pair_name=pair_name,
            capture_epoch=capture_epoch,
            fallback_dt_s=time_between_images,
        )

        print(f"Inlier matches: {inlier_count} | Collected rows: {len(match_data)}")
        all_match_data.extend(match_data)

        # Slide window (reuse last image)
        image_1 = image_2

        # If we’re dangerously close to the mission end, stop capturing/processing now
        if (mission_end - time.time()) <= shutdown_margin:
            break

    # Graceful shutdown phase (guaranteed time window)
    print("=" * 60)
    print("CAPTURE PHASE COMPLETE — SHUTDOWN PHASE")
    print("=" * 60)
    print(f"Images saved: {image_number}")
    print(f"Rows before filtering: {len(all_match_data)}")

    filtered = apply_filters(all_match_data, MINIMUM_MATCHES_CONFIG, PERCENTILE_KEEP_FRACTION)
    print(f"Rows after filtering:  {len(filtered)}")

    if filtered:
        final_speed = mean([m["speed"] for m in filtered])
    else:
        final_speed = 0.0

    # Write outputs
    write_result_to_file(final_speed, result_path)
    write_data_to_csv(filtered, data_path)
    write_exif_to_csv(exif_map, exif_path)

    print(f"Final speed written to: {result_path.name}")
    print(f"Diagnostics written to: {data_path.name}")
    print(f"EXIF dump written to: {exif_path.name}")

    # Ensure we exit within the 10-minute window
    now = time.time()
    if now < mission_end:
        # No need to sleep; just exit early.
        pass
    else:
        # We overran; still exit cleanly.
        print("Note: mission end reached during shutdown; exiting now.")


if __name__ == "__main__":
    main()
