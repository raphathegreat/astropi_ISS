# main.py — Astro Pi ISS Speed Challenge (submission-safe timing + graceful shutdown)
# High-level plan (kid style):
# 1) Take a bunch of pics with the Pi cam (about every 15s).
# 2) Figure out the time between pics (use EXIF if it exists, else our timer).
# 3) Find matching dots with SIFT for each photo pair.
# 4) Dump ALL matches into one giant list; if a pair has <50 matches, ignore it; toss out weird outliers.
# 5) Turn the good matches into speeds and average them; write the speed to result.txt and details to data.csv.

from exif import Image
from datetime import datetime
from pathlib import Path
import cv2
import math
import time
import csv
import numpy as np
from picamzero import Camera
from statistics import mean, stdev


# -----------------------
# Mission timing settings
# -----------------------
MISSION_DURATION = 600          # seconds (total runtime from start to finish)
SHUTDOWN_MARGIN = 20            # seconds reserved for filtering + writing files
CAPTURE_DURATION = MISSION_DURATION - SHUTDOWN_MARGIN

# Capture cadence (choose to keep <= 42 images at end)
TIME_BETWEEN_IMAGES = 15        # seconds (600/15 ≈ 40 images incl first)

# Processing / model settings
GSD_CM_PER_PIXEL = 12648        # cm per pixel
MAX_FEATURES = 1500
RANSAC_THRESHOLD = 8
RANSAC_MIN_MATCHES = 20

# Filters
MINIMUM_MATCHES_CONFIG = {"enabled": True, "minimum_matches": 50}
STANDARD_DEVIATION_CONFIG = {"enabled": True, "multiplier": 2}


# -----------------------
# EXIF time helpers
# -----------------------
def get_time_from_exif(image_path: str):
    """Return datetime from EXIF datetime_original, or None if missing/unreadable."""
    try:
        with open(image_path, "rb") as f:
            img = Image(f)
            time_str = img.get("datetime_original")
            if not time_str:
                return None
            return datetime.strptime(time_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


def get_time_difference_seconds(image_1: str, image_2: str, capture_epoch: dict, fallback_seconds: float):
    """
    Compute dt in seconds using EXIF when available.
    If EXIF missing, use recorded capture_epoch timestamps.
    Never returns <= 0.
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
# Vision helpers (ORB recommended over SIFT for portability)
# -----------------------
def convert_to_cv_gray(image_1: str, image_2: str):
    """Load images as grayscale. Returns (img1, img2) or (None, None) if load fails."""
    img1 = cv2.imread(image_1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return None, None
    return img1, img2


def calculate_features_orb(img1_gray, img2_gray, max_features=1000):
    """Detect ORB features and descriptors."""
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)
    return kp1, kp2, des1, des2


def calculate_matches_hamming(des1, des2):
    """Match ORB descriptors using Hamming distance."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    return matches


def apply_ransac(keypoints_1, keypoints_2, matches, ransac_threshold=5, min_matches=10):
    """Filter matches via homography RANSAC; returns inlier matches."""
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
def apply_filters(all_match_data, minimum_matches_config, standard_deviation_config):
    filtered = list(all_match_data)

    # Filter 1: Minimum matches per pair
    if minimum_matches_config.get("enabled", False):
        pair_counts = {}
        for m in filtered:
            pair_counts[m["pair_image_name"]] = pair_counts.get(m["pair_image_name"], 0) + 1

        min_n = int(minimum_matches_config.get("minimum_matches", 0))
        valid_pairs = {p for p, c in pair_counts.items() if c >= min_n}
        filtered = [m for m in filtered if m["pair_image_name"] in valid_pairs]

    # Filter 2: Remove outliers based on standard deviation
    if standard_deviation_config.get("enabled", False) and len(filtered) > 1:
        speeds = [m["speed"] for m in filtered]
        mu = mean(speeds)
        sigma = stdev(speeds)
        k = float(standard_deviation_config.get("multiplier", 2))
        thresh = k * sigma
        if thresh > 0:
            filtered = [m for m in filtered if abs(m["speed"] - mu) <= thresh]

    return filtered


def write_result_to_file(final_speed_km_s, path: Path):
    """Write one number (<= 5 significant figures) to result.txt."""
    with open(path, "w") as f:
        f.write(f"{final_speed_km_s:.5g}\n")


def write_data_to_csv(match_data, path: Path):
    """Optional: write diagnostics. Safe even if empty."""
    fieldnames = ["speed", "pixel_distance", "time_difference", "gsd_used", "pair_image_name"]
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in match_data:
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

    img1, img2 = convert_to_cv_gray(image_1_path, image_2_path)
    if img1 is None or img2 is None:
        return [], 0

    kp1, kp2, des1, des2 = calculate_features_orb(img1, img2, max_features=MAX_FEATURES)
    if des1 is None or des2 is None or kp1 is None or kp2 is None:
        return [], 0

    try:
        matches = calculate_matches_hamming(des1, des2)
    except Exception:
        return [], 0

    inliers, _H = apply_ransac(kp1, kp2, matches, ransac_threshold=RANSAC_THRESHOLD, min_matches=RANSAC_MIN_MATCHES)

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
    BASE_DIR = Path(__file__).parent
    result_path = BASE_DIR / "result.txt"
    data_path = BASE_DIR / "data.csv"

    mission_start = time.time()
    mission_end = mission_start + MISSION_DURATION
    capture_end = mission_start + CAPTURE_DURATION

    print("Initializing camera...")
    cam = Camera()

    # Track capture times for robust dt fallback (epoch seconds)
    capture_epoch = {}

    all_match_data = []

    # Capture the first image immediately (counts inside 10 minutes)
    image_number = 1
    image_1 = str(BASE_DIR / "image_001.jpg")
    print(f"Capturing image {image_number:03d} ...")
    cam.take_photo(image_1)
    capture_epoch[image_1] = time.time()

    # Main capture+process loop (ends early to leave shutdown margin)
    while time.time() < capture_end:
        # Sleep until next capture slot, but never overrun capture_end
        remaining_to_capture_end = capture_end - time.time()
        if remaining_to_capture_end <= 0:
            break
        time.sleep(min(TIME_BETWEEN_IMAGES, remaining_to_capture_end))

        if time.time() >= capture_end:
            break

        image_number += 1
        image_2 = str(BASE_DIR / f"image_{image_number:03d}.jpg")

        print(f"Capturing image {image_number:03d} ...")
        cam.take_photo(image_2)
        capture_epoch[image_2] = time.time()

        pair_name = f"image_{image_number-1:03d}_to_image_{image_number:03d}"
        print(f"Processing pair: {pair_name}")

        match_data, inlier_count = process_image_pair(
            image_1_path=image_1,
            image_2_path=image_2,
            pair_name=pair_name,
            capture_epoch=capture_epoch,
            fallback_dt_s=TIME_BETWEEN_IMAGES,
        )

        print(f"Inlier matches: {inlier_count} | Collected rows: {len(match_data)}")
        all_match_data.extend(match_data)

        # Slide window
        image_1 = image_2

        # If we’re dangerously close to the mission end, stop capturing/processing now
        if (mission_end - time.time()) <= SHUTDOWN_MARGIN:
            break

    # Graceful shutdown phase (guaranteed time window)
    print("=" * 60)
    print("CAPTURE PHASE COMPLETE — SHUTDOWN PHASE")
    print("=" * 60)
    print(f"Images saved: {image_number}")
    print(f"Rows before filtering: {len(all_match_data)}")

    filtered = apply_filters(all_match_data, MINIMUM_MATCHES_CONFIG, STANDARD_DEVIATION_CONFIG)
    print(f"Rows after filtering:  {len(filtered)}")

    if filtered:
        final_speed = mean([m["speed"] for m in filtered])
    else:
        final_speed = 0.0

    # Write outputs
    write_result_to_file(final_speed, result_path)
    write_data_to_csv(filtered, data_path)

    print(f"Final speed written to: {result_path.name}")
    print(f"Diagnostics written to: {data_path.name}")

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
