# ISS Speed Calculation Project

**Team:** astroduo  
**Members:** Christopher and Raphael (Year 9 Chancellors)  
**Challenge:** [Astro Pi Mission Space Lab](https://astro-pi.org/mission-space-lab)

## Overview

This project calculates the speed of the International Space Station (ISS) using image processing techniques. By analyzing consecutive images taken from the ISS, we detect matching features between images, measure their displacement, and calculate the orbital speed based on the time difference and ground sample distance (GSD).

## Challenge Context

As part of the European Space Agency's Astro Pi Mission Space Lab challenge, we are using images captured by the Astro Pi computer on board the ISS to perform scientific analysis. This project focuses on calculating the orbital velocity of the ISS using computer vision and image processing.

## How It Works

### Methodology

1. **Image Acquisition**: Two consecutive images are loaded from the ISS photo collection
2. **Time Extraction**: EXIF metadata is read to extract the `datetime_original` timestamp from each image
3. **Feature Detection**: ORB (Oriented FAST and Rotated BRIEF) algorithm detects key features in both images
4. **Feature Matching**: Brute-force matcher finds corresponding features between the two images
5. **Displacement Calculation**: The average pixel displacement of matched features is calculated
6. **Speed Calculation**: Using the GSD (Ground Sample Distance) and time difference, the speed is calculated:
   ```
   Real Distance = Pixel Distance × GSD / 100,000 (converts to km)
   Speed = Real Distance / Time Difference
   ```

### Key Parameters

- **GSD (Ground Sample Distance)**: 12,648 meters per pixel
- **Feature Detection**: ORB with 10,000 features
- **Matching Method**: Brute-force matcher with Hamming distance

## Installation

### Prerequisites

- Python 3.7 or higher
- OpenCV (`cv2`)
- exif library

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/raphathegreat/astropi_ISS.git
   cd astropi_ISS
   ```

2. Install required dependencies:
   ```bash
   pip install opencv-python exif
   ```

3. Ensure you have image directories (`photos-1/` and `photos-2/`) with ISS images (these are excluded from git due to size)

## Usage

### Basic Usage

Edit the image paths in `iss_v1.py`:

```python
image_1 = 'photos-1/20230423-112638_53238988145_o.jpg'
image_2 = 'photos-1/20230423-112652_53238502736_o.jpg'
```

Then run:

```bash
python iss_v1.py
```

The script will output the calculated speed in km/s.

### Understanding the Output

The script prints a single value representing the calculated speed of the ISS in kilometers per second (km/s). The expected orbital speed of the ISS is approximately **7.66 km/s**.

## Project Structure

```
ISS_speed/
├── iss_v1.py              # Main calculation script
├── iss_v1_backup.py       # Backup version (not tracked)
├── README.md              # This file
├── DEVELOPMENT_RULES.md   # Development guidelines
├── GITHUB_WORKFLOW.md     # Git workflow documentation
├── SAFETY_CHECKS.md       # Safety and validation checks
├── .gitignore            # Git ignore rules (excludes photos)
├── photos-1/              # ISS images (not tracked in git)
├── photos-2/              # Additional ISS images (not tracked in git)
└── data/                  # Data files (optional)
```

## Technical Details

### Functions

- `get_time(image)`: Extracts datetime from EXIF metadata
- `get_time_difference(image_1, image_2)`: Calculates time difference in seconds
- `convert_to_cv(image_1, image_2)`: Converts images to OpenCV grayscale format
- `calculate_features(image_1, image_2, feature_number)`: Detects ORB features
- `calculate_matches(descriptors_1, descriptors_2)`: Matches features using brute-force
- `find_matching_coordinates(keypoints_1, keypoints_2, matches)`: Extracts coordinate pairs
- `calculate_mean_distance(coordinates_1, coordinates_2)`: Calculates average pixel displacement
- `calculate_speed_in_kmps(feature_distance, GSD, time_difference)`: Converts to speed

### Image Processing Pipeline

1. **Grayscale Conversion**: Images are converted to grayscale for feature detection
2. **Feature Detection**: ORB algorithm finds distinctive keypoints
3. **Descriptor Matching**: Features are matched based on descriptor similarity
4. **Coordinate Extraction**: Pixel coordinates of matched features are extracted
5. **Distance Calculation**: Euclidean distance between matched points is calculated
6. **Averaging**: Mean displacement across all matches provides the average movement

## Dependencies

- **opencv-python**: Computer vision and image processing
- **exif**: EXIF metadata extraction from images
- **datetime**: Time calculation (built-in)
- **math**: Mathematical operations (built-in)

## Limitations & Considerations

- **Image Quality**: Results depend on image sharpness and feature visibility
- **Time Interval**: Very short or very long intervals may affect accuracy
- **GSD Accuracy**: Ground Sample Distance is an approximation
- **Feature Matching**: Poor image quality may result in fewer or incorrect matches
- **Single Pair**: Current implementation processes one image pair at a time

## Future Improvements

Potential enhancements for future versions:

- [ ] Process multiple image pairs and calculate statistics (mean, median, mode)
- [ ] Implement RANSAC for outlier removal
- [ ] Support for different feature detectors (SIFT, AKAZE)
- [ ] Image quality assessment and filtering
- [ ] Batch processing of all image pairs
- [ ] Statistical analysis of results
- [ ] Visualization of matched features
- [ ] Error estimation and confidence intervals

## Contributing

This is a school project for the Astro Pi challenge. For questions or collaboration, please contact the team members.

## License

This project is part of the Astro Pi Mission Space Lab challenge. Please refer to the [Astro Pi website](https://astro-pi.org/) for challenge guidelines and terms.

## References

- [Astro Pi Mission Space Lab](https://astro-pi.org/mission-space-lab)
- [OpenCV Documentation](https://docs.opencv.org/)
- [ORB Feature Detector](https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html)
- [ISS Orbital Speed](https://www.nasa.gov/feature/facts-and-figures)

## Acknowledgments

- European Space Agency (ESA) for the Astro Pi challenge
- Raspberry Pi Foundation for the Astro Pi program
- Year 9 Chancellors for support and resources

---

**Note**: The `photos-1/` and `photos-2/` directories containing ISS images are excluded from git tracking due to their large size. These images must be obtained separately or are available locally for development.
