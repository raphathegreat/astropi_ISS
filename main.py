from exif import Image
from datetime import datetime

import cv2
import math


def get_time(image):
    # Extract the datetime from EXIF data - this is when the photo was actually taken
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
        return time
            
def get_time_difference(image_1, image_2):
    # Calculate how many seconds passed between the two photos
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.seconds


#print(get_time_difference('ISS_speed/photos-1/20230423-112610_53237634632_o.jpg','ISS_speed/photos-1/20230423-112652_53238502736_o.jpg')) 
        
def convert_to_cv(image_1, image_2):
    # Load images as grayscale (0 = grayscale mode) for feature detection
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv

def calculate_features(image_1, image_2, feature_number):
    # ORB is good for this - fast and works well with ISS images
    # feature_number limits how many keypoints we detect (10k seems to work well)
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

def calculate_matches(descriptors_1, descriptors_2):
    # Hamming distance works with ORB descriptors, crossCheck ensures bidirectional matching
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    # Sort by distance so best matches come first
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    # Just for debugging - shows first 100 matches visually
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize)
    cv2.waitKey(0)
    cv2.destroyWindow('matches')
    
def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    # Extract the actual pixel coordinates for each matched keypoint pair
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))
    return coordinates_1, coordinates_2

def calculate_mean_distance(coordinates_1, coordinates_2):
    # Calculate pixel distance between each matched pair, then average them
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances = all_distances + distance
    return all_distances / len(merged_coordinates)

def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    # GSD is in cm/pixel, convert to km then divide by time to get km/s
    # 100000 converts cm to km
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed
        

image_1 = 'photos-1/20230423-112638_53238988145_o.jpg'
image_2 = 'photos-1/20230423-112652_53238502736_o.jpg'

time_difference = get_time_difference(image_1, image_2)
image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)
keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 10000)
matches = calculate_matches(descriptors_1, descriptors_2)
#display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches)  # uncomment to visualize matches
coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
# GSD value from ISS camera specs - 12648 cm/pixel
speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference)

# Display keypoint information
print(f"Keypoints detected in image 1: {len(keypoints_1)}")
print(f"Keypoints detected in image 2: {len(keypoints_2)}")
print(f"Number of matches found: {len(matches)}")
print(f"Number of keypoints used in speed calculation: {len(coordinates_1)}")
print(f"Calculated speed: {speed} km/s")