import numpy as np
import cv2
from pathlib import Path
import os
import torch
import logging
PROJECT_ROOT = Path(__file__).parents[8].resolve()
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "source/car_training/car_training/tasks/manager_based/car_training/mdp/utils/test_data/")

if not os.path.exists(TEST_DATA_PATH):
    os.makedirs(TEST_DATA_PATH)

#### For all of the functions below, we compute reward for only a single copy of the simulation. If we have multiple copies running in parallel, we can call the functions below repeatedly, and that is what we do above. ####

logger = logging.getLogger("car_training" + __name__.rpartition(".car_training")[-1])

def find_track_points_from_image(image: np.ndarray, debug = False):
    """
    Process the image to find track points.

    :param image: Input image from the car's camera.
    :type image: np.ndarray
    :param debug: If True, will save some debug images. The images will be save in test_data folder.
    :type debug: bool
    :return: List of track points as (x, y) tuples.
    :rtype: list[tuple[float, float]]
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold (invert so line is white)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Optional: clean small noise
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,   # only outer contour
        cv2.CHAIN_APPROX_NONE  # keep all points
    )

    # Select the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Extract coordinates
    track_points = contour.squeeze()  # shape (N, 2)

    logger.debug(f"Number of track points detected: {len(track_points)}")

    # if debug draw the track points on the image
    if debug:
        debug_image = image.copy()
        for point in track_points:
            cv2.circle(debug_image, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
        cv2.imwrite(f"{TEST_DATA_PATH}/debug_track_points.png", debug_image)

    # Debugging: Save images if debug is True
    if debug:
        cv2.imwrite(f"{TEST_DATA_PATH}/debug_gray.png", gray)
        debug_contour_image = image.copy()
        cv2.drawContours(debug_contour_image, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(f"{TEST_DATA_PATH}/debug_contours.png", debug_contour_image)

    return track_points


# TODO: This reward is not being used right now!
def track_completion_reward(completed_laps: int, total_laps: int, weight=1.0):
    """
    Calculate reward based on the completion of laps.

    :param completed_laps: Number of laps completed by the car.
    :type completed_laps: int
    :param total_laps: Total number of laps in the race.
    :type total_laps: int
    :param weight: Weight for the reward.
    :type weight: float
    :return: Reward based on the completion of laps.
    :rtype: float
    """
    if total_laps <= 0:
        return 0.0  # Avoid division by zero
    normalized_completion = completed_laps / total_laps
    reward = normalized_completion * weight
    return reward


def distance_from_center_reward(car_center: tuple[float, float], track_points_list: list[tuple[float, float]], weight=1.0):
    """
    Calculate reward based on the car's distance from the center of the track.

    :param car_center: Tuple representing the car's center coordinates (x, y).
    :type car_center: tuple[float, float]
    :param track_points_list: List of tuples representing the track points (x, y).
    :type track_points_list: list[tuple[float, float]]
    :param weight: Weight for the reward.
    :type weight: float

    :return: Reward based on the distance from the center of the track.
    :rtype: float
    
    """
    closest_track_point = __find_closest_track_point(car_center, track_points_list)
    distance = np.linalg.norm(np.array(car_center) - np.array(closest_track_point))
    logger.debug(f"Distance from center: {distance}")
    # print(f"Distance from center: {distance}")
    # Assuming a maximum distance for normalization, e.g., the width of the track
    max_distance = 20.0  # This should be defined based on the track's characteristics
    normalized_distance = distance / max_distance
    # Reward is higher when the distance is smaller
    reward = 1.0 - normalized_distance
    return reward * weight


def __find_closest_track_point(car_center: tuple[float, float], track_points_list: list[tuple[float, float]], weight=1.0):
    """
    Find the closest track point to the car's center. If the track points list is empty, return (0.0, 0.0).
    
    :param car_center: Tuple representing the car's center coordinates (x, y).
    :type car_center: tuple[float, float]
    :param track_points_list: List of tuples representing the track points (x, y).
    :type track_points_list: list[tuple[float, float]]
    :param weight: Weight for the reward.
    :type weight: float

    :return: The closest track point to the car's center.
    :rtype: tuple[float, float]
    """
    if len(track_points_list) == 0:
        return (0.0, 0.0)  # Return a default value if the list is empty

    distances = [np.linalg.norm(np.array(car_center) - np.array(point)) for point in track_points_list]
    closest_index = np.argmin(distances)
    return track_points_list[closest_index] * weight


def velocity_reward(car_speed: float, max_speed: float, weight=1.0):
    """
    Calculates reward based on the car's speed relative to the maximum speed.

    :param car_speed: Current speed of the car.
    :type car_speed: float
    :param max_speed: Maximum speed of the car.
    :type max_speed: float
    :param weight: Weight for the reward.
    :type weight: float
    """
    normalized_speed = car_speed / max_speed
    # Reward is higher when the speed is closer to the maximum speed
    reward = normalized_speed * weight
    return reward


def remains_alive_reward(weight=1.0):
    """
    Returns a constant reward for remaining alive.

    :param weight: Weight for the reward.
    :type weight: float

    Returns:
        float: Constant reward value.
    """
    reward = 1.0 * weight
    return reward


def leaves_track_penalty(car_center: tuple[float, float], track_points_list: list[tuple[float, float]], weight=1.0):
    """
    Calculate penalty if the car leaves the track.

    :param car_center: Tuple representing the car's center coordinates (x, y).
    :type car_center: tuple[float, float]
    :param track_points_list: List of tuples representing the track points (x, y).
    :type track_points_list: list[tuple[float, float]]
    :param weight: Weight for the penalty.
    :type weight: float

    :return: Penalty value if the car leaves the track, otherwise 0.
    :rtype: float
    """
    closest_track_point = __find_closest_track_point(car_center, track_points_list, weight)
    distance = np.linalg.norm(np.array(car_center) - np.array(closest_track_point))
    # Assuming a threshold distance to determine if the car is off track
    threshold_distance = 5.0  # This should be defined based on the track's characteristics
    penalty = 0.0
    if distance > threshold_distance:
        penalty = 1.0 - (distance / threshold_distance) * weight  # Normalize the penalty
        penalty = max(penalty, 0.0)  # Ensure penalty is not negative
    else:
        penalty = 0.0  # No penalty if the car is within the track limits
    return penalty


def get_green_car_mask(image, debug=False):
    """
    Generate a mask for the green car in the image.

    :param image: Input image from the car's camera.
    :type image: np.ndarray

    :return: Binary mask where the green car is detected.
    :rtype: np.ndarray
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for green color in HSV
    lower_green = np.array([40, 80, 100])
    upper_green = np.array([120, 255, 255])

    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Debugging: Save mask image if debug is True
    if debug:
        cv2.imwrite(f"{TEST_DATA_PATH}/debug_green_car_mask.png", mask)

    return mask


def get_green_car_center(image, debug=False):
    """
    Detect the green car in the image and return its center coordinates.

    :param image: Input image from the car's camera.
    :type image: np.ndarray

    :return: Center coordinates of the green car as (x, y) tuple. Returns None if the car is not found.
    :rtype: tuple[float, float] | None
    """
    mask = get_green_car_mask(image, debug=debug)

    # Find contours of the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour which will correspond to the car
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if debug:
                debug_image = image.copy()
                cv2.circle(debug_image, (cX, cY), 5, (255, 0, 0), -1)
                cv2.imwrite(f"{TEST_DATA_PATH}/debug_green_car_center.png", debug_image)
            return (cX, cY)
    
    return (0,0)  # Car not found


def convert_coords(
    isaaclab_coords: torch.Tensor,
    isaaclab_dimensions: tuple[float, float],
    image_dimensions: tuple[float, float],
    device=None,
    dtype=torch.float32
) -> torch.Tensor:
    """
    Convert Isaaclab coordinates to image coordinates. 
    Isaaclab origin is at center, image origin is at top-left.
    Accepts and returns torch tensors.
    """
    # Ensure input is a tensor
    if not torch.is_tensor(isaaclab_coords):
        isaaclab_coords = torch.tensor(isaaclab_coords, dtype=dtype, device=device)

    width_isaaclab, height_isaaclab = isaaclab_dimensions
    width_image, height_image = image_dimensions

    # Unpack coordinates
    x_isaaclab = isaaclab_coords[..., 0]
    y_isaaclab = isaaclab_coords[..., 1]

    # Convert coordinates
    x_image = (x_isaaclab + width_isaaclab / 2) * (width_image / width_isaaclab)
    y_image = (height_isaaclab / 2 - y_isaaclab) * (height_image / height_isaaclab)

    # Stack and return as tensor
    image_coords = torch.stack((x_image, y_image), dim=-1)
    if device is not None:
        image_coords = image_coords.to(device)
    return image_coords.type(dtype)


if __name__ == "__main__":
    test_image = cv2.imread("debug_image_testing.png")
    car_center = get_green_car_center(test_image, debug=True)
    print(f"Detected green car center: {car_center}")

    track_points = find_track_points_from_image(test_image, debug=True)
    print(f"Number of track points detected: {len(track_points)}")

    closest_point = __find_closest_track_point(car_center, track_points)
    print(f"Closest track point to car center: {closest_point}")

    isaaclab_coords = (2.5, 2.5)
    isaaclab_dimensions = (10.0, 10.0)  # Example dimensions
    image_dimensions = (512, 512)  # Example image dimensions
    image_coords = convert_coords(isaaclab_coords, isaaclab_dimensions, image_dimensions)
    print(f"Converted image coordinates: {image_coords}")

    cv2.circle(test_image, (int(closest_point[0]), int(closest_point[1])), 5, (0, 255, 0), -1)
    cv2.imshow("Car and Closest Track Point", test_image)
    cv2.waitKey(0)







