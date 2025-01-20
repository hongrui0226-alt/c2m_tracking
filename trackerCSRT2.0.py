# Channel and Spatial Reliability Tracker


import os
import time
import cv2
from utils.common import resize_frame, binarize_image, background_subtraction, get_image_files, generate_distinct_colors, save_bboxes, load_bboxes


def preprocess_frame(frame, background):
    """
    Preprocess a frame for tracking by applying background subtraction and binarization.
    
    Args:
        frame (numpy.ndarray): The frame to process
        background (numpy.ndarray): Background image for subtraction
    
    Returns:
        numpy.ndarray: Preprocessed frame for tracking
    """
    frame_subtraction = background_subtraction(frame, background)
    frame_binarized = binarize_image(frame_subtraction,
                                     method='global',
                                     threshold_value=1)
    return frame_binarized


def get_tracking_bboxes(first_frame, scale, bboxes_path):
    """
    Get bounding boxes either from saved file or by interactive selection.
    
    Args:
        first_frame_resized (numpy.ndarray): Resized first frame
        scale (float): Scaling factor
        bboxes_path (str): Path to saved bounding boxes file
    
    Returns:
        list: List of selected bounding boxes
    """
    saved_bboxes = load_bboxes(bboxes_path)

    if saved_bboxes:
        return saved_bboxes

    tracked_object_boxes = []
    first_frame_resized = resize_frame(first_frame, scale)
    while True:
        bbox_scaled = cv2.selectROI('MultiTracker',
                                    first_frame_resized,
                                    fromCenter=False,
                                    showCrosshair=True)

        if (bbox_scaled[2] * bbox_scaled[3]) > 70000 * scale * scale:
            break

        bbox = (
            int(bbox_scaled[0] / scale),  # x
            int(bbox_scaled[1] / scale),  # y
            int(bbox_scaled[2] / scale),  # width
            int(bbox_scaled[3] / scale)  # height
        )
        tracked_object_boxes.append(bbox)

    save_bboxes(tracked_object_boxes, bboxes_path)
    return tracked_object_boxes


def initialize_object_trackers(frame, bboxes, max_colors):
    """
    Initialize multi-tracker with distinct colors for each tracker.
    
    Args:
        frame (numpy.ndarray): First frame to initialize trackers
        bboxes (list): List of bounding boxes
        max_colors (int): Maximum number of distinct colors
    
    Returns:
        tuple: Multi-tracker and tracker colors dictionary
    """
    object_trackers = []
    tracker_colors_list = generate_distinct_colors(max_colors)
    tracker_color_map = {}

    for bbox in bboxes:
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(frame, bbox)
        object_trackers.append(tracker)

        current_color = tracker_colors_list[len(object_trackers) - 1]
        tracker_color_map[tracker] = current_color

        # # Alternative trackers for each selected target area
        # multi_tracker.add(cv2.legacy.TrackerKCF_create(), first_frame, bbox)
        # multi_tracker.add(cv2.legacy.TrackerMIL_create(), first_frame, bbox)
        # multi_tracker.add(cv2.legacy.TrackerTLD_create(), first_frame, bbox)
        # multi_tracker.add(cv2.legacy.TrackerMOSSE_create(), first_frame, bbox)
        # multi_tracker.add(cv2.legacy.TrackerCSRT_create(), first_frame, bbox) # Nice
        # multi_tracker.add(cv2.legacy.TrackerMedianFlow_create(), first_frame, bbox)
        # multi_tracker.add(cv2.legacy.TrackerBoosting_create(), first_frame, bbox) #Nice

    return object_trackers, tracker_color_map


def track_objects(frame_files, background, object_trackers, tracker_color_map,
                  frame_delay, scale, interval):
    """
    Track objects across multiple frames.
    
    Args:
        frame_files (list): List of frame file paths
        background (numpy.ndarray): Background image
        object_trackers (list): List of trackers
        tracker_color_map (dict): Colors for each tracker
        frame_delay (int): Delay between frames
        scale (float): Scaling factor
        interval (int): Frame processing interval
    
    Returns:
        None
    """
    frame_counter = 0
    for frame_path in frame_files:
        frame_counter += 1
        if (frame_counter % (interval + 1) != 0):
            continue

        frame = cv2.imread(frame_path)
        frame_preprocessed = preprocess_frame(frame, background)
        cv2.imshow('frame_preprocessed', resize_frame(frame_preprocessed,
                                                      scale))

        tracking_success_flags = []
        tracked_object_boxes = []

        start_time = time.time()
        for tracker in object_trackers:
            # Update each tracker separately
            success, bbox = tracker.update(frame_preprocessed)
            tracked_object_boxes.append(bbox)
            tracking_success_flags.append(success)

        end_time = time.time()
        print(f"Tracking Time: {end_time - start_time}")

        # Remove failed trackers
        object_trackers[:] = [
            tracker for tracker, success in zip(object_trackers, tracking_success_flags)
            if success
        ]
        tracked_object_boxes[:] = [
            box for box, success in zip(tracked_object_boxes, tracking_success_flags) if success
        ]

        # preview tracking boxes
        for tracker, box in zip(object_trackers, tracked_object_boxes):
            (x, y, w, h) = [int(v) for v in box]
            # Skip tracking boxes with negative coordinates
            if x < 0 or y < 0:
                print(f"Skip invalid tracker {frame_counter}: coordinates ({x}, {y})")
                continue
            color = tracker_color_map.get(tracker, (0, 0, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        frame_resized = resize_frame(frame, scale)
        cv2.imshow('MultiTracker', frame_resized)

        key = cv2.waitKey(frame_delay)
        if key & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main(frame_path, background_path, bboxes_filename, fps, scale, interval,
         max_colors):
    """
    Main tracking function that orchestrates the entire tracking process.
    
    Args:
        frame_path (str): Path to frame files
        background_path (str): Path to background image
        bboxes_filename (str): Filename for saved bounding boxes
        fps (int): Frames per second
        scale (float): Scaling factor
        interval (int): Frame processing interval
        max_colors (int): Maximum number of distinct colors
    
    Returns:
        None
    """

    frame_delay = 1000 // fps
    bboxes_path = os.path.join(frame_path, bboxes_filename)

    frame_files = get_image_files(frame_path)
    background = cv2.imread(background_path)

    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print("Unable to read the first frame")
        return

    first_frame_preprocessed = preprocess_frame(first_frame, background)

    cv2.imshow('frame_preprocessed',
               resize_frame(first_frame_preprocessed, scale))

    bboxes = get_tracking_bboxes(first_frame, scale,
                                   bboxes_path)

    object_trackers, tracker_color_map = initialize_object_trackers(
        first_frame_preprocessed, bboxes, max_colors)

    # preview initial bboxes
    for tracker, box in zip(object_trackers, bboxes):
        (x, y, w, h) = [int(v) for v in box]
        color = tracker_color_map.get(tracker, (0, 0, 0))
        cv2.rectangle(first_frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow('MultiTracker', resize_frame(first_frame, scale))
    cv2.waitKey(0)  # Press any key to end preview

    # start tracking
    track_objects(frame_files[1:], background, object_trackers, tracker_color_map,
                  frame_delay, scale, interval)


if __name__ == "__main__":
    FRAME_PATH = "./data/20250117/60small_2"  # directory of image frames
    BACKGROUND_PATH = "./data/20250117/frame_id240.jpg"  # directory of background frames
    BBOXES_FILENAME = 'saved_bboxes1.json'  # *Important*: bboxes save file, if exists, use directly, no selectROI

    FPS = 1  # Set frame rate (frames per second), smaller values play slower
    SCALE = 0.5  # Resize to 50% of original size
    INTERVAL = 0  # Frame processing interval, 0 for no interval
    MAX_COLORS = 30  # Maximum number of distinct colors

    main(FRAME_PATH, BACKGROUND_PATH, BBOXES_FILENAME, FPS, SCALE, INTERVAL,
         MAX_COLORS)
