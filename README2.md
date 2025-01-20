# Channel and Spatial Reliability Tracker (CSRT) 2.0

## Overview
`trackerCSRT2.0.py` is an advanced object tracking script using OpenCV's Channel and Spatial Reliability Tracker (CSRT). This script provides a robust multi-object tracking solution with preprocessing, interactive object selection, and visualization capabilities.

## Dependencies (Reference)
- opencv-contrib-python    4.10.0.84
- numpy                    1.24.4

## Tracking Workflow with Parameters
1. Load frames from `FRAME_PATH`
2. Use `BACKGROUND_PATH` for preprocessing
3. Load or create bounding boxes using `BBOXES_FILENAME`
4. Track objects across the image sequence

## Configuration Parameters

### `FRAME_PATH`
- **Path**: `"./data/"`
- **Description**: Directory containing the sequence of image frames to be processed
- **Purpose**: Provides the input image sequence for multi-object tracking


### `BACKGROUND_PATH`
- **Path**: `"./background.jpg"`
- **Description**: Path to the background image used for frame preprocessing
- **Purpose**: Enables background subtraction technique

### `BBOXES_FILENAME`
- **Filename**: `'saved_bboxes.json'`
- **Description**: JSON file for storing and loading object bounding boxes
- **Workflow**:
  1. First run: Manually select objects (ROI)
  2. Bounding boxes automatically saved to `saved_bboxes.json`
  3. Subsequent runs use saved bounding boxes without manual selection

### `FPS`
- **Value**: `1`
- **Description**: Controls the frame rate of video playback

### `SCALE`
- **Value**: `0.5`
- **Description**: Image resizing parameter *only for display*

### `INTERVAL`
- **Value**: `0`
- **Description**: Frame processing interval control. `0`: Process every single frame

### `MAX_COLORS`
- **Value**: `30`
- **Description**: Maximum number of distinct colors for boudning boxes

## Other Algorithm Tests
- `calcOpticalFlowFarneback.py`
- `calcOpticalFlowPyrLK.py`
- `correlation_track.py`
- `trackerCSRT.py`

**Caution**: These files are incomplete and unrefined research prototypes. They are not production-ready and may contain experimental or unoptimized code.
