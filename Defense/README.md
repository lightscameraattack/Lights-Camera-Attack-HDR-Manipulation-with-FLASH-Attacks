# Online Flash Detector

This script provides a tool to detect flashes in video streams. It uses the Median Absolute Deviation (MAD) method to find frames with sudden brightness changes.

## How It Works

The function `online_flash_detector` reads a video file and finds flashes using these steps:

1.  **Brightness Calculation**: It converts each frame to grayscale to find the average brightness.
2.  **History Tracking**: It keeps a list of brightness values from recent frames. The default length is 30 frames.
3.  **Detection**:
    *   It calculates the Median and Median Absolute Deviation (MAD) from the history list.
    *   It compares the difference between the current frame's brightness and the median value against a limit.
    *   The limit is set as `threshold_multiplier` times the MAD value. The default multiplier is 8.0.
4.  **Update**: It adds the current frame's brightness to the history list only if it is not a flash. This keeps the history data accurate for future frames.

## Usage

```python
from online_flash_detector import online_flash_detector

# Run detection on a video file
stats = online_flash_detector(
    video_path='path/to/video.mp4',
    history_len=30,
    threshold_multiplier=8.0
)
```

## Returns

The function returns a dictionary with the following data:
*   `total_frames`: The total number of processed frames.
*   `flagged_indices`: A list of frame numbers identified as flashes.
*   `avg_latency_ms`: The average time taken to process each frame in milliseconds.

## Performance Statistics

The script prints the following statistics to the console when it finishes:
*   Total Frames
*   Flagged Frames (Dropped)
*   Drop Rate (Percent)
*   Average Detection Latency (milliseconds per frame)
