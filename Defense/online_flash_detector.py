import cv2
import numpy as np
import time
from collections import deque

def online_flash_detector(video_path, history_len=30, threshold_multiplier=8.0):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: I could not open video {video_path}")
        return

    luminance_history = deque(maxlen=history_len)
    
    frame_count = 0
    flagged_frames = []
    latencies = []
    
    print(f"I am processing {video_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.perf_counter()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        current_intensity = np.mean(gray)
        
        is_anomaly = False
        
        if len(luminance_history) >= 5:
            history_arr = np.array(luminance_history)
            
            median_val = np.median(history_arr)
            
            abs_dev = np.abs(history_arr - median_val)
            mad_val = np.median(abs_dev)
            
            if mad_val < 1e-5:
                mad_val = 1.0 
            
            deviation = np.abs(current_intensity - median_val)
            
            if deviation > (threshold_multiplier * mad_val):
                is_anomaly = True
                flagged_frames.append(frame_count)
        
        if not is_anomaly:
            luminance_history.append(current_intensity)
            
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)
        
        frame_count += 1

    cap.release()
    
    avg_latency = np.mean(latencies)
    drop_rate = (len(flagged_frames) / frame_count) * 100 if frame_count > 0 else 0
    
    print("-" * 30)
    print(f"My Total Frames: {frame_count}")
    print(f"Frames I Flagged (Dropped): {len(flagged_frames)}")
    print(f"My Drop Rate: {drop_rate:.2f}%")
    print(f"My Avg Detection Latency: {avg_latency:.4f} ms per frame")
    print("-" * 30)
    
    return {
        "total_frames": frame_count,
        "flagged_indices": flagged_frames,
        "avg_latency_ms": avg_latency
    }
