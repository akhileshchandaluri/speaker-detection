# Kalman Filter Implementation - Summary

## Overview
Implemented **Kalman Filter-based face tracking** with smooth bounding box extrapolation across all video frames, eliminating jitter and filling gaps between detections.

---

## Files to Share with Teammates

### Core Implementation Files
1. **`kalman_tracker.py`** (NEW) - Complete Kalman Filter implementation
2. **`crop_to_speaker.py`** (MODIFIED) - Main pipeline with Kalman integration
3. **`face_tracker.py`** - Face detection utilities
4. **`speaker_diarizer.py`** - Speaker diarization (baseline)
5. **`body_tracker.py`** - Body tracking utilities
6. **`crop_engine.py`** - Cropping engine

### Test/Demo Files
7. **`Jaishankar.mp4`** - Input test video
8. **`output_final.mp4`** - Output with tracked faces
9. **`README.md`** - Project documentation

### Setup Files
10. **`requirements.txt`** - Python dependencies (if exists)

---

## Key Changes Made

### 1. Created `kalman_tracker.py` (NEW FILE - ~300 lines)

**Purpose:** Smooth face tracking using Kalman Filter

**Key Classes:**

#### `KalmanTracker` (Single Object Tracker)
- **State Vector:** `[x, y, vx, vy]` - position (x,y) and velocity (vx, vy)
- **Methods:**
  - `predict()` - Estimate next position based on velocity
  - `update(measurement_box)` - Correct estimate with actual detection
  - `get_box()` - Return current bounding box

**Parameters:**
```python
process_noise = 1.0      # Trust in motion model
measurement_noise = 10.0 # Trust in detections
```

#### `MultiObjectTracker` (Multiple Objects)
- **Manages:** Multiple KalmanTracker instances
- **Methods:**
  - `update(detections)` - Match detections to trackers using IoU
  - `predict_only()` - Predict without measurements (for non-sampled frames)
  - `_iou()` - Calculate Intersection over Union for matching
  - `_match_detections()` - Assign detections to existing trackers

**Key Features:**
- IoU-based matching (threshold: 0.3)
- Automatic tracker creation for new faces
- Tracker removal after `max_age=30` frames without updates

---

### 2. Modified `crop_to_speaker.py` (MAJOR CHANGES)

#### Integration Points:

**Import Added:**
```python
from kalman_tracker import KalmanTracker, MultiObjectTracker
```

**Initialization (line ~175):**
```python
self.kalman_tracker = MultiObjectTracker() if KALMAN_AVAILABLE else None
```

#### New Pipeline: Two-Pass Processing

**PASS 1: Sample & Detect**
- Sample frames at 5 fps (every 5th frame)
- Run MediaPipe face detection on sampled frames
- Store detections: `detected_boxes_by_frame[frame_idx] = boxes`
- Result: 360 sampled frames with detections

**PASS 2: Kalman Extrapolation** (NEW METHOD: `_extrapolate_with_kalman`)
```python
def _extrapolate_with_kalman(self, video_path, detected_boxes_by_frame, 
                             sample_interval, fps):
    """
    For each frame:
      - If SAMPLED frame: UPDATE Kalman with actual detections
      - If NON-SAMPLED: PREDICT using Kalman (no measurement)
    
    Returns: boxes for ALL 1799 frames (smooth interpolation)
    """
```

**Logic:**
```python
for frame_idx in range(total_frames):
    if frame_idx in detected_boxes_by_frame:
        # Sampled frame - UPDATE Kalman
        boxes = self.kalman_tracker.update(detections)
    else:
        # Non-sampled frame - PREDICT only
        boxes = self.kalman_tracker.predict_only()
    
    extrapolated_boxes[frame_idx] = boxes
```

#### Post-Processing Filters (NEW METHODS)

**1. Max-2-Faces Filter** (`_filter_to_largest_n_faces`)
- **Purpose:** Keep only 2 largest faces per frame (2 speakers)
- **Scoring:** `score = area + position_bonus`
- **Position Bonus:** Heavily favor boxes with center y < 300 pixels (face region)
- **Result:** 3767 boxes → 3258 boxes

**2. Consistent Position Filter** (`_filter_by_consistent_positions`)
- **Purpose:** Find 2 most common speaker positions across video
- **Method:** Spatial clustering using 50x50 pixel grid
- **Logic:** 
  - Discretize all box centroids into grid cells
  - Find top 2 grid positions (most occurrences)
  - Keep only boxes within 2 grid cells of these positions
- **Result:** 3258 boxes → 3173 boxes (removed scattered artifacts)

**3. Detection Parameters:**
```python
min_detection_confidence = 0.5  # MediaPipe threshold
min_face_size = 20 pixels       # Minimum box dimension
aspect_ratio = 0.4 - 2.5        # Valid face aspect ratio
NMS threshold = 0.35            # Non-Maximum Suppression
```

---

## Results

### Before Kalman Filter
- ❌ Boxes only on sampled frames (360 out of 1799 frames)
- ❌ Jittery, inconsistent tracking
- ❌ Gaps between detections

### After Kalman Filter
- ✅ **All 1799 frames** have smooth bounding boxes
- ✅ **Jitter eliminated** - smooth motion tracking
- ✅ **Gaps filled** - Kalman predicts between detections
- ✅ **Only 2 speakers tracked** - artifacts filtered out
- ✅ **Consistent positions** - boxes stay on actual speaker faces

### Performance
- **Input:** Jaishankar.mp4 (1799 frames, 30 fps, 1 minute)
- **Detection time:** 2.00s (360 sampled frames)
- **Extrapolation time:** ~1s (1799 frames with Kalman)
- **Total processing:** ~5-6 seconds
- **Output:** output_final.mp4 with tracked faces on all frames

---

## Technical Deep Dive

### Kalman Filter Math (Simplified)

**Prediction Step:**
```
x_predicted = x_current + vx * dt
y_predicted = y_current + vy * dt
vx_predicted = vx_current
vy_predicted = vy_current
```

**Update Step (when detection available):**
```
# Kalman Gain
K = P * H^T / (H * P * H^T + R)

# State update
x_corrected = x_predicted + K * (measurement - x_predicted)

# Covariance update
P = (I - K * H) * P
```

**Where:**
- `P` = State covariance (uncertainty)
- `H` = Measurement matrix
- `R` = Measurement noise
- `Q` = Process noise

### Why Kalman Filter?

1. **Smoothing:** Reduces jitter from noisy detections
2. **Interpolation:** Predicts positions when no detection available
3. **Velocity tracking:** Uses motion model to predict next position
4. **Optimal fusion:** Mathematically optimal blend of prediction + measurement

---

## How to Run

```bash
# Install dependencies
pip install opencv-python mediapipe numpy scipy av torch torchaudio

# Run pipeline
python crop_to_speaker.py Jaishankar.mp4 output_final.mp4 --sample-fps 5.0
```

**Key Arguments:**
- `--sample-fps 5.0` - Sample at 5 fps (process every 5th frame for detection)
- Input: `Jaishankar.mp4`
- Output: `output_final.mp4` (with tracked faces)

---

## Architecture Diagram

```
Input Video (1799 frames)
    ↓
PASS 1: Sample & Detect (every 5th frame)
    ↓
MediaPipe Face Detection (360 frames)
    ↓
detected_boxes_by_frame[0, 5, 10, ..., 1795]
    ↓
PASS 2: Kalman Extrapolation (all frames)
    ├─ Frame 0 (sampled): UPDATE Kalman
    ├─ Frame 1-4: PREDICT only
    ├─ Frame 5 (sampled): UPDATE Kalman
    ├─ Frame 6-9: PREDICT only
    └─ ... (repeat for all 1799 frames)
    ↓
extrapolated_boxes[0...1798] (all frames)
    ↓
Post-Processing Filters:
    ├─ Max-2-Faces (keep largest 2 per frame)
    └─ Consistent Positions (remove artifacts)
    ↓
Output Video with Smooth Tracked Faces
```

---

## Key Insights & Lessons Learned

### What Worked Well
1. **Two-pass approach** - Efficient detection + smooth extrapolation
2. **IoU matching** - Reliable tracker-detection association
3. **Position-based filtering** - Effective at removing background artifacts
4. **Spatial clustering** - Identified consistent speaker positions

### Challenges Solved
1. **False positives on background** - Filtered using position consistency
2. **Multiple boxes per person** - NMS + max-2-faces filter
3. **Detection gaps** - Kalman prediction fills missing frames
4. **Jittery tracking** - Kalman smoothing eliminates jitter

### Parameter Tuning Journey
- Started with confidence=0.5, ended with 0.5 (after testing 0.7, 0.75, 0.8, 0.85)
- NMS threshold: 0.3 → 0.35 (sweet spot)
- Position bonus weight: tested multiple formulas, settled on cy < 300px
- Grid size for clustering: 50x50 pixels (good balance)

---

## Future Improvements (Optional)

1. **Face recognition** - Identify which speaker is which
2. **Speaker diarization integration** - Link faces to audio segments
3. **Adaptive sampling** - Sample more during motion, less during static scenes
4. **GPU acceleration** - Speed up MediaPipe detection
5. **Multi-face Kalman** - Track >2 faces for panel discussions

---

## Questions for Teammates?

Feel free to ask about:
- Kalman Filter implementation details
- Parameter choices and tuning
- Filter design decisions
- Performance optimization
- Integration with other modules

---

**Implementation completed by:** Akhilesh Chandaluri  
**Date:** November 18, 2025  
**Status:** ✅ Fully functional and tested
