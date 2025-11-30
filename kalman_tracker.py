"""
KalmanTracker: Smooth bounding box extrapolation using Kalman Filter

How it works:
1. Predict: Estimate next box position based on velocity
2. Measure: Get actual detection (if available)
3. Correct: Update position and velocity estimates
4. Output: Use prediction for missing frames

This reduces jitter and interpolates smoothly between detections.
"""

import numpy as np
from typing import Tuple, Optional, List


class KalmanTracker:
    """
    Single object Kalman Filter for bounding box tracking.
    Tracks a 2D position (center x, y) and velocity (vx, vy).
    
    State vector: [x, y, vx, vy]
    - x, y: center of bounding box
    - vx, vy: velocity in x, y directions
    """
    
    def __init__(self, initial_box: Tuple[int, int, int, int], tracker_id: int = -1, process_noise: float = 0.1, measurement_noise: float = 5.0):
        """
        Parameters
        ----------
        initial_box : (x1, y1, x2, y2)
            Initial bounding box coordinates
        tracker_id : int
            Unique ID for this tracker
        process_noise : float
            How much we trust the motion model (higher = less trust in prediction)
        measurement_noise : float
            How much we trust measurements (higher = less trust in detection)
        """
        self.id = tracker_id
        x1, y1, x2, y2 = initial_box
        
        # State: [x, y, vx, vy] (center position and velocity)
        self.x = np.array([
            (x1 + x2) / 2.0,  # center x
            (y1 + y2) / 2.0,  # center y
            0.0,              # velocity x (initially 0)
            0.0               # velocity y (initially 0)
        ], dtype=np.float32)
        
        # Store box dimensions
        self.width = x2 - x1
        self.height = y2 - y1
        
        # Track when this tracker was last updated
        self.frame_count = 0
        self.frames_since_update = 0
        
        # State covariance matrix (uncertainty in our estimates)
        self.P = np.eye(4, dtype=np.float32) * 50.0
        
        # Process noise covariance (model uncertainty) - LOWER for smoother tracking
        self.Q = np.array([
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, process_noise, 0.0],
            [0.0, 0.0, 0.0, process_noise]
        ], dtype=np.float32)
        
        # Measurement noise covariance (detection uncertainty) - LOWER to trust detections more
        self.R = np.array([
            [measurement_noise, 0.0],
            [0.0, measurement_noise]
        ], dtype=np.float32)
        
        # Measurement matrix (we only measure x, y, not velocity)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        self.dt = 1.0  # Time step (1 frame)
    
    def predict(self) -> Tuple[int, int, int, int]:
        """
        Predict next state WITHOUT measurement.
        Used for frames where we don't have a detection.
        
        Returns
        -------
        box : (x1, y1, x2, y2) predicted bounding box
        """
        self.frames_since_update += 1
        
        # State transition: next position = current position + velocity
        # Motion model: x_{k+1} = F * x_k
        F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 0.98, 0.0],  # Velocity decay for stability
            [0.0, 0.0, 0.0, 0.98]
        ], dtype=np.float32)
        
        # Predict state
        self.x = F @ self.x
        
        # Predict covariance: P_{k|k-1} = F * P_{k-1} * F^T + Q
        self.P = F @ self.P @ F.T + self.Q
        
        # Convert center + size back to box coordinates
        cx, cy = self.x[0], self.x[1]
        x1 = int(cx - self.width / 2.0)
        y1 = int(cy - self.height / 2.0)
        x2 = int(cx + self.width / 2.0)
        y2 = int(cy + self.height / 2.0)
        
        return (x1, y1, x2, y2)
    
    def update(self, measurement_box: Tuple[int, int, int, int]):
        """
        Update state WITH measurement (actual detection).
        Used for frames where we have a face detection.
        
        Parameters
        ----------
        measurement_box : (x1, y1, x2, y2)
            Detected bounding box
        """
        x1, y1, x2, y2 = measurement_box
        
        # Extract measurement (center position)
        z = np.array([
            (x1 + x2) / 2.0,  # measured center x
            (y1 + y2) / 2.0   # measured center y
        ], dtype=np.float32)
        
        # Update box dimensions
        self.width = x2 - x1
        self.height = y2 - y1
        
        # Kalman gain: K = P * H^T / (H * P * H^T + R)
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        # Innovation (residual): y = z - H * x
        y = z - self.H @ self.x
        
        # Update state: x = x + K * y
        self.x = self.x + K @ y
        
        # Update covariance: P = (I - K * H) * P
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        # Track update time
        self.frame_count = -1  # Signal that this tracker was just updated
        self.frames_since_update = 0  # Reset counter
    
    def get_box(self) -> Tuple[int, int, int, int]:
        """Get current estimated box."""
        cx, cy = self.x[0], self.x[1]
        x1 = int(cx - self.width / 2.0)
        y1 = int(cy - self.height / 2.0)
        x2 = int(cx + self.width / 2.0)
        y2 = int(cy + self.height / 2.0)
        return (x1, y1, x2, y2)


class MultiObjectTracker:
    """
    Track multiple objects (faces) across frames using Kalman Filters.
    Matches detections to existing trackers and creates new ones as needed.
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3):
        """
        Parameters
        ----------
        max_age : int
            Maximum frames to keep a tracker alive without detections
        min_hits : int
            Minimum detections before tracker is considered "confirmed"
        """
        self.trackers: List[KalmanTracker] = []
        self.max_age = max_age
        self.frame_count = 0
        self.next_id = 0
    
    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int, int]]:
        """
        Update all trackers with new detections and return interpolated boxes for all frames.
        
        Parameters
        ----------
        detections : list of (x1, y1, x2, y2)
            Detected bounding boxes in current frame
        
        Returns
        -------
        boxes : list of (x1, y1, x2, y2, id)
            Interpolated boxes for all tracked objects with their IDs
        """
        self.frame_count += 1
        
        # Predict step for all trackers
        for tracker in self.trackers:
            tracker.predict()
        
        # Match detections to trackers using IoU (Intersection over Union)
        if len(detections) > 0 and len(self.trackers) > 0:
            matched_pairs = self._match_detections(detections)
            
            # Update matched trackers
            matched_tracker_indices = set()
            for det_idx, trk_idx in matched_pairs:
                self.trackers[trk_idx].update(detections[det_idx])
                self.trackers[trk_idx].frame_count = self.frame_count
                matched_tracker_indices.add(trk_idx)
            
            # Create new trackers for unmatched detections
            for det_idx, det_box in enumerate(detections):
                if det_idx not in [p[0] for p in matched_pairs]:
                    new_tracker = KalmanTracker(det_box, self.next_id)
                    self.next_id += 1
                    new_tracker.frame_count = self.frame_count
                    self.trackers.append(new_tracker)
        else:
            # No detections: create trackers for all detections
            for det_box in detections:
                new_tracker = KalmanTracker(det_box, self.next_id)
                self.next_id += 1
                new_tracker.frame_count = self.frame_count
                self.trackers.append(new_tracker)
        
        # Remove dead trackers (not updated in last max_age frames)
        self.trackers = [t for t in self.trackers if self.frame_count - t.frame_count < self.max_age]
        
        # Return boxes from all active trackers
        return [(*t.get_box(), t.id) for t in self.trackers]
    
    def predict_only(self) -> List[Tuple[int, int, int, int]]:
        """
        Predict boxes for all trackers WITHOUT updating with measurements.
        Used for frames between detections.
        
        Returns
        -------
        boxes : list of (x1, y1, x2, y2)
            Predicted boxes for all tracked objects
        """
        boxes = []
        for tracker in self.trackers:
            boxes.append(tracker.predict())
        return boxes
    
    @staticmethod
    def _iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _match_detections(self, detections: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
        """
        Match detections to existing trackers using IoU.
        
        Returns
        -------
        matches : list of (detection_idx, tracker_idx) pairs
        """
        matches = []
        iou_threshold = 0.3  # Lower threshold = better matching of same objects across frames
        
        for det_idx, det_box in enumerate(detections):
            best_iou = 0
            best_trk_idx = -1
            
            for trk_idx, tracker in enumerate(self.trackers):
                current_box = tracker.get_box()
                iou = self._iou(det_box, current_box)
                
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_trk_idx = trk_idx
            
            if best_trk_idx != -1:
                matches.append((det_idx, best_trk_idx))
        
        return matches
