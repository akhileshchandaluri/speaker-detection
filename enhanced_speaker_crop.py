#!/usr/bin/env python3
# enhanced_speaker_crop.py

import cv2
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional
import subprocess
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()
HF_TOKEN_ENV = os.getenv("HF_TOKEN")

from speaker_diarizer import SpeakerDiarizer
from kalman_tracker import MultiObjectTracker

import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_detection

# ------------------------- Face Detection -------------------------
class FaceDetector:
    def __init__(self, min_confidence=0.5):
        self.detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=min_confidence)

    def detect(self, frame) -> List[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.detector.process(rgb)
        boxes = []
        if res.detections:
            for d in res.detections:
                bbox = d.location_data.relative_bounding_box
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, int((bbox.xmin + bbox.width) * w))
                y2 = min(h, int((bbox.ymin + bbox.height) * h))
                boxes.append((x1, y1, x2, y2))
        return boxes

def find_active_box(boxes: List[Tuple[int, int, int, int]], frame_prev, frame_curr) -> Optional[Tuple[int, int, int, int]]:
    if not boxes:
        return None
    if frame_prev is None:
        return boxes[0]
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
    best_idx = None
    best_score = -1.0
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(gray_curr.shape[1], x2), min(gray_curr.shape[0], y2)
        if x2c <= x1c or y2c <= y1c:
            score = 0.0
        else:
            prev_roi = gray_prev[y1c:y2c, x1c:x2c]
            curr_roi = gray_curr[y1c:y2c, x1c:x2c]
            if prev_roi.shape == curr_roi.shape and prev_roi.size > 0:
                diff = cv2.absdiff(prev_roi, curr_roi)
                score = float(diff.mean()) / 255.0
            else:
                score = 0.0
        if score > best_score:
            best_score = score
            best_idx = i
    if best_score > 0.02:
        return boxes[best_idx]
    return boxes[0] if boxes else None

def timestamp_to_sec(frame_idx: int, fps: float) -> float:
    return frame_idx / fps

# ------------------------- Main Processing -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video", help="Input video path")
    parser.add_argument("output_video", help="Output video path")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Sampling fps (frames/sec)")
    parser.add_argument("--token", dest="hf_token", help="HF token (optional for gated models)")
    parser.add_argument("--map", dest="speaker_map", help="Manual speaker mapping (e.g., 'SPEAKER_00:left,SPEAKER_01:right')")
    args = parser.parse_args()

    input_path = Path(args.input_video)
    output_path = Path(args.output_video)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    # Use CLI token if provided, else fallback to .env
    HF_TOKEN = args.hf_token or HF_TOKEN_ENV
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN is not set in CLI or .env file")

    print("[INFO] Loading diarizer...")
    diar = SpeakerDiarizer(hf_token=HF_TOKEN)
    print("[INFO] Running diarization (this extracts audio with ffmpeg)...")
    segments = diar.diarize(str(input_path))
    print(f"[INFO] Diarization returned {len(segments)} segments")

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sample_fps = args.sample_fps
    sample_interval_frames = max(1, int(round(fps / sample_fps)))

    print(f"[INFO] Video fps: {fps}, sample_fps: {sample_fps}, interval frames: {sample_interval_frames}")

    face_detector = FaceDetector(min_confidence=0.3)
    
    # Check if manual mapping is provided
    manual_mapping = {}
    if args.speaker_map:
        print(f"[INFO] Using MANUAL speaker mapping: {args.speaker_map}")
        for pair in args.speaker_map.split(','):
            speaker, position = pair.split(':')
            manual_mapping[speaker.strip()] = 0 if position.strip().lower() == 'left' else 1
        print(f"[INFO] Manual mapping parsed: {manual_mapping}")
    
    # Build speaker-to-face mapping by analyzing multiple samples per speaker
    print("[INFO] Building speaker-to-face mapping from diarization...")
    speaker_positions_samples = {}  # Collect all position samples per speaker
    
    cap_analysis = cv2.VideoCapture(str(input_path))
    for start_time, end_time, speaker_label in segments:
        if speaker_label not in speaker_positions_samples:
            speaker_positions_samples[speaker_label] = []
        
        # Sample 3 points: start, middle, end of segment
        sample_times = [start_time + 0.5, (start_time + end_time) / 2, end_time - 0.5]
        for sample_time in sample_times:
            if sample_time < start_time or sample_time > end_time:
                continue
            frame_idx_sample = int(sample_time * fps)
            cap_analysis.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_sample)
            ret, frame = cap_analysis.read()
            if ret:
                boxes = face_detector.detect(frame)
                if len(boxes) == 2:
                    # Sort boxes left to right by x-center
                    sorted_boxes = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2)
                    left_center = (sorted_boxes[0][0] + sorted_boxes[0][2]) / 2
                    right_center = (sorted_boxes[1][0] + sorted_boxes[1][2]) / 2
                    # Store BOTH positions with labels
                    speaker_positions_samples[speaker_label].append({
                        'left': left_center,
                        'right': right_center,
                        'time': sample_time
                    })
    
    cap_analysis.release()
    
    # Use manual mapping if provided, otherwise auto-detect
    if manual_mapping:
        speaker_to_position_idx = manual_mapping
        print(f"[INFO] Using MANUAL speaker-to-position mapping: {speaker_to_position_idx}")
    else:
        # Determine position for each speaker using motion/activity analysis
        # Strategy: Track which face position has more consistent presence during speaker's segments
        speaker_to_position_idx = {}
        
        for speaker, samples in speaker_positions_samples.items():
            if not samples:
                continue
            
            # Calculate variance of left and right positions
            # The speaker's face should have LESS variance (more stable position)
            left_positions = [s['left'] for s in samples]
            right_positions = [s['right'] for s in samples]
            
            if len(left_positions) > 1:
                import statistics
                left_var = statistics.variance(left_positions)
                right_var = statistics.variance(right_positions)
                
                # Also check median position - speaker on left will have lower x-coords
                left_median = statistics.median(left_positions)
                right_median = statistics.median(right_positions)
                
                # Decision: Use median position as primary indicator
                # If this speaker's segments show left face more consistently, assign to left (0)
                avg_left = sum(left_positions) / len(left_positions)
                avg_right = sum(right_positions) / len(right_positions)
                
                # Simple heuristic: alternate speakers between left and right
                # First speaker detected -> assign based on variance
                if len(speaker_to_position_idx) == 0:
                    # First speaker: assign to side with lower variance (more stable = likely speaker)
                    speaker_to_position_idx[speaker] = 0 if left_var < right_var else 1
                else:
                    # Subsequent speakers: assign to opposite side
                    existing_positions = set(speaker_to_position_idx.values())
                    if 0 not in existing_positions:
                        speaker_to_position_idx[speaker] = 0
                    elif 1 not in existing_positions:
                        speaker_to_position_idx[speaker] = 1
                    else:
                        # Both sides taken, use variance
                        speaker_to_position_idx[speaker] = 0 if left_var < right_var else 1
            else:
                # Fallback: assign alternately
                speaker_to_position_idx[speaker] = len(speaker_to_position_idx) % 2
        
        print(f"[INFO] AUTO-DETECTED speaker-to-position mapping: {speaker_to_position_idx}")
    
    # Initialize TWO Kalman trackers: one for left person (idx=0), one for right person (idx=1)
    # Increased max_age for better long-term tracking, min_hits=1 for immediate response
    tracker_left = MultiObjectTracker(max_age=60, min_hits=1)
    tracker_right = MultiObjectTracker(max_age=60, min_hits=1)
    
    # Maps to store tracked boxes for both people across ALL frames
    left_boxes = {}   # frame_idx -> box
    right_boxes = {}  # frame_idx -> box
    
    # ------------------------- First Pass: Detect and track BOTH faces continuously -------------------------
    frame_idx = 0
    print("[INFO] First pass: tracking both faces with Kalman filter...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces on EVERY frame for maximum smoothness (not just sampled)
        boxes = face_detector.detect(frame)
        
        if len(boxes) >= 2:
            # Sort left to right
            sorted_boxes = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2)
            # Update both trackers with actual detections
            tracker_left.update([sorted_boxes[0]])
            tracker_right.update([sorted_boxes[1]])
            
            # Store detected boxes
            left_boxes[frame_idx] = sorted_boxes[0]
            right_boxes[frame_idx] = sorted_boxes[1]
        elif len(boxes) == 1:
            # Only one face detected - determine which side
            box_center = (boxes[0][0] + boxes[0][2]) / 2
            if box_center < width / 2:
                tracker_left.update(boxes)
                left_boxes[frame_idx] = boxes[0]
                # Predict for right
                pred_right = tracker_right.predict_only()
                if pred_right:
                    right_boxes[frame_idx] = pred_right[0]
            else:
                tracker_right.update(boxes)
                right_boxes[frame_idx] = boxes[0]
                # Predict for left
                pred_left = tracker_left.predict_only()
                if pred_left:
                    left_boxes[frame_idx] = pred_left[0]
        else:
            # No faces detected - use Kalman prediction for both
            pred_left = tracker_left.predict_only()
            pred_right = tracker_right.predict_only()
            
            if pred_left:
                left_boxes[frame_idx] = pred_left[0]
            if pred_right:
                right_boxes[frame_idx] = pred_right[0]
        
        frame_idx += 1
    
    cap.release()
    print(f"[INFO] Tracked {len(left_boxes)} frames for left person, {len(right_boxes)} frames for right person.")

    # ------------------------- Second Pass: Select active speaker box per frame -------------------------
    tmp_video = output_path.with_suffix(".noaudio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (width, height))

    cap = cv2.VideoCapture(str(input_path))
    frame_idx = 0
    print("[INFO] Second pass: writing output with active speaker boxes...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_to_write = frame.copy()
        
        # Determine ALL active speakers at this timestamp (support overlapping speech)
        t_sec = timestamp_to_sec(frame_idx, fps)
        active_speakers = []
        for s, e, label in segments:
            if s <= t_sec <= e:
                active_speakers.append(label)
        
        # Draw boxes for ALL active speakers (supports simultaneous speech)
        boxes_to_draw = []
        for speaker in active_speakers:
            if speaker in speaker_to_position_idx:
                pos_idx = speaker_to_position_idx[speaker]
                box = None
                if pos_idx == 0 and frame_idx in left_boxes:
                    box = left_boxes[frame_idx]
                elif pos_idx == 1 and frame_idx in right_boxes:
                    box = right_boxes[frame_idx]
                
                if box is not None:
                    boxes_to_draw.append((box, speaker))
        
        # Draw all valid bounding boxes
        for box, speaker_label in boxes_to_draw:
            x1, y1, x2, y2 = box
            
            # Validate box is reasonable (not empty, not out of bounds)
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Skip invalid boxes (too small, negative size, or completely out of frame)
            if box_width > 10 and box_height > 10 and x2 > 0 and y2 > 0 and x1 < width and y1 < height:
                # Clamp to frame bounds
                x1 = max(0, min(int(x1), width - 1))
                y1 = max(0, min(int(y1), height - 1))
                x2 = max(x1 + 1, min(int(x2), width))
                y2 = max(y1 + 1, min(int(y2), height))
                
                # Draw green bounding box
                cv2.rectangle(frame_to_write, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Add speaker label
                label_text = f"Speaking: {speaker_label}"
                cv2.putText(frame_to_write, label_text, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        out_writer.write(frame_to_write)
        frame_idx += 1

    cap.release()
    out_writer.release()
    print(f"[INFO] Temporary video written to {tmp_video}")

    # ------------------------- Remux Audio -------------------------
    final_tmp = output_path.with_suffix(".final.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(tmp_video),
        "-i", str(input_path),
        "-c", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(final_tmp)
    ]
    print("[INFO] Muxing original audio back into processed video (ffmpeg)...")
    subprocess.run(cmd, check=True)
    final_tmp.replace(output_path)

    try:
        tmp_video.unlink()
    except Exception:
        pass

    print(f"[INFO] Output saved to: {output_path}")
    print("[DONE]")

if __name__ == "__main__":
    main()
