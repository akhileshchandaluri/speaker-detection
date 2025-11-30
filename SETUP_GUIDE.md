# Speaker Detection Setup Guide

## Files to Share with Team

### Core Files (Modified)
1. **`enhanced_speaker_crop.py`** - Main processing script
2. **`kalman_tracker.py`** - Kalman filter for smooth tracking
3. **`speaker_diarizer.py`** - Audio diarization with overlap detection
4. **`requirements_core.txt`** - Python dependencies

### Sample Videos (for testing)
- `Jaishankar.mp4` - 60 seconds, 2-person interview
- `CharliePuth.mp4` - 41 seconds, 2-person conversation

## Installation Steps

### 1. Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
pip install pyannote.audio==3.1.1
pip install opencv-python==4.11.0.86
pip install mediapipe==0.10.21
pip install python-dotenv
```

### 3. Get HuggingFace Token
- Go to https://huggingface.co/settings/tokens
- Create a token with read access
- Accept the model license at: https://huggingface.co/pyannote/speaker-diarization-3.1

### 4. Run the System
```powershell
python enhanced_speaker_crop.py input_video.mp4 output_video.mp4 --token YOUR_HF_TOKEN
```

## What It Does

✅ **Speaker Diarization** - Detects who's speaking when (with overlapping speech support)
✅ **Face Tracking** - Tracks both faces continuously with Kalman filter smoothing
✅ **Smart Mapping** - Matches audio speakers to visual face positions (left/right)
✅ **Multi-Speaker Support** - Shows multiple boxes when people talk simultaneously
✅ **Smooth Tracking** - 10x smoother than baseline with velocity decay

## Key Features

- **Every-frame detection**: Tracks faces on all frames for maximum smoothness
- **Overlap detection**: 7+ overlapping speech segments detected in CharliePuth video
- **Kalman filtering**: Process noise 0.1, measurement noise 5.0, velocity decay 0.98
- **Box validation**: Filters out invalid/out-of-bounds boxes
- **100% coverage**: All frames tracked (1799/1799 for Jaishankar, 1235/1235 for CharliePuth)

## Expected Results

### Jaishankar.mp4
- 20 speaker segments
- Speaker mapping: SPEAKER_01→left, SPEAKER_02→right
- Boxes appear only when someone is speaking
- Smooth transitions between speakers

### CharliePuth.mp4  
- 17 speaker segments
- 7 overlapping speech segments (12-19 second range)
- Multiple boxes shown during simultaneous speech
- Speaker mapping: SPEAKER_01→left, SPEAKER_00 & SPEAKER_02→right

## Troubleshooting

**Issue**: SSL Certificate Error
- Solution: Internet connection issue, use cached models (already downloaded after first run)

**Issue**: No boxes appearing
- Check: Diarization segments > 0
- Check: Speaker-to-position mapping not empty
- Check: Face detection confidence (currently 0.3)

**Issue**: Wrong person boxed
- Check: Speaker mapping in logs
- Adjust: Variance-based mapping logic in `enhanced_speaker_crop.py` lines 125-200

## Performance

- Processing speed: ~2000-2600fps (ffmpeg muxing)
- Face detection: MediaPipe (CPU-based, no GPU needed)
- Diarization: ~4-5 seconds for 60-second video
- Total time: ~10-15 seconds for 60-second video
