"""
CropEngine
Compute a smooth, 16:9 crop window that follows a bounding box.
Returns slice objects for fast numpy/OpenCV cropping.

Dependencies
------------
numpy, opencv-python (only for clamping utilities)
"""

from typing import Tuple
import numpy as np
import cv2


class CropEngine:
    def __init__(
        self,
        fps: float = 30.0,
        smooth_alpha: float = 0.05,
        max_velocity: float = 0.05,   # 5 % of frame width/height per frame
        margin_ratio: float = 0.20,   # extra margin around box
        aspect_ratio: str = "9:16"    # "9:16" for vertical, "4:3" for standard, "16:9" for wide
    ):
        self.alpha = smooth_alpha
        self.max_vel = max_velocity
        self.margin = margin_ratio
        
        # Parse aspect ratio
        if aspect_ratio == "9:16":
            self.aspect_w, self.aspect_h = 9, 16
        elif aspect_ratio == "4:3":
            self.aspect_w, self.aspect_h = 4, 3
        elif aspect_ratio == "16:9":
            self.aspect_w, self.aspect_h = 16, 9
        else:
            raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")

        # internal state
        self.cx: float = None
        self.cy: float = None
        self.scale: float = None  # pixels / crop_pixel

    # ------------------------------------------------------------------ #
    def update(
        self, box: Tuple[int, int, int, int], frame_shape: Tuple[int, int]
    ) -> Tuple[slice, slice]:
        """
        Parameters
        ----------
        box : (x1, y1, x2, y2) in pixel coordinates
        frame_shape : (H, W) of the original frame

        Returns
        -------
        yslice, xslice : slice objects for cropping the frame
        """
        x1, y1, x2, y2 = box
        H, W = frame_shape[:2]

        # 1. Target center & size with margin
        target_cx = (x1 + x2) * 0.5
        target_cy = (y1 + y2) * 0.5

        box_w = (x2 - x1) * (1 + self.margin)
        box_h = (y2 - y1) * (1 + self.margin)

        # 2. Calculate crop dimensions based on aspect ratio
        crop_w = max(box_w, box_h * self.aspect_w / self.aspect_h)
        crop_h = max(box_h, box_w * self.aspect_h / self.aspect_w)

        target_scale = min(W / crop_w, H / crop_h)  # >1 means zoom-in

        # 3. First frame → initialise
        if self.cx is None:
            self.cx, self.cy, self.scale = target_cx, target_cy, target_scale
        else:
            # Exponential smoothing + velocity clamp
            def clamp_step(curr, target):
                delta = np.clip(target - curr, -self.max_vel, self.max_vel)
                return curr + delta

            self.cx = self.alpha * target_cx + (1 - self.alpha) * self.cx
            self.cy = self.alpha * target_cy + (1 - self.alpha) * self.cy
            self.scale = self.alpha * target_scale + (1 - self.alpha) * self.scale

            # extra clamping against velocity
            self.cx = clamp_step(self.cx, target_cx)
            self.cy = clamp_step(self.cy, target_cy)
            self.scale = clamp_step(self.scale, target_scale)

        # 4. Compute final crop window
        crop_w_px = int(W / self.scale)
        crop_h_px = int(H / self.scale)

        x0 = int(np.clip(self.cx - crop_w_px // 2, 0, W - crop_w_px))
        y0 = int(np.clip(self.cy - crop_h_px // 2, 0, H - crop_h_px))

        return slice(y0, y0 + crop_h_px), slice(x0, x0 + crop_w_px)