"""
Mini Metro AI Agent — Line Detector (Layer 2B)

Detects metro lines by their distinct colors using HSV masking,
traces their paths via skeletonization, and determines which
stations are connected by each line.

Usage:
    python -m vision.line_detector    # Live colored skeleton overlay
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

import config
from vision.station_detector import DetectedStation


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class DetectedLine:
    """A metro line detected by color."""
    color: str                           # e.g., "red", "blue"
    station_ids: List[int]               # IDs of connected stations (ordered along path)
    path_points: List[Tuple[int, int]]   # skeleton pixel coordinates
    pixel_count: int                     # total colored pixels (line thickness indicator)
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # binary mask for this line
    endpoints: List[Tuple[int, int]] = field(default_factory=list) # geometric endpoints (T-caps)


# =============================================================================
# Color name → BGR for drawing
# =============================================================================

COLOR_BGR = {
    "red":    (0, 0, 255),
    "blue":   (255, 0, 0),
    "green":  (0, 200, 0),
    "yellow": (0, 255, 255),
    "purple": (200, 0, 200),
    "orange": (0, 140, 255),
    "brown":  (30, 70, 130),
}


# =============================================================================
# Line Detector
# =============================================================================

class LineDetector:
    """Detects metro lines from HSV color masks."""

    def __init__(self):
        self.color_ranges = config.load_calibrated_colors()
        from vision.line_end_detector import LineEndDetector
        self.end_detector = LineEndDetector()

    def detect(
        self,
        frame: np.ndarray,
        stations: List[DetectedStation] = None,
    ) -> List[DetectedLine]:
        """
        Detect all metro lines in the frame.

        Args:
            frame: BGR numpy array (H x W x 3)
            stations: optional list of detected stations for connection mapping

        Returns:
            List of DetectedLine objects (one per active color)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        diag = math.sqrt(w * w + h * h)
        proximity = diag * config.LINE_PROXIMITY_RATIO

        lines = []

        for color_name, ranges in self.color_ranges.items():
            # Build combined mask for this color
            mask = np.zeros((h, w), dtype=np.uint8)
            for (lo, hi) in ranges:
                lo_arr = np.array(lo, dtype=np.uint8)
                hi_arr = np.array(hi, dtype=np.uint8)
                partial = cv2.inRange(hsv, lo_arr, hi_arr)
                mask = cv2.bitwise_or(mask, partial)

            # Mask out HUD regions (top ~6%, bottom ~12%)
            top_margin = int(h * getattr(config, 'STATION_TOP_MARGIN', 0.06))
            hud_cutoff = int(h * 0.88)
            mask[:top_margin, :] = 0
            mask[hud_cutoff:, :] = 0

            # Count colored pixels — skip if too few (no line of this color)
            pixel_count = cv2.countNonZero(mask)
            # A real metro line is a thick path spanning significant area.
            # At 1920x1080, even a short line segment has thousands of pixels.
            min_pixels = int(frame.shape[0] * 2)  # at least 2 full rows of pixels
            if pixel_count < min_pixels:
                continue

            # Morphological close to fill gaps in the line
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE),
            )
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Skeletonize to get the center path of the line
            skeleton = self._skeletonize(mask_clean)

            # Detect endpoints
            endpoints = self.end_detector.get_endpoints(mask_clean, skeleton)

            # Extract path points from skeleton
            path_points = self._extract_path_points(skeleton)

            # Determine which stations are connected
            connected_ids = []
            if stations:
                connected_ids = self._find_connected_stations(
                    path_points, mask_clean, stations, proximity
                )

            lines.append(DetectedLine(
                color=color_name,
                station_ids=connected_ids,
                path_points=path_points,
                pixel_count=pixel_count,
                mask=mask_clean,
                endpoints=endpoints,
            ))

        return lines

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """
        Reduce a binary mask to a 1-pixel-wide skeleton.
        
        Uses cv2.ximgproc.thinning if available, otherwise falls back
        to morphological skeleton via erosion.
        """
        try:
            skeleton = cv2.ximgproc.thinning(
                mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
            return skeleton
        except AttributeError:
            # Fallback: morphological skeleton
            return self._morphological_skeleton(mask)

    def _morphological_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """Morphological skeleton fallback (Zhang-Suen approximation)."""
        skeleton = np.zeros_like(mask)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = mask.copy()

        while True:
            eroded = cv2.erode(temp, element)
            dilated = cv2.dilate(eroded, element)
            diff = cv2.subtract(temp, dilated)
            skeleton = cv2.bitwise_or(skeleton, diff)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break

        return skeleton

    def _extract_path_points(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extract ordered path points from a skeleton image.
        
        Returns a subsampled list of (x, y) coordinates along the skeleton.
        """
        # Get all nonzero points
        ys, xs = np.nonzero(skeleton)
        if len(xs) == 0:
            return []

        # Subsample to reduce noise (every 5th point)
        points = list(zip(xs.tolist(), ys.tolist()))

        # Sort by x then y for rough ordering
        # (for complex routes this is imperfect but sufficient for proximity checks)
        points.sort(key=lambda p: (p[0], p[1]))

        # Subsample
        step = max(1, len(points) // 200)  # keep ~200 points max
        return points[::step]

    def _find_connected_stations(
        self,
        path_points: List[Tuple[int, int]],
        line_mask: np.ndarray,
        stations: List[DetectedStation],
        proximity: float,
    ) -> List[int]:
        """
        Determine which stations are connected by this line.
        
        A station is "connected" if:
        1. It is within `proximity` pixels of any path point, OR
        2. There are colored line pixels within a small radius of the station center
        """
        connected = []

        for station in stations:
            # Method 1: Check line mask near station center
            mask_roi_r = int(proximity * 0.8)
            x1 = max(0, station.cx - mask_roi_r)
            y1 = max(0, station.cy - mask_roi_r)
            x2 = min(line_mask.shape[1], station.cx + mask_roi_r)
            y2 = min(line_mask.shape[0], station.cy + mask_roi_r)

            roi = line_mask[y1:y2, x1:x2]
            if roi.size > 0 and cv2.countNonZero(roi) > 20:
                connected.append(station.id)
                continue

            # Method 2: Check distance to nearest path point
            if path_points:
                min_dist = min(
                    math.sqrt((station.cx - px)**2 + (station.cy - py)**2)
                    for px, py in path_points
                )
                if min_dist < proximity:
                    connected.append(station.id)

        return connected


# =============================================================================
# Debug drawing
# =============================================================================

def draw_lines(
    frame: np.ndarray,
    lines: List[DetectedLine],
    stations: List[DetectedStation] = None,
) -> np.ndarray:
    """Draw detected lines and their connections onto a copy of the frame."""
    overlay = frame.copy()

    for line in lines:
        color = COLOR_BGR.get(line.color, (255, 255, 255))

        # Draw path points as a polyline
        if len(line.path_points) > 1:
            pts = np.array(line.path_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], False, color, 2)

        # Draw connected stations
        if stations:
            station_map = {s.id: s for s in stations}
            for sid in line.station_ids:
                if sid in station_map:
                    s = station_map[sid]
                    cv2.circle(overlay, (s.cx, s.cy), 12, color, 3)

        # Label
        if line.path_points:
            lx, ly = line.path_points[0]
            cv2.putText(
                overlay,
                f"{line.color} ({len(line.station_ids)} stn)",
                (lx, ly - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

    return overlay


# =============================================================================
# Self-test (snapshot mode)
# =============================================================================

def main():
    """Snapshot mode: capture one frame, detect lines, save annotated image."""
    import sys
    import os
    import time as _time
    from capture import ScreenCapture
    from vision.station_detector import StationDetector

    print("=" * 60)
    print("  Mini Metro AI -- Line Detector Test (Snapshot)")
    print("=" * 60)

    cap = ScreenCapture()
    station_det = StationDetector()
    line_det = LineDetector()

    print("\nCapturing frame in 2 seconds (switch to game window)...")
    _time.sleep(2)

    frame = cap.grab_frame()
    stations = station_det.detect(frame)
    lines = line_det.detect(frame, stations)

    print(f"\nDetected {len(lines)} lines, {len(stations)} stations:")
    for line in lines:
        print(f"  [{line.color:8s}] {len(line.station_ids)} stations "
              f"connected, {line.pixel_count} pixels")

    overlay = draw_lines(frame, lines, stations)
    info = f"Lines: {len(lines)} | Stations: {len(stations)}"
    cv2.putText(overlay, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(out_dir, exist_ok=True)
    ann_path = os.path.join(out_dir, "line_detect_annotated.png")
    cv2.imwrite(ann_path, overlay)
    print(f"\nSaved annotated: {ann_path}")

    print("Showing preview (press any key to close)...")
    cv2.namedWindow("Line Detection Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Line Detection Result", 960, 540)
    cv2.imshow("Line Detection Result", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

