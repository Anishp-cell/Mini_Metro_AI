"""
Mini Metro AI Agent — Station Detector (Layer 2A)

Detects station positions, classifies their shapes, and counts
passenger queues using contour analysis on each captured frame.

Mini Metro visual style:
  - Dark background (charcoal/dark gray)
  - Station shapes are outlined geometric shapes (lighter than background)
  - Stations are a specific size range (not tiny like passengers)
  - HUD is at the bottom ~10% of the screen
  - Lines are thick colored paths

Usage:
    python -m vision.station_detector              # Snapshot mode (recommended)
    python -m vision.station_detector --live        # Live mode (needs 2nd monitor)
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import cv2
import numpy as np

import config


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class DetectedStation:
    """A station detected in a single frame."""
    id: int
    cx: int              # center x in pixels
    cy: int              # center y in pixels
    nx: float            # normalized x (0-1)
    ny: float            # normalized y (0-1)
    shape: str           # "circle", "triangle", "square", "diamond", "pentagon", "star", "cross", "unknown"
    queue_size: int      # number of passenger icons nearby
    contour_area: float  # for debugging
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) bounding box


# =============================================================================
# Shape classification
# =============================================================================

def classify_shape(contour: np.ndarray, area: float) -> str:
    """
    Classify a contour as a geometric shape.
    Uses vertex count from polygon approximation and circularity.
    """
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "unknown"

    # Circularity: 4pi * area / perimeter^2  (1.0 = perfect circle)
    circularity = (4 * math.pi * area) / (perimeter * perimeter)

    # Polygon approximation
    epsilon = 0.04 * perimeter  # slightly more aggressive simplification
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    # Convexity check — stations should be roughly convex
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # High circularity = circle
    if circularity > 0.82 and vertices >= 6:
        return "circle"

    if vertices == 3:
        return "triangle"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / h if h > 0 else 1
        # Rotated 45° = diamond
        rect_area = w * h
        fill = area / rect_area if rect_area > 0 else 0
        if fill < 0.65:
            return "diamond"
        elif 0.75 < aspect < 1.3:
            return "square"
        else:
            return "diamond"
    elif vertices == 5:
        return "pentagon"
    elif vertices == 6:
        if solidity > 0.85:
            return "circle"  # hexagon ≈ circle
        return "star"
    elif vertices >= 7 and vertices <= 10:
        if solidity < 0.65:
            return "star"
        if circularity > 0.7:
            return "circle"
        return "cross"
    elif vertices > 10:
        if circularity > 0.7:
            return "circle"
        return "unknown"

    return "unknown"


# =============================================================================
# Station Detector
# =============================================================================

class StationDetector:
    """Detects and classifies stations in a game frame."""

    def __init__(self):
        self._next_id = 0
        self._prev_stations: List[DetectedStation] = []
        self._id_counter = 0

    def detect(self, frame: np.ndarray) -> List[DetectedStation]:
        """
        Detect stations in the given BGR frame.

        Args:
            frame: BGR numpy array (H x W x 3)

        Returns:
            List of DetectedStation objects
        """
        h, w = frame.shape[:2]
        frame_area = h * w

        # --- Size thresholds (relative to frame) ---
        min_area = frame_area * config.STATION_MIN_AREA_RATIO
        max_area = frame_area * config.STATION_MAX_AREA_RATIO

        # --- Exclude HUD regions ---
        top_margin = int(h * getattr(config, 'STATION_TOP_MARGIN', 0.06))
        hud_cutoff = int(h * 0.88)
        map_region = frame[top_margin:hud_cutoff, :]

        # --- Mask out colored metro lines ---
        # Metro lines are high-saturation colored paths. Stations are white/gray
        # outlines (low saturation). By zeroing out high-saturation pixels, we
        # reveal station outlines hidden under colored lines.
        hsv = cv2.cvtColor(map_region, cv2.COLOR_BGR2HSV)
        sat_mask = hsv[:, :, 1] > 80  # pixels with saturation > 80 are colored
        map_clean = map_region.copy()
        # Replace colored pixels with the median background color
        # This handles day/night mode and different city palettes dynamically
        median_bg = np.median(map_clean, axis=(0, 1)).astype(np.uint8)
        map_clean[sat_mask] = median_bg

        # --- Convert to grayscale ---
        gray = cv2.cvtColor(map_clean, cv2.COLOR_BGR2GRAY)

        # === PASS 1: Canny edge detection ===
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blurred, 40, 120)  # slightly lower thresholds
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours_pass1, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # === PASS 2: Binary threshold (catches bright outlines on dark bg) ===
        _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        contours_pass2, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # === PASS 3: Adaptive threshold (catches stations on varied backgrounds
        # like the light blue river where fixed threshold fails) ===
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, -10
        )
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        contours_pass3, _ = cv2.findContours(
            adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Combine contours from all passes
        all_contours = list(contours_pass1) + list(contours_pass2) + list(contours_pass3)

        raw_stations = []
        for contour in all_contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < min_area or area > max_area:
                continue

            # Get bounding box
            x, y, bw, bh = cv2.boundingRect(contour)

            # Filter: aspect ratio should be roughly square-ish
            aspect = bw / bh if bh > 0 else 999
            if aspect < 0.4 or aspect > 2.5:
                continue

            # Filter: minimum solidity (stations are convex or near-convex)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.3:
                continue

            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"]) + top_margin  # offset back to full frame

            # Reject HUD, color palette, and bottom corner false positives using localized exclusions
            # center color palette & locked track placeholders
            if 0.28 * w < cx < 0.72 * w and cy > h * 0.76:
                continue
            # bottom right assets & river bends
            if cx > w * 0.83 and cy > h * 0.78:
                continue
            # bottom left line selectors
            if cx < w * 0.17 and cy > h * 0.78:
                continue


            # Classify shape
            shape = classify_shape(contour, area)

            raw_stations.append(DetectedStation(
                id=-1,
                cx=cx, cy=cy,
                nx=cx / w, ny=cy / h,
                shape=shape,
                queue_size=0,
                contour_area=area,
                bbox=(x, y + top_margin, bw, bh),
            ))

        # --- Deduplicate: merge detections within 50px of each other ---
        raw_stations = self._deduplicate(raw_stations, merge_dist=50)

        # --- Count passengers near each station ---
        self._count_passengers(frame[:hud_cutoff, :], raw_stations, frame_area)

        # --- Track station IDs across frames ---
        self._track_ids(raw_stations)

        self._prev_stations = raw_stations
        return raw_stations

    def _deduplicate(self, stations: List[DetectedStation],
                     merge_dist: int = 30) -> List[DetectedStation]:
        """Merge detections that are very close together (duplicates)."""
        if not stations:
            return stations

        merged = []
        used = set()

        # Sort by area (prefer larger detections)
        stations.sort(key=lambda s: s.contour_area, reverse=True)

        for i, s in enumerate(stations):
            if i in used:
                continue
            # Check against all others
            for j in range(i + 1, len(stations)):
                if j in used:
                    continue
                dist = math.sqrt((s.cx - stations[j].cx)**2 +
                                 (s.cy - stations[j].cy)**2)
                if dist < merge_dist:
                    used.add(j)
            merged.append(s)

        return merged

    def _count_passengers(
        self,
        frame: np.ndarray,
        stations: List[DetectedStation],
        frame_area: float,
    ):
        """Count small passenger icons near each station."""
        h, w = frame.shape[:2]
        diag = math.sqrt(w * w + h * h)
        roi_radius = int(diag * config.PASSENGER_ROI_RADIUS_RATIO)

        # Passengers are very small shapes (much smaller than stations)
        min_pax_area = frame_area * config.PASSENGER_MIN_AREA_RATIO
        max_pax_area = frame_area * config.PASSENGER_MAX_AREA_RATIO

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 1)
        edges = cv2.Canny(blurred, 40, 120)

        for station in stations:
            # Define ROI around station
            x1 = max(0, station.cx - roi_radius)
            y1 = max(0, station.cy - roi_radius)
            x2 = min(w, station.cx + roi_radius)
            y2 = min(h, station.cy + roi_radius)

            roi = edges[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            contours, _ = cv2.findContours(
                roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            passenger_count = 0
            for c in contours:
                area = cv2.contourArea(c)
                if min_pax_area < area < max_pax_area:
                    # Make sure this isn't the station itself
                    M = cv2.moments(c)
                    if M["m00"] == 0:
                        continue
                    pcx = int(M["m10"] / M["m00"]) + x1
                    pcy = int(M["m01"] / M["m00"]) + y1
                    # Must be outside the station's bounding box
                    sx, sy, sw, sh = station.bbox
                    if not (sx - 5 <= pcx <= sx + sw + 5 and
                            sy - 5 <= pcy <= sy + sh + 5):
                        passenger_count += 1

            station.queue_size = passenger_count

    def _track_ids(self, current: List[DetectedStation]):
        """Assign consistent IDs across frames using nearest-centroid matching."""
        if not self._prev_stations:
            for s in current:
                s.id = self._id_counter
                self._id_counter += 1
            return

        used_prev = set()
        max_match_dist = 40  # pixels

        for s in current:
            best_dist = float("inf")
            best_prev = None
            for ps in self._prev_stations:
                if ps.id in used_prev:
                    continue
                dist = math.sqrt((s.cx - ps.cx)**2 + (s.cy - ps.cy)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_prev = ps

            if best_prev is not None and best_dist < max_match_dist:
                s.id = best_prev.id
                used_prev.add(best_prev.id)
            else:
                s.id = self._id_counter
                self._id_counter += 1


# =============================================================================
# Debug drawing
# =============================================================================

SHAPE_COLORS = {
    "circle":   (0, 255, 0),
    "triangle": (255, 100, 0),
    "square":   (0, 100, 255),
    "diamond":  (255, 255, 0),
    "pentagon": (255, 0, 255),
    "star":     (0, 255, 255),
    "cross":    (128, 128, 255),
    "unknown":  (128, 128, 128),
}


def draw_stations(frame: np.ndarray, stations: List[DetectedStation]) -> np.ndarray:
    """Draw detected stations onto a copy of the frame."""
    overlay = frame.copy()

    for s in stations:
        color = SHAPE_COLORS.get(s.shape, (255, 255, 255))

        # Draw marker circle
        cv2.circle(overlay, (s.cx, s.cy), 20, color, 2)

        # Draw shape label + queue
        label = f"#{s.id} {s.shape[:4]} Q:{s.queue_size}"
        cv2.putText(
            overlay, label,
            (s.cx - 40, s.cy - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )

        # Draw bounding box
        x, y, w, h = s.bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)

    return overlay


# =============================================================================
# Self-test (snapshot mode — no feedback loop!)
# =============================================================================

def main():
    """
    Snapshot mode: capture ONE frame, detect stations, save annotated image.
    This avoids the feedback loop of showing a cv2 window on top of the game.
    """
    import sys
    import os
    import time as _time

    live_mode = "--live" in sys.argv

    from capture import ScreenCapture

    print("=" * 60)
    print("  Mini Metro AI -- Station Detector Test")
    if live_mode:
        print("  LIVE MODE: Press 'Q' in overlay to quit")
        print("  WARNING: Overlay window may corrupt capture on single monitor!")
    else:
        print("  SNAPSHOT MODE: Capturing one frame for analysis")
    print("=" * 60)

    cap = ScreenCapture()
    detector = StationDetector()

    if live_mode:
        # Live mode — only use with second monitor
        cv2.namedWindow("Station Detector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Station Detector", 960, 540)

        while True:
            t0 = _time.time()
            frame = cap.grab_frame()
            stations = detector.detect(frame)
            dt = _time.time() - t0

            overlay = draw_stations(frame, stations)
            info = f"Stations: {len(stations)} | {dt*1000:.0f}ms"
            cv2.putText(overlay, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Station Detector", overlay)
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()

    else:
        # Snapshot mode — capture one frame, process, save
        print("\nCapturing frame in 2 seconds (switch to game window)...")
        _time.sleep(2)

        frame = cap.grab_frame()
        stations = detector.detect(frame)

        print(f"\nDetected {len(stations)} stations:")
        for s in stations:
            print(f"  #{s.id:3d} {s.shape:10s} at ({s.cx:4d},{s.cy:4d}) "
                  f"Q={s.queue_size} area={s.contour_area:.0f}")

        # Save annotated image
        overlay = draw_stations(frame, stations)
        info = f"Detected: {len(stations)} stations"
        cv2.putText(overlay, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(out_dir, exist_ok=True)

        raw_path = os.path.join(out_dir, "station_detect_raw.png")
        ann_path = os.path.join(out_dir, "station_detect_annotated.png")
        cv2.imwrite(raw_path, frame)
        cv2.imwrite(ann_path, overlay)
        print(f"\nSaved raw frame:  {raw_path}")
        print(f"Saved annotated:  {ann_path}")
        print("\nOpen the annotated image to see detection results.")

        # Also show briefly if user wants
        print("Showing preview (press any key to close)...")
        cv2.namedWindow("Station Detection Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Station Detection Result", 960, 540)
        cv2.imshow("Station Detection Result", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
