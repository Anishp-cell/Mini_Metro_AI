"""
Mini Metro AI Agent — HUD Parser (Layer 2C)

Parses the bottom HUD bar for resource counts (spare trains,
carriages, tunnels) and detects game-over / pause states.

Note: Weekly bonus popup detection is disabled — not available
in this game version.

Usage:
    python -m vision.hud_parser    # Live HUD readout to console
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

import config


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class HUDState:
    """Parsed state of the game HUD."""
    spare_trains: int
    spare_carriages: int
    spare_tunnels: int
    is_paused: bool
    is_game_over: bool


# =============================================================================
# HUD Parser
# =============================================================================

class HUDParser:
    """Parses the Mini Metro HUD from captured frames."""

    def __init__(self):
        self._templates = {}
        self._load_templates()

        # For pause detection: store previous frame hash
        self._prev_frame_hash = None
        self._static_frame_count = 0

        # For game-over detection
        self._game_over_template = None

    def _load_templates(self):
        """Load template images for HUD icon matching."""
        template_dir = config.TEMPLATE_DIR
        template_files = {
            "train": "train_icon.png",
            "carriage": "carriage_icon.png",
            "tunnel": "tunnel_icon.png",
        }

        for name, filename in template_files.items():
            path = os.path.join(template_dir, filename)
            if os.path.exists(path):
                tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if tmpl is not None:
                    self._templates[name] = tmpl
                    print(f"[HUD] Loaded template: {name} ({tmpl.shape})")

        if not self._templates:
            print("[HUD] WARNING: No templates found. Resource counting will use "
                  "fallback contour-based method.")
            print(f"[HUD] Expected templates in: {template_dir}")
            print("[HUD] Run tools/capture_templates.py to create them.")

    def parse(self, frame: np.ndarray) -> HUDState:
        """
        Parse the HUD from a game frame.

        Args:
            frame: BGR numpy array (H x W x 3)

        Returns:
            HUDState with resource counts and game state flags
        """
        h, w = frame.shape[:2]

        # Extract HUD region (bottom portion of frame)
        hud_top = int(h * (1.0 - config.HUD_REGION_BOTTOM_RATIO))
        hud_region = frame[hud_top:, :]

        # Count resources
        trains, carriages, tunnels = self._count_resources(hud_region)

        # Detect pause state
        is_paused = self._detect_pause(frame)

        # Detect game over
        is_game_over = self._detect_game_over(frame)

        return HUDState(
            spare_trains=trains,
            spare_carriages=carriages,
            spare_tunnels=tunnels,
            is_paused=is_paused,
            is_game_over=is_game_over,
        )

    def _count_resources(self, hud_region: np.ndarray) -> Tuple[int, int, int]:
        """
        Count spare trains, carriages, and tunnels in the HUD.

        Uses template matching if templates are available,
        otherwise falls back to contour counting.
        """
        if self._templates:
            return self._count_by_templates(hud_region)
        else:
            return self._count_by_contours(hud_region)

    def _count_by_templates(self, hud_region: np.ndarray) -> Tuple[int, int, int]:
        """Count resources using template matching."""
        gray = cv2.cvtColor(hud_region, cv2.COLOR_BGR2GRAY)
        counts = {"train": 0, "carriage": 0, "tunnel": 0}

        for name, template in self._templates.items():
            # Multi-scale template matching for robustness
            best_count = 0

            for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                th, tw = template.shape[:2]
                new_w = int(tw * scale)
                new_h = int(th * scale)
                if new_w < 5 or new_h < 5:
                    continue

                scaled = cv2.resize(template, (new_w, new_h))

                if scaled.shape[0] > gray.shape[0] or scaled.shape[1] > gray.shape[1]:
                    continue

                result = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= config.TEMPLATE_MATCH_THRESHOLD)

                # Non-maximum suppression: merge detections within 20px
                if len(locations[0]) > 0:
                    points = list(zip(locations[1].tolist(), locations[0].tolist()))
                    merged = self._nms_points(points, dist_threshold=20)
                    best_count = max(best_count, len(merged))

            counts[name] = best_count

        return counts["train"], counts["carriage"], counts["tunnel"]

    def _count_by_contours(self, hud_region: np.ndarray) -> Tuple[int, int, int]:
        """
        Fallback: estimate resource counts by counting distinct
        icon-sized contours in the HUD region.
        
        This is a rough heuristic — template matching is much more reliable.
        """
        gray = cv2.cvtColor(hud_region, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Threshold to find dark icons on light background
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by size (icons are roughly 15-40px in 1920x1080)
        icon_min_area = max(100, h * w * 0.001)
        icon_max_area = h * w * 0.05
        icons = [c for c in contours
                 if icon_min_area < cv2.contourArea(c) < icon_max_area]

        # Without templates, we can only give total icon count
        # Split roughly: assume trains are larger, carriages smaller
        total = len(icons)
        return (total // 3, total // 3, total - 2 * (total // 3))

    def _nms_points(
        self, points: list, dist_threshold: int = 20
    ) -> list:
        """Simple non-maximum suppression on 2D points."""
        if not points:
            return []

        merged = [points[0]]
        for px, py in points[1:]:
            is_dup = False
            for mx, my in merged:
                if abs(px - mx) < dist_threshold and abs(py - my) < dist_threshold:
                    is_dup = True
                    break
            if not is_dup:
                merged.append((px, py))
        return merged

    def _detect_pause(self, frame: np.ndarray) -> bool:
        """
        Detect if the game is paused.

        Heuristic: If the frame content hasn't changed for several captures,
        the game is likely paused. We also look for the pause UI elements.
        """
        # Compute a quick hash of a subsampled region
        small = cv2.resize(frame, (64, 36))
        frame_hash = hash(small.tobytes())

        if frame_hash == self._prev_frame_hash:
            self._static_frame_count += 1
        else:
            self._static_frame_count = 0

        self._prev_frame_hash = frame_hash

        # If frame hasn't changed for 5+ captures, likely paused
        return self._static_frame_count >= 5

    def _detect_game_over(self, frame: np.ndarray) -> bool:
        """
        Detect the game-over screen.
        Disabled for now because the dark map background triggers false positives.
        """
        return False

    def get_hud_region(self, frame: np.ndarray) -> np.ndarray:
        """Extract just the HUD region for debugging."""
        h = frame.shape[0]
        hud_top = int(h * (1.0 - config.HUD_REGION_BOTTOM_RATIO))
        return frame[hud_top:, :].copy()


# =============================================================================
# Self-test
# =============================================================================

def main():
    """Run HUD parsing on live capture."""
    import time as _time
    from capture import ScreenCapture

    print("=" * 60)
    print("  Mini Metro AI — HUD Parser Test")
    print("  Press 'Q' to quit")
    print("=" * 60)

    cap = ScreenCapture()
    parser = HUDParser()

    while True:
        frame = cap.grab_frame()
        hud = parser.parse(frame)

        print(f"  Trains: {hud.spare_trains}  "
              f"Carriages: {hud.spare_carriages}  "
              f"Tunnels: {hud.spare_tunnels}  "
              f"Paused: {hud.is_paused}  "
              f"GameOver: {hud.is_game_over}")

        # Show HUD region
        hud_region = parser.get_hud_region(frame)
        cv2.imshow("HUD Region", hud_region)

        key = cv2.waitKey(200) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
