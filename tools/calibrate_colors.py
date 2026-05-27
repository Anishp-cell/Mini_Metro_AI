"""
Mini Metro AI — Color Calibration Tool

Interactive tool to calibrate HSV color ranges for line detection.
Captures a screenshot, lets you click on each line color,
and saves the ranges to calibration.json.

Usage:
    python tools/calibrate_colors.py
"""

import json
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

import config


class ColorCalibrator:
    """Interactive HSV color calibration."""

    def __init__(self):
        self.samples = {}
        self.current_color = None
        self.frame = None

    def run(self):
        """Main calibration flow."""
        from capture import ScreenCapture

        print("=" * 60)
        print("  Mini Metro AI — Color Calibration Tool")
        print("=" * 60)
        print()
        print("This tool helps calibrate the HSV ranges for each line color.")
        print("Make sure Mini Metro is running with some lines drawn.")
        print()

        cap = ScreenCapture()
        self.frame = cap.grab_frame()

        color_names = list(config.LINE_COLORS_HSV.keys())
        print(f"Colors to calibrate: {', '.join(color_names)}")
        print()

        for color_name in color_names:
            print(f"\n--- Calibrating: {color_name.upper()} ---")
            print(f"Click on 3-5 points along a {color_name} line in the game.")
            print(f"Press 'N' when done with this color, or 'S' to skip.")

            self.current_color = color_name
            self.samples[color_name] = []

            cv2.namedWindow("Calibrate", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Calibrate", self._on_click)

            while True:
                display = self._draw_samples()
                cv2.imshow("Calibrate", display)
                key = cv2.waitKey(50) & 0xFF

                if key == ord("n") or key == ord("N"):
                    if self.samples[color_name]:
                        print(f"  Collected {len(self.samples[color_name])} "
                              f"samples for {color_name}")
                    break
                elif key == ord("s") or key == ord("S"):
                    print(f"  Skipped {color_name}")
                    del self.samples[color_name]
                    break
                elif key == ord("q") or key == ord("Q"):
                    print("Calibration aborted.")
                    cv2.destroyAllWindows()
                    return

        cv2.destroyAllWindows()

        # Compute and save ranges
        if self.samples:
            ranges = self._compute_ranges()
            self._save_ranges(ranges)
            print("\nCalibration complete!")
        else:
            print("\nNo samples collected.")

    def _on_click(self, event, x, y, flags, param):
        """Mouse callback for sampling colors."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.current_color is None or self.frame is None:
            return

        # Sample a 10x10 region around the click
        h, w = self.frame.shape[:2]
        x1 = max(0, x - 5)
        y1 = max(0, y - 5)
        x2 = min(w, x + 5)
        y2 = min(h, y + 5)

        region = self.frame[y1:y2, x1:x2]
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        mean_hsv = hsv_region.mean(axis=(0, 1))
        self.samples[self.current_color].append(mean_hsv)

        print(f"  Sample at ({x},{y}): HSV = ({mean_hsv[0]:.0f}, "
              f"{mean_hsv[1]:.0f}, {mean_hsv[2]:.0f})")

    def _draw_samples(self) -> np.ndarray:
        """Draw sample points on the frame."""
        display = self.frame.copy()

        for color_name, samples in self.samples.items():
            for hsv in samples:
                # We don't have positions stored, so just show info text
                pass

        # Show current color being calibrated
        cv2.putText(
            display,
            f"Calibrating: {self.current_color} "
            f"({len(self.samples.get(self.current_color, []))} samples) | "
            f"Click line pixels | N=next S=skip Q=quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
        return display

    def _compute_ranges(self) -> dict:
        """Compute HSV ranges from collected samples."""
        ranges = {}

        for color_name, samples in self.samples.items():
            if not samples:
                continue

            arr = np.array(samples)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)

            # Range = mean ± 2*std, clamped to valid HSV ranges
            margin = np.maximum(std * 2.5, [10, 30, 30])

            lo = np.clip(mean - margin, [0, 0, 0], [180, 255, 255])
            hi = np.clip(mean + margin, [0, 0, 0], [180, 255, 255])

            # Special case for red (wraps around H=0/180)
            if color_name == "red" and mean[0] < 15:
                ranges[color_name] = [
                    (lo.astype(int).tolist(), hi.astype(int).tolist()),
                    ([170, int(lo[1]), int(lo[2])], [180, int(hi[1]), int(hi[2])]),
                ]
            elif color_name == "red" and mean[0] > 165:
                ranges[color_name] = [
                    ([0, int(lo[1]), int(lo[2])], [10, int(hi[1]), int(hi[2])]),
                    (lo.astype(int).tolist(), hi.astype(int).tolist()),
                ]
            else:
                ranges[color_name] = [
                    (lo.astype(int).tolist(), hi.astype(int).tolist()),
                ]

            print(f"\n  {color_name}: {ranges[color_name]}")

        return ranges

    def _save_ranges(self, ranges: dict):
        """Save calibrated ranges to calibration.json."""
        path = config.CALIBRATION_FILE
        with open(path, "w") as f:
            json.dump(ranges, f, indent=2)
        print(f"\nSaved calibration to: {path}")


def main():
    calibrator = ColorCalibrator()
    calibrator.run()


if __name__ == "__main__":
    main()
