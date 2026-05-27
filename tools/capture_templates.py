"""
Mini Metro AI — Template Capture Tool

Helps capture HUD icon templates for the HUD parser.
Opens a game screenshot and lets you draw ROIs around icons.

Usage:
    python tools/capture_templates.py
"""

import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

import config


class TemplateCapturer:
    """Interactive template capture from game screenshots."""

    def __init__(self):
        self.roi_start = None
        self.roi_end = None
        self.drawing = False

    def run(self):
        """Main template capture flow."""
        from capture import ScreenCapture

        print("=" * 60)
        print("  Mini Metro AI — Template Capture Tool")
        print("=" * 60)
        print()
        print("This tool captures HUD icon templates for resource counting.")
        print("Make sure Mini Metro is running.")
        print()

        cap = ScreenCapture()
        frame = cap.grab_frame()

        templates_to_capture = ["train_icon", "carriage_icon", "tunnel_icon"]

        for template_name in templates_to_capture:
            print(f"\n--- Capture: {template_name} ---")
            print("Draw a rectangle around the icon. Press 'S' to save, "
                  "'R' to redo, 'K' to skip.")

            self.roi_start = None
            self.roi_end = None

            window_name = f"Capture: {template_name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, self._on_mouse)

            while True:
                display = frame.copy()

                if self.roi_start and self.roi_end:
                    cv2.rectangle(
                        display, self.roi_start, self.roi_end,
                        (0, 255, 0), 2
                    )

                cv2.putText(
                    display,
                    f"Select ROI for: {template_name} | "
                    f"S=save R=redo K=skip Q=quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

                cv2.imshow(window_name, display)
                key = cv2.waitKey(50) & 0xFF

                if key == ord("s") or key == ord("S"):
                    if self.roi_start and self.roi_end:
                        self._save_template(frame, template_name)
                    break
                elif key == ord("r") or key == ord("R"):
                    self.roi_start = None
                    self.roi_end = None
                elif key == ord("k") or key == ord("K"):
                    print(f"  Skipped {template_name}")
                    break
                elif key == ord("q") or key == ord("Q"):
                    cv2.destroyAllWindows()
                    return

            cv2.destroyWindow(window_name)

        cv2.destroyAllWindows()
        print("\nTemplate capture complete!")

    def _on_mouse(self, event, x, y, flags, param):
        """Mouse callback for ROI selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_start = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.roi_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_end = (x, y)
            self.drawing = False

    def _save_template(self, frame: np.ndarray, name: str):
        """Crop and save a template."""
        x1 = min(self.roi_start[0], self.roi_end[0])
        y1 = min(self.roi_start[1], self.roi_end[1])
        x2 = max(self.roi_start[0], self.roi_end[0])
        y2 = max(self.roi_start[1], self.roi_end[1])

        crop = frame[y1:y2, x1:x2]

        # Convert to grayscale (templates are matched in grayscale)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        path = os.path.join(config.TEMPLATE_DIR, f"{name}.png")
        cv2.imwrite(path, gray)
        print(f"  Saved template: {path} ({gray.shape})")


def main():
    capturer = TemplateCapturer()
    capturer.run()


if __name__ == "__main__":
    main()
