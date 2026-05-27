"""
Mini Metro AI Agent — Screen Capture Module (Layer 1)

Captures the Mini Metro game window as numpy arrays.
Primary backend: mss (fast, reliable for visible windows).
Fallback: pywin32 BitBlt (works when partially obscured).

IMPORTANT: Enables DPI awareness so coordinates match actual pixels
on high-DPI / scaled displays (e.g., 125% scaling on Windows).

Usage:
    python capture.py          # Live preview of captured frames
"""

import time
import ctypes
import numpy as np

# =============================================================================
# DPI Awareness — MUST be set before any win32 calls
# This ensures GetWindowRect / GetClientRect return actual pixel values
# instead of scaled values on high-DPI displays (e.g., 125% scaling).
# =============================================================================
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # Fallback for older Windows
    except Exception:
        pass

try:
    import win32gui
    import win32ui
    import win32con
    HAS_PYWIN32 = True
except ImportError:
    HAS_PYWIN32 = False

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

import config


class ScreenCapture:
    """Captures frames from the Mini Metro game window."""

    def __init__(self, window_title: str = None, backend: str = "mss"):
        """
        Args:
            window_title: Window title to search for (default from config).
            backend: "mss" (default, fast, needs visible window) or
                     "pywin32" (BitBlt, works when partially obscured but
                     may fail on some Unity games).
        """
        self.window_title = window_title or config.WINDOW_TITLE
        self.hwnd = None
        self.width = 0
        self.height = 0
        self.window_rect = (0, 0, 0, 0)  # screen-space (left, top, right, bottom)
        self._backend = backend
        self._mss_instance = None

        self._find_window()

    def _find_window(self):
        """Locate the game window by title."""
        if not HAS_PYWIN32:
            raise RuntimeError(
                "pywin32 is required for window detection. "
                "Install with: pip install pywin32"
            )

        self.hwnd = win32gui.FindWindow(None, self.window_title)
        if not self.hwnd:
            # Try partial match — enumerate all windows
            self.hwnd = self._find_window_partial(self.window_title)

        if not self.hwnd:
            raise RuntimeError(
                f"Could not find window with title containing '{self.window_title}'. "
                f"Make sure Mini Metro is running."
            )

        self._update_dimensions()

        # Validate backend
        if self._backend == "mss" and not HAS_MSS:
            print("[Capture] WARNING: mss not installed, falling back to pywin32")
            self._backend = "pywin32"
        if self._backend == "pywin32" and not HAS_PYWIN32:
            raise RuntimeError("pywin32 not installed and mss backend not selected.")

        # Pre-create mss instance for speed
        if self._backend == "mss":
            self._mss_instance = mss.mss()

        print(f"[Capture] Found window: '{self.window_title}' (HWND={self.hwnd})")
        print(f"[Capture] Client area: {self.width}x{self.height}")
        print(f"[Capture] Window rect (screen): {self.window_rect}")
        print(f"[Capture] Client origin (screen): {self._client_origin}")
        print(f"[Capture] Backend: {self._backend}")

    def _find_window_partial(self, partial_title: str):
        """Find window with partial title match (case-insensitive)."""
        result = []

        def enum_callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if partial_title.lower() in title.lower():
                    result.append(hwnd)
            return True

        win32gui.EnumWindows(enum_callback, None)
        return result[0] if result else None

    def _update_dimensions(self):
        """Update window dimensions and position."""
        # Client rect (content area, no borders/titlebar)
        client_rect = win32gui.GetClientRect(self.hwnd)
        self.width = client_rect[2] - client_rect[0]
        self.height = client_rect[3] - client_rect[1]

        # Screen-space rect (with borders) for coordinate mapping
        self.window_rect = win32gui.GetWindowRect(self.hwnd)

        # Client area origin in screen coordinates
        left, top = win32gui.ClientToScreen(self.hwnd, (0, 0))
        self._client_origin = (left, top)

    def grab_frame(self) -> np.ndarray:
        """Capture a single frame as a BGR numpy array (H x W x 3)."""
        if not self.is_window_alive():
            raise RuntimeError("Game window is no longer alive.")

        self._update_dimensions()

        if self._backend == "mss":
            return self._grab_mss()
        else:
            return self._grab_pywin32()

    def _grab_mss(self) -> np.ndarray:
        """
        Capture using mss — fast and reliable.
        Requires the game window to be visible (not minimized/covered).
        """
        if self._mss_instance is None:
            self._mss_instance = mss.mss()

        left, top = self._client_origin
        region = {
            "left": left,
            "top": top,
            "width": self.width,
            "height": self.height,
        }

        screenshot = self._mss_instance.grab(region)
        # mss returns BGRA; convert to BGR numpy array
        img = np.array(screenshot, dtype=np.uint8)
        return img[:, :, :3].copy()  # drop alpha channel

    def _grab_pywin32(self) -> np.ndarray:
        """
        Capture using pywin32 BitBlt.
        Works for most windows; avoids PrintWindow which hangs on Unity games.
        """
        hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, self.width, self.height)
        save_dc.SelectObject(bitmap)

        # Use BitBlt instead of PrintWindow (PrintWindow hangs on Unity games)
        save_dc.BitBlt(
            (0, 0), (self.width, self.height),
            mfc_dc, (0, 0), win32con.SRCCOPY
        )

        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)

        img = np.frombuffer(bmpstr, dtype=np.uint8).reshape(
            (bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4)
        )

        # Clean up GDI objects
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwnd_dc)

        # BGRA → BGR
        return img[:, :, :3].copy()

    def is_window_alive(self) -> bool:
        """Check if the game window still exists."""
        return self.hwnd and win32gui.IsWindow(self.hwnd)

    def get_client_origin(self) -> tuple:
        """
        Get the top-left corner of the client area in screen coordinates.
        Use this to convert normalized coords → screen coords for mouse actions.
        """
        self._update_dimensions()
        return self._client_origin

    def norm_to_screen(self, nx: float, ny: float) -> tuple:
        """
        Convert normalized coordinates (0.0-1.0) to screen pixel coordinates.

        Args:
            nx: normalized x (0 = left edge, 1 = right edge of client area)
            ny: normalized y (0 = top edge, 1 = bottom edge of client area)

        Returns:
            (screen_x, screen_y) in absolute screen pixels
        """
        origin_x, origin_y = self._client_origin
        screen_x = int(origin_x + nx * self.width)
        screen_y = int(origin_y + ny * self.height)
        return (screen_x, screen_y)

    def pixel_to_norm(self, px: int, py: int) -> tuple:
        """
        Convert pixel coordinates (within captured frame) to normalized (0-1).
        """
        return (px / self.width if self.width else 0,
                py / self.height if self.height else 0)


# =============================================================================
# Self-test: live preview
# =============================================================================
def main():
    """Run a live preview window showing captured frames."""
    import cv2

    print("=" * 60)
    print("  Mini Metro AI — Screen Capture Test")
    print("  Press 'Q' in the preview window to quit")
    print("=" * 60)

    cap = ScreenCapture()

    fps_target = config.CAPTURE_FPS
    frame_delay = 1.0 / fps_target
    frame_count = 0
    t_start = time.time()

    while True:
        t0 = time.time()

        try:
            frame = cap.grab_frame()
        except RuntimeError as e:
            print(f"[Capture] Error: {e}")
            break

        if frame is None or frame.size == 0:
            print("[Capture] Empty frame received, retrying...")
            time.sleep(0.1)
            continue

        frame_count += 1
        elapsed = time.time() - t_start
        actual_fps = frame_count / elapsed if elapsed > 0 else 0

        # Draw info overlay
        info_text = (f"FPS: {actual_fps:.1f} | "
                     f"{cap.width}x{cap.height} | "
                     f"{cap._backend} | "
                     f"Press Q to quit")
        cv2.putText(
            frame, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        # Resize preview if the frame is very large
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            preview = cv2.resize(frame, (1280, int(h * scale)))
        else:
            preview = frame

        cv2.imshow("Mini Metro AI - Capture Test", preview)

        # Wait remaining time to hit target FPS, check for quit
        dt = time.time() - t0
        wait_ms = max(1, int((frame_delay - dt) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break

    cv2.destroyAllWindows()
    print(f"[Capture] Done. {frame_count} frames in {elapsed:.1f}s "
          f"({frame_count/elapsed:.1f} fps)")


if __name__ == "__main__":
    main()
