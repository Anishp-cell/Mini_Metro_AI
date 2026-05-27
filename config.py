"""
Mini Metro AI Agent — Central Configuration
All tunable parameters live here.
"""

import os
import json

# =============================================================================
# Window & Capture
# =============================================================================
WINDOW_TITLE = "Mini Metro"
GAME_EXE_PATH = None  # Set to full path of MiniMetro.exe if you want auto-launch
BASELINE_RESOLUTION = (1920, 1080)

CAPTURE_FPS = 10        # frames/sec during normal play
BURST_FPS = 30           # frames/sec right after unpausing
BURST_DURATION_SEC = 2   # how long to stay in burst mode after unpause

# =============================================================================
# Vision — HSV Color Ranges for Line Detection
# Format: {color_name: [(h_low, s_low, v_low), (h_high, s_high, v_high)]}
# Some colors (like red) wrap around H=0/180, so they have two ranges.
# =============================================================================
LINE_COLORS_HSV = {
    "red":    [((0, 150, 150), (10, 255, 255)),
               ((170, 150, 150), (180, 255, 255))],
    "blue":   [((100, 120, 120), (130, 255, 255))],
    "green":  [((40, 120, 120), (80, 255, 255))],
    "yellow": [((20, 150, 150), (35, 255, 255))],
    "purple": [((130, 120, 120), (160, 255, 255))],
    "orange": [((10, 150, 150), (20, 255, 255))],
    "brown":  [((10, 80, 80), (20, 180, 180))],
}

# Path to calibration overrides (written by tools/calibrate_colors.py)
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")

def load_calibrated_colors():
    """Load HSV ranges from calibration file if it exists, else use defaults."""
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            data = json.load(f)
        # Convert lists back to tuples
        return {
            color: [(tuple(lo), tuple(hi)) for lo, hi in ranges]
            for color, ranges in data.items()
        }
    return LINE_COLORS_HSV

# =============================================================================
# Vision — Station Detection
# =============================================================================
STATION_MIN_AREA_RATIO = 0.0003    # min contour area as fraction of frame area
                                   # At 1920x1080: ~622 px² (zoomed-out stations)
STATION_MAX_AREA_RATIO = 0.015     # max contour area as fraction of frame area
                                   # At 1920x1080: ~31104 px² (allows large stations)
PASSENGER_MIN_AREA_RATIO = 0.00005 # min area for passenger icons
PASSENGER_MAX_AREA_RATIO = 0.001   # max area for passenger icons
PASSENGER_ROI_RADIUS_RATIO = 0.05  # radius around station to search for passengers
STATION_TOP_MARGIN = 0.06          # ignore top 6% of screen (clock/day HUD)
CIRCULARITY_THRESHOLD = 0.80       # above this = circle

# =============================================================================
# Vision — HUD Parsing
# =============================================================================
HUD_REGION_BOTTOM_RATIO = 0.15     # bottom 15% of frame is HUD
TEMPLATE_MATCH_THRESHOLD = 0.75    # min confidence for template matching
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "assets", "templates")

# =============================================================================
# Vision — Line Detection
# =============================================================================
LINE_PROXIMITY_RATIO = 0.025       # how close (fraction of frame diag) a station
                                   # must be to a line path to be "connected"
MORPH_KERNEL_SIZE = 5              # morphological close kernel size

# =============================================================================
# Game State & Scoring
# =============================================================================
QUEUE_WARN_THRESHOLD = 4           # passengers before WARNING flag
QUEUE_CRITICAL_THRESHOLD = 6       # passengers before CRITICAL / auto-pause
STATE_HISTORY_SIZE = 30            # number of past states to keep
TREND_WINDOW = 5                   # frames to check for rising trend

# =============================================================================
# Planner
# =============================================================================
REBALANCE_INTERVAL_SEC = 30        # seconds between proactive rebalances
LINE_OVERLOAD_THRESHOLD = 6        # line load score triggering train addition
LONG_LINE_PENALTY_STATIONS = 5     # lines longer than this get penalized

# =============================================================================
# Executor
# =============================================================================
ACTION_DELAY_SEC = 0.15            # delay between distinct mouse/key actions
DRAG_DURATION_SEC = 0.3            # duration of smooth mouse drags
PAUSE_VERIFY_DELAY_SEC = 0.3      # time to wait after pressing Space before verifying

# =============================================================================
# Debug
# =============================================================================
SHOW_OVERLAY = True                # show the debug overlay window
LOG_FRAMES = False                 # save frames to disk for review
LOG_FRAME_INTERVAL = 10            # save every Nth frame
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
FRAME_LOG_DIR = os.path.join(LOG_DIR, "frames")

# =============================================================================
# Ensure directories exist
# =============================================================================
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
if LOG_FRAMES:
    os.makedirs(FRAME_LOG_DIR, exist_ok=True)
