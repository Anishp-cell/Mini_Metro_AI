"""
Mini Metro AI Agent — Action Executor (Layer 5)

Translates abstract Action objects into concrete mouse/keyboard events
via pyautogui. All coordinates are converted from pixel coords to
screen coords using the capture module's window position.

Usage:
    python executor.py --test pause    # Test pause/unpause
    python executor.py --test draw     # Test drawing a line
"""

import time
import logging
from typing import Tuple, Optional

import pyautogui

import config

# Safety settings
pyautogui.FAILSAFE = True      # Move mouse to top-left corner to abort
pyautogui.PAUSE = 0.05         # Small default pause between pyautogui calls

logger = logging.getLogger("executor")


class Executor:
    """
    Executes game actions via mouse and keyboard.
    
    All position parameters should be in screen pixel coordinates.
    Use capture.norm_to_screen() to convert from normalized coords.
    """

    def __init__(self, capture=None):
        """
        Args:
            capture: ScreenCapture instance for coordinate conversion.
                     If None, positions must already be screen coords.
        """
        self._capture = capture

    def _to_screen(self, px: int, py: int) -> Tuple[int, int]:
        """
        Convert pixel coordinates (within the game frame) to
        absolute screen coordinates.
        """
        if self._capture:
            origin_x, origin_y = self._capture.get_client_origin()
            return (origin_x + px, origin_y + py)
        return (px, py)

    # -----------------------------------------------------------------
    # Basic actions
    # -----------------------------------------------------------------

    def press_key(self, key: str):
        """Press and release a keyboard key."""
        logger.info(f"Key press: {key}")
        pyautogui.press(key)
        time.sleep(config.ACTION_DELAY_SEC)

    def click_at(self, px: int, py: int, button: str = "left"):
        """Click at a game-frame pixel position."""
        sx, sy = self._to_screen(px, py)
        logger.info(f"Click {button} at frame({px},{py}) -> screen({sx},{sy})")
        pyautogui.click(sx, sy, button=button)
        time.sleep(config.ACTION_DELAY_SEC)

    def drag(self, from_px: Tuple[int, int], to_px: Tuple[int, int],
             duration: float = None):
        """
        Smooth drag from one frame position to another.
        Used for drawing/extending lines.
        """
        if duration is None:
            duration = config.DRAG_DURATION_SEC

        sx1, sy1 = self._to_screen(*from_px)
        sx2, sy2 = self._to_screen(*to_px)

        logger.info(f"Drag from ({sx1},{sy1}) -> ({sx2},{sy2}) "
                    f"duration={duration:.2f}s")

        pyautogui.moveTo(sx1, sy1)
        time.sleep(0.05)
        pyautogui.mouseDown()
        time.sleep(0.05)
        pyautogui.moveTo(sx2, sy2, duration=duration)
        time.sleep(0.05)
        pyautogui.mouseUp()
        time.sleep(config.ACTION_DELAY_SEC)

    def drag_multi(self, points: list, duration_per_segment: float = None):
        """
        Drag through multiple points (for multi-station line drawing).
        
        Args:
            points: list of (px, py) game-frame positions
            duration_per_segment: time for each segment
        """
        if len(points) < 2:
            return

        if duration_per_segment is None:
            duration_per_segment = config.DRAG_DURATION_SEC * 0.5

        sx, sy = self._to_screen(*points[0])
        pyautogui.moveTo(sx, sy)
        time.sleep(0.05)
        pyautogui.mouseDown()

        for point in points[1:]:
            sx, sy = self._to_screen(*point)
            pyautogui.moveTo(sx, sy, duration=duration_per_segment)
            time.sleep(0.05)

        pyautogui.mouseUp()
        time.sleep(config.ACTION_DELAY_SEC)

    # -----------------------------------------------------------------
    # Game-specific actions
    # -----------------------------------------------------------------

    def pause_game(self):
        """Pause the game."""
        logger.info("PAUSE game")
        self.press_key("space")
        time.sleep(config.PAUSE_VERIFY_DELAY_SEC)

    def unpause_game(self):
        """Unpause the game."""
        logger.info("UNPAUSE game")
        self.press_key("space")
        time.sleep(config.PAUSE_VERIFY_DELAY_SEC)

    def draw_line(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """
        Draw a new line from one station to another.
        
        Args:
            from_pos: (px, py) of source station in game frame
            to_pos: (px, py) of destination station in game frame
        """
        logger.info(f"Draw line: {from_pos} -> {to_pos}")
        self.drag(from_pos, to_pos)

    def extend_line(self, line_end_pos: Tuple[int, int],
                    new_station_pos: Tuple[int, int]):
        """
        Extend an existing line from its end to a new station.
        
        Args:
            line_end_pos: (px, py) of the last station on the line
            new_station_pos: (px, py) of the station to connect
        """
        logger.info(f"Extend line: {line_end_pos} -> {new_station_pos}")
        self.drag(line_end_pos, new_station_pos)

    def delete_line_segment(self, point_pos: Tuple[int, int]):
        """
        Right-click on a line segment to delete/modify it.
        
        Args:
            point_pos: (px, py) of a point on the line to delete
        """
        logger.info(f"Delete line segment at {point_pos}")
        self.click_at(*point_pos, button="right")

    def add_train_to_line(self, hud_train_pos: Tuple[int, int], target_station_pos: Tuple[int, int]):
        """
        Add a spare train to a line by dragging it from the HUD onto the target station.
        """
        logger.info(f"Add train: drag from HUD {hud_train_pos} -> station {target_station_pos}")
        self.drag(hud_train_pos, target_station_pos)

    def add_carriage_to_line(self, hud_carriage_pos: Tuple[int, int], target_station_pos: Tuple[int, int]):
        """
        Add a spare carriage to a line by dragging it from the HUD onto the target station.
        """
        logger.info(f"Add carriage: drag from HUD {hud_carriage_pos} -> station {target_station_pos}")
        self.drag(hud_carriage_pos, target_station_pos)

    # -----------------------------------------------------------------
    # High-level action dispatcher
    # -----------------------------------------------------------------

    def execute(self, action) -> bool:
        """
        Execute a planner Action object.
        
        Args:
            action: Action from the planner
            
        Returns:
            True if action was executed, False if skipped
        """
        t = action.type

        if t == "noop":
            return False

        elif t == "pause":
            self.pause_game()

        elif t == "unpause":
            self.unpause_game()

        elif t == "connect_station":
            from_pos = action.params.get("from_pos")
            to_pos = action.params.get("to_pos")
            if from_pos and to_pos:
                self.draw_line(from_pos, to_pos)
            else:
                logger.warning(f"connect_station missing positions: {action.params}")
                return False

        elif t == "extend_line":
            target_pos = action.params.get("target_pos")
            station_pos = action.params.get("station_pos")
            if target_pos and station_pos:
                self.extend_line(target_pos, station_pos)
            else:
                logger.warning(f"extend_line missing positions: {action.params}")
                return False

        elif t == "add_train":
            hud_pos = action.params.get("hud_pos")
            station_pos = action.params.get("station_pos")
            if hud_pos and station_pos:
                self.add_train_to_line(hud_pos, station_pos)
            else:
                logger.warning(f"add_train missing positions: {action.params}")
                return False

        elif t == "add_carriage":
            hud_pos = action.params.get("hud_pos")
            station_pos = action.params.get("station_pos")
            if hud_pos and station_pos:
                self.add_carriage_to_line(hud_pos, station_pos)
            else:
                logger.warning(f"add_carriage missing positions: {action.params}")
                return False

        elif t == "reroute_line":
            # Complex action — for now just log it
            logger.info(f"Reroute requested: {action.description}")
            # Future: implement delete + redraw sequence
            return False

        else:
            logger.warning(f"Unknown action type: {t}")
            return False

        logger.info(f"Executed: {action}")
        return True


# =============================================================================
# Self-test
# =============================================================================

def main():
    """Test executor actions individually."""
    import sys

    logging.basicConfig(level=logging.INFO)

    test = sys.argv[1] if len(sys.argv) > 1 else "help"

    if test == "help" or test == "--help":
        print("Usage: python executor.py <test>")
        print("Tests:")
        print("  pause    — Press Space to pause/unpause (press twice)")
        print("  draw     — Draw a line between two hardcoded positions")
        print("  click    — Click at center of screen")
        return

    # Try to get the game window for coordinate conversion
    try:
        from capture import ScreenCapture
        cap = ScreenCapture()
        exec = Executor(capture=cap)
    except Exception as e:
        print(f"Could not find game window: {e}")
        print("Running without coordinate conversion (raw screen coords)")
        exec = Executor()
        cap = None

    print(f"\nRunning test: {test}")
    print("You have 3 seconds to focus the game window...")
    time.sleep(3)

    if test == "pause":
        print("Pausing...")
        exec.pause_game()
        time.sleep(2)
        print("Unpausing...")
        exec.unpause_game()
        print("Done!")

    elif test == "draw":
        if cap:
            # Draw from center-left to center-right of game window
            w, h = cap.width, cap.height
            exec.draw_line((w // 4, h // 2), (3 * w // 4, h // 2))
        else:
            exec.draw_line((400, 400), (800, 400))
        print("Done!")

    elif test == "click":
        if cap:
            exec.click_at(cap.width // 2, cap.height // 2)
        else:
            exec.click_at(960, 540)
        print("Done!")


if __name__ == "__main__":
    main()
