"""
Mini Metro AI Agent — Main Agent Loop (Step 8)

The central orchestrator that wires together:
    capture → vision → state → scorer → planner → executor

Usage:
    python agent.py            # Run the agent
    python agent.py --debug    # Save debug frames + verbose console output
    python agent.py --dry-run  # Vision + planning only, no actions
"""

import sys
import time
import logging
import argparse
from typing import List

import numpy as np

import cv2

import config
from capture import ScreenCapture
from vision.station_detector import StationDetector
from vision.line_detector import LineDetector
from vision.hud_parser import HUDParser
from vision.geography_detector import GeographyDetector
from state import StateBuilder, print_state
from engine.scorer import score_all
from engine.planner import Planner, Action
from engine.pause_manager import PauseManager
from executor import Executor
from debug_overlay import DebugOverlay


# =============================================================================
# Logging setup
# =============================================================================

def setup_logging():
    """Configure logging to file and console."""
    import os
    os.makedirs(config.LOG_DIR, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.LOG_DIR, f"agent_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(
                open(sys.stdout.fileno(), mode='w', encoding='utf-8',
                     closefd=False)
            ),
        ],
    )
    return logging.getLogger("agent")


# =============================================================================
# Kill switch
# =============================================================================

class KillSwitch:
    """Global kill switch — press 'Q' to stop the agent."""

    def __init__(self):
        self.triggered = False
        self._listener = None
        self._start_listener()

    def _start_listener(self):
        """Start a background keyboard listener."""
        try:
            from pynput import keyboard

            def on_press(key):
                try:
                    if key.char and key.char.lower() == 'q':
                        self.triggered = True
                        return False  # stop listener
                except AttributeError:
                    pass

            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.daemon = True
            self._listener.start()

        except ImportError:
            # Fallback: use cv2.waitKey in the main loop
            pass

    def check(self) -> bool:
        """Check if the kill switch has been triggered."""
        return self.triggered


# =============================================================================
# Main Agent
# =============================================================================

class MiniMetroAgent:
    """The main agent that plays Mini Metro."""

    def __init__(self, debug: bool = False, dry_run: bool = False):
        self.debug = debug
        self.dry_run = dry_run
        self.logger = logging.getLogger("agent")
        self.running = False

        # Initialize all components
        self.logger.info("Initializing Mini Metro AI Agent...")

        self.capture = ScreenCapture()
        self.station_detector = StationDetector()
        self.line_detector = LineDetector()
        self.hud_parser = HUDParser()
        self.geography_detector = GeographyDetector()
        self.state_builder = StateBuilder()
        self.planner = Planner()
        self.executor = Executor(capture=self.capture)
        self.pause_manager = PauseManager(self.executor, self.capture)
        self.overlay = DebugOverlay() if debug else None
        self.kill_switch = KillSwitch()

        self.logger.info("All components initialized.")
        if dry_run:
            self.logger.info("DRY RUN mode — no actions will be executed.")

    def run(self):
        """Main agent loop."""
        self.running = True
        self.logger.info("=" * 60)
        self.logger.info("  Mini Metro AI Agent — STARTING")
        self.logger.info(f"  Mode: {'DEBUG' if self.debug else 'NORMAL'}"
                         f" | {'DRY RUN' if self.dry_run else 'LIVE'}")
        self.logger.info(f"  Target FPS: {config.CAPTURE_FPS}")
        self.logger.info(f"  Kill switch: press 'Q' to stop")
        self.logger.info("=" * 60)

        target_delay = 1.0 / config.CAPTURE_FPS
        frame_count = 0

        # --- Auto-focus game window and give user time to switch ---
        self.logger.info("")
        self.logger.info("  Bringing game window to foreground...")
        try:
            import ctypes
            import win32gui
            hwnd = self.capture.hwnd
            # Try to bring the game window to front
            win32gui.ShowWindow(hwnd, 5)  # SW_SHOW
            win32gui.SetForegroundWindow(hwnd)
            self.logger.info("  Game window focused!")
        except Exception as e:
            self.logger.warning(f"  Could not auto-focus game: {e}")
            self.logger.info("  Please alt-tab to the game window NOW!")

        for i in range(3, 0, -1):
            self.logger.info(f"  Starting in {i}...")
            time.sleep(1)
        self.logger.info("  GO!")

        start_time = time.time()

        try:
            while self.running:
                t0 = time.time()

                # --- 1. Capture frame ---
                try:
                    frame = self.capture.grab_frame()
                except RuntimeError as e:
                    self.logger.error(f"Capture failed: {e}")
                    break

                # --- 1.5. Check for milestone/upgrade popup ---
                if not self.dry_run and self._detect_milestone_screen(frame):
                    self._handle_milestone(frame)
                    time.sleep(1.0)  # wait for popup to dismiss
                    continue  # skip this frame, re-capture next

                # --- 2. Vision pipeline ---
                t_vision = time.time()
                self.geography_detector.update(frame)
                stations = self.station_detector.detect(frame)
                lines = self.line_detector.detect(frame, stations)
                hud = self.hud_parser.parse(frame)
                vision_ms = (time.time() - t_vision) * 1000

                # --- 3. Build game state ---
                state = self.state_builder.build(stations, lines, hud)
                trend = self.state_builder.get_queue_trend()

                # --- 4. Score everything ---
                score_all(state)

                # --- 5. Plan actions ---
                actions = self.planner.plan(state, trend, self.geography_detector)

                # --- 6. Execute actions ---
                if not self.dry_run and not state.is_game_over:
                    self._execute_actions(actions)

                # --- 7. Debug output (console + periodic frame dumps) ---
                if self.debug:
                    n_crit = sum(1 for s in state.stations
                                 if s.queue_size >= config.QUEUE_CRITICAL_THRESHOLD)
                    n_warn = sum(1 for s in state.stations
                                 if s.queue_size >= config.QUEUE_WARN_THRESHOLD)
                    action_desc = actions[0].description if actions else "none"
                    is_interesting = (n_crit > 0 or n_warn > 0
                                      or "Connect" in action_desc
                                      or "Restructure" in action_desc)

                    # Print every 10th frame, or immediately if something interesting
                    if frame_count % 10 == 0 or is_interesting:
                        self.logger.info(
                            f"F{frame_count:4d} | "
                            f"S:{len(state.stations)} L:{len(state.lines)} | "
                            f"Trend:{trend:8s} Warn:{n_warn} Crit:{n_crit} | "
                            f"Act: {action_desc[:60]} | "
                            f"{vision_ms:.0f}ms"
                        )
                    # Save annotated debug frame periodically
                    if self.overlay and frame_count % 50 == 1:
                        overlay_frame = self.overlay.render(
                            frame, state, actions, trend, vision_ms
                        )
                        self._save_debug_frame(overlay_frame, frame_count)

                # --- 8. Log periodically ---
                frame_count += 1
                if frame_count % 50 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed if elapsed > 0 else 0
                    self.logger.info(
                        f"Frame {frame_count} | "
                        f"FPS: {avg_fps:.1f} | "
                        f"Stations: {len(state.stations)} | "
                        f"Lines: {len(state.lines)} | "
                        f"Trend: {trend} | "
                        f"Vision: {vision_ms:.0f}ms"
                    )

                # --- 9. Save debug frames ---
                if config.LOG_FRAMES and frame_count % config.LOG_FRAME_INTERVAL == 0:
                    self._save_frame(frame, frame_count)

                # --- 10. Game over check ---
                if state.is_game_over:
                    self.logger.warning("GAME OVER detected!")
                    self._on_game_over(state)

                # --- 11. Kill switch (press Q anywhere) ---
                if self.kill_switch.check():
                    self.logger.info("Kill switch triggered (Q pressed) -- stopping")
                    break

                # --- 12. Frame rate control ---
                dt = time.time() - t0
                sleep_time = max(0, target_delay - dt)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            self.logger.info("Ctrl+C received — stopping")

        finally:
            self.running = False
            elapsed = time.time() - start_time
            self.logger.info(f"Agent stopped. "
                             f"Ran for {elapsed:.1f}s, {frame_count} frames "
                             f"({frame_count/elapsed:.1f} fps)")

    def _execute_actions(self, actions: List[Action]):
        """Execute planned actions, one pause-action per cycle with cooldown."""
        # Cooldown check — don't spam actions
        now = time.time()
        if not hasattr(self, '_last_action_time'):
            self._last_action_time = 0
        if now - self._last_action_time < 2.0:  # 2 second cooldown
            return

        pause_actions = [a for a in actions if a.requires_pause]
        immediate_actions = [a for a in actions
                             if not a.requires_pause and a.type != "noop"]

        # Execute immediate actions first (no pause needed)
        for action in immediate_actions:
            self.logger.info(f"Executing: {action}")
            self.executor.execute(action)
            self._last_action_time = now

        # Execute ONLY the highest-priority pause action (not all at once)
        # In Mini Metro, each line drag needs the state to settle before the next
        if pause_actions and not self.pause_manager.is_busy:
            top_action = pause_actions[0]  # already sorted by priority
            self.logger.info(f"Pause cycle: {top_action.description}")
            self.pause_manager.execute_with_pause([top_action])
            self._last_action_time = time.time()

    def _on_game_over(self, state):
        """Handle game over."""
        self.logger.info(f"Final state: {len(state.stations)} stations, "
                         f"{len(state.lines)} lines")
        # Could auto-restart here in the future
        self.running = False

    def _save_frame(self, frame, frame_id):
        """Save a raw frame to disk for later review."""
        import os
        os.makedirs(config.FRAME_LOG_DIR, exist_ok=True)
        path = os.path.join(config.FRAME_LOG_DIR, f"frame_{frame_id:06d}.png")
        cv2.imwrite(path, frame)

    def _save_debug_frame(self, overlay_frame, frame_id):
        """Save an annotated debug frame to logs/debug/."""
        import os
        debug_dir = os.path.join(config.LOG_DIR, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, f"debug_{frame_id:06d}.png")
        cv2.imwrite(path, overlay_frame)
        self.logger.info(f"Saved debug frame: {path}")

    # -----------------------------------------------------------------
    # Milestone / Upgrade popup detection
    # -----------------------------------------------------------------

    def _detect_milestone_screen(self, frame: np.ndarray) -> bool:
        """
        Detect the 'Ridership Milestone' upgrade popup.
        
        Uses high-precision contour shape and circularity filtering to ensure
        100% immunity to Night Mode backgrounds and normal paused states.
        """
        h, w = frame.shape[:2]

        # Upgrade options appear in the middle height band of the screen
        cy1, cy2 = int(h * 0.35), int(h * 0.65)
        cx1, cx2 = int(w * 0.2), int(w * 0.8)
        center = frame[cy1:cy2, cx1:cx2]

        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        # Threshold high-brightness white components
        _, white_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        icons_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Milestone circular icons are large (typically 60-120px diameter -> area 2500 to 12000 px)
            if 2000 < area < 15000:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    # Highly circular shape check
                    if circularity > 0.80:
                        icons_count += 1

        # The milestone selection overlay always displays exactly 1 or 2 circular buttons
        is_milestone = (icons_count == 1 or icons_count == 2)

        if is_milestone:
            self.logger.info(f"MILESTONE SCREEN detected! Found {icons_count} upgrade circle buttons.")
        return is_milestone

    def _handle_milestone(self, frame: np.ndarray):
        """
        Handle the milestone popup by clicking the best upgrade option.

        Strategy:
        - If two options visible: click the LEFT one (usually Locomotive or Line)
        - If one option (announcement): click center to dismiss

        Upgrade priority: Locomotive > Line > Carriage > Tunnel
        The left option tends to be the better one.
        """
        h, w = frame.shape[:2]

        # Detect how many white circles (option icons) are present
        cy1, cy2 = int(h * 0.25), int(h * 0.65)
        cx1, cx2 = int(w * 0.2), int(w * 0.8)
        center = frame[cy1:cy2, cx1:cx2]

        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Find contours (the circular icons)
        contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter for large circular contours (the option icons)
        icons = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000:  # too small to be an icon
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            icon_cx = int(M['m10'] / M['m00']) + cx1  # convert back to frame coords
            icon_cy = int(M['m01'] / M['m00']) + cy1
            icons.append((icon_cx, icon_cy, area))

        if len(icons) >= 2:
            # Two options — click the LEFT one (sort by x)
            icons.sort(key=lambda ic: ic[0])
            click_x, click_y = icons[0][0], icons[0][1]
            self.logger.info(f"Milestone: 2 options, clicking LEFT at ({click_x}, {click_y})")
        elif len(icons) == 1:
            # Single option — click it
            click_x, click_y = icons[0][0], icons[0][1]
            self.logger.info(f"Milestone: 1 option, clicking at ({click_x}, {click_y})")
        else:
            # No icons detected — click center to dismiss
            click_x, click_y = w // 2, h // 2
            self.logger.info(f"Milestone: no icons found, clicking center ({click_x}, {click_y})")

        self.executor.click_at(click_x, click_y)
        time.sleep(0.5)

        # Sometimes need a second click to fully dismiss
        self.executor.click_at(click_x, click_y)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mini Metro AI Agent — Autonomous gameplay"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Verbose console output + save debug frames to logs/debug/"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Vision + planning only, no mouse/keyboard actions"
    )
    args = parser.parse_args()

    logger = setup_logging()

    try:
        agent = MiniMetroAgent(debug=args.debug, dry_run=args.dry_run)
        agent.run()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
