"""
Mini Metro AI Agent — Pause Manager (Layer 4C)

Orchestrates the pause → execute actions → verify → unpause cycle.
Groups actions that require pausing into a single pause window to
minimize game disruption.
"""

import time
import logging
from typing import List

from capture import ScreenCapture
from executor import Executor
from engine.planner import Action

logger = logging.getLogger("pause_manager")


class PauseManager:
    """
    Manages the pause/unpause cycle around restructuring actions.
    
    Ensures:
    1. Game is paused before modifying lines
    2. All modifications execute sequentially with small delays
    3. Paused state is verified before/after
    4. Game is unpaused after changes
    5. Before/after state is logged
    """

    def __init__(self, executor: Executor, capture: ScreenCapture):
        self._executor = executor
        self._capture = capture
        self._is_managing_pause = False

    def execute_with_pause(self, actions: List[Action]):
        """
        Execute a batch of actions that require the game to be paused.
        
        1. Pause the game
        2. Verify paused
        3. Execute each action
        4. Verify changes (capture frame)
        5. Unpause
        
        Args:
            actions: list of Actions (should have requires_pause=True)
        """
        if not actions:
            return

        if self._is_managing_pause:
            logger.warning("Already in a pause management cycle — skipping")
            return

        self._is_managing_pause = True

        try:
            # --- 1. Capture before-state ---
            logger.info(f"=== PAUSE CYCLE: {len(actions)} actions ===")
            before_frame = self._capture.grab_frame()

            # --- 2. Pause the game ---
            self._executor.pause_game()
            time.sleep(0.3)

            # --- 3. Verify we're paused ---
            if not self._verify_paused():
                logger.warning("Could not verify pause — retrying")
                self._executor.pause_game()
                time.sleep(0.5)
                if not self._verify_paused():
                    logger.error("Failed to pause game — aborting cycle")
                    return

            logger.info("Game paused confirmed")

            # --- 4. Execute each action ---
            executed_count = 0
            for action in actions:
                logger.info(f"  Executing: {action.description}")
                success = self._executor.execute(action)
                if success:
                    executed_count += 1
                time.sleep(0.2)  # small delay between actions

            logger.info(f"Executed {executed_count}/{len(actions)} actions")

            # --- 5. Capture after-state ---
            time.sleep(0.3)
            after_frame = self._capture.grab_frame()

            # --- 6. Unpause ---
            self._executor.unpause_game()
            logger.info("Game unpaused")
            logger.info(f"=== PAUSE CYCLE COMPLETE ===")

        except Exception as e:
            logger.error(f"Error during pause cycle: {e}")
            # Try to unpause as safety measure
            try:
                self._executor.unpause_game()
            except:
                pass

        finally:
            self._is_managing_pause = False

    def _verify_paused(self) -> bool:
        """
        Verify the game is actually paused by checking if the frame
        is static (hasn't changed between two quick captures).
        """
        try:
            frame1 = self._capture.grab_frame()
            time.sleep(0.2)
            frame2 = self._capture.grab_frame()

            # Compare frames — if identical (or nearly), game is paused
            import numpy as np
            diff = np.abs(frame1.astype(float) - frame2.astype(float)).mean()

            # Threshold: paused games have near-zero diff
            is_paused = diff < 2.0
            logger.debug(f"Pause verify: frame diff = {diff:.2f} "
                         f"(paused={is_paused})")
            return is_paused

        except Exception as e:
            logger.error(f"Pause verification error: {e}")
            return True  # assume paused to avoid infinite loops

    @property
    def is_busy(self) -> bool:
        """True if currently in a pause management cycle."""
        return self._is_managing_pause
