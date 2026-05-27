"""
Mini Metro AI Agent — Debug Overlay (Layer 6)

Renders a debug visualization window showing detected game state
overlaid on the captured frame. Shows stations, lines, queues,
resources, planned actions, and trend indicators.
"""

import cv2
import numpy as np
from typing import List, Optional

import config
from state import GameState, Station, Line
from engine.planner import Action
from vision.station_detector import SHAPE_COLORS
from vision.line_detector import COLOR_BGR


class DebugOverlay:
    """Renders debug visualization of the agent's perception and decisions."""

    WINDOW_NAME = "Mini Metro AI — Debug"

    def __init__(self):
        self._frame_count = 0
        self._fps = 0.0
        self._last_fps_time = 0.0
        self._fps_frame_count = 0

    def render(
        self,
        frame: np.ndarray,
        state: Optional[GameState] = None,
        actions: Optional[List[Action]] = None,
        trend: str = "STABLE",
        processing_ms: float = 0,
    ) -> np.ndarray:
        """
        Render debug overlay on a copy of the frame.
        
        Args:
            frame: raw BGR frame
            state: current GameState (if available)
            actions: planned actions (if available)
            trend: queue trend string
            processing_ms: time spent on vision+planning in ms
            
        Returns:
            Annotated frame for display
        """
        overlay = frame.copy()
        self._frame_count += 1
        self._update_fps()

        if state:
            self._draw_lines(overlay, state)
            self._draw_stations(overlay, state)
            self._draw_resources(overlay, state)
            self._draw_trend(overlay, trend)

        if actions:
            self._draw_actions(overlay, actions)

        self._draw_info_bar(overlay, state, processing_ms)

        return overlay

    def show(self, overlay: np.ndarray):
        """Display the overlay in a cv2 window."""
        cv2.imshow(self.WINDOW_NAME, overlay)

    def _update_fps(self):
        """Update rolling FPS counter."""
        import time
        now = time.time()
        self._fps_frame_count += 1
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._last_fps_time = now

    def _draw_stations(self, frame: np.ndarray, state: GameState):
        """Draw station markers with shape labels and queue counts."""
        for s in state.stations:
            color = SHAPE_COLORS.get(s.shape, (255, 255, 255))

            # Outer ring — color by urgency
            if s.queue_size >= config.QUEUE_CRITICAL_THRESHOLD:
                ring_color = (0, 0, 255)  # red = critical
                ring_thick = 3
            elif s.queue_size >= config.QUEUE_WARN_THRESHOLD:
                ring_color = (0, 165, 255)  # orange = warning
                ring_thick = 2
            else:
                ring_color = (0, 200, 0)  # green = ok
                ring_thick = 1

            cv2.circle(frame, (s.px, s.py), 22, ring_color, ring_thick)

            # Shape icon (inner)
            self._draw_shape_icon(frame, s.px, s.py, s.shape, color)

            # ID label
            cv2.putText(
                frame, f"#{s.id}",
                (s.px - 12, s.py - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
            )

            # Queue count
            if s.queue_size > 0:
                q_color = (0, 0, 255) if s.queue_size >= config.QUEUE_CRITICAL_THRESHOLD \
                    else (0, 200, 255)
                cv2.putText(
                    frame, f"Q:{s.queue_size}",
                    (s.px + 24, s.py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, q_color, 1,
                )

            # Score
            if s.score > 0:
                cv2.putText(
                    frame, f"S:{s.score:.1f}",
                    (s.px + 24, s.py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1,
                )

    def _draw_shape_icon(self, frame, cx, cy, shape, color):
        """Draw a small shape icon at the given position."""
        r = 8
        if shape == "circle":
            cv2.circle(frame, (cx, cy), r, color, 2)
        elif shape == "triangle":
            pts = np.array([
                [cx, cy - r],
                [cx - r, cy + r],
                [cx + r, cy + r],
            ], dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 2)
        elif shape == "square":
            cv2.rectangle(frame, (cx - r, cy - r), (cx + r, cy + r), color, 2)
        elif shape == "diamond":
            pts = np.array([
                [cx, cy - r],
                [cx + r, cy],
                [cx, cy + r],
                [cx - r, cy],
            ], dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 2)
        elif shape == "pentagon":
            import math
            pts = []
            for i in range(5):
                angle = math.radians(90 + i * 72)
                px = int(cx + r * math.cos(angle))
                py = int(cy - r * math.sin(angle))
                pts.append([px, py])
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, color, 2)
        elif shape == "star":
            cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_STAR, r * 2, 2)
        elif shape == "cross":
            cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, r * 2, 2)
        else:
            cv2.circle(frame, (cx, cy), r, (128, 128, 128), 1)

    def _draw_lines(self, frame: np.ndarray, state: GameState):
        """Draw detected metro lines with labels."""
        station_map = {s.id: s for s in state.stations}

        for line in state.lines:
            color = COLOR_BGR.get(line.color, (255, 255, 255))

            # Draw connections between consecutive stations
            for i in range(len(line.station_ids) - 1):
                s1 = station_map.get(line.station_ids[i])
                s2 = station_map.get(line.station_ids[i + 1])
                if s1 and s2:
                    cv2.line(frame, (s1.px, s1.py), (s2.px, s2.py), color, 2)

            # Line label
            if line.station_ids and line.station_ids[0] in station_map:
                s = station_map[line.station_ids[0]]
                cv2.putText(
                    frame,
                    f"{line.color} L:{line.load_score:.0f}",
                    (s.px - 30, s.py + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                )

    def _draw_resources(self, frame: np.ndarray, state: GameState):
        """Draw resource counts in bottom-left corner."""
        h = frame.shape[0]
        y_base = h - 80

        res = state.resources
        texts = [
            f"Trains: {res.spare_trains}",
            f"Carriages: {res.spare_carriages}",
            f"Tunnels: {res.spare_tunnels}",
        ]

        for i, text in enumerate(texts):
            cv2.putText(
                frame, text,
                (10, y_base + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

    def _draw_trend(self, frame: np.ndarray, trend: str):
        """Draw trend indicator in top-right corner."""
        w = frame.shape[1]

        trend_colors = {
            "STABLE": (0, 200, 0),
            "RISING": (0, 165, 255),
            "CRITICAL": (0, 0, 255),
        }
        color = trend_colors.get(trend, (200, 200, 200))

        # Background rectangle
        text = f"TREND: {trend}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (w - tw - 20, 5), (w - 5, th + 15), (0, 0, 0), -1)
        cv2.putText(
            frame, text,
            (w - tw - 15, th + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    def _draw_actions(self, frame: np.ndarray, actions: List[Action]):
        """Draw planned actions in top-left area."""
        y = 60
        cv2.putText(
            frame, "PLANNED ACTIONS:",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
        )
        y += 18

        for action in actions[:5]:  # show top 5
            if action.type == "noop":
                continue
            color = (0, 255, 255) if action.requires_pause else (200, 255, 200)
            prefix = "[P]" if action.requires_pause else "   "
            text = f"{prefix} p{action.priority}: {action.description[:60]}"
            cv2.putText(
                frame, text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
            )
            y += 16

    def _draw_info_bar(self, frame: np.ndarray, state: Optional[GameState],
                        processing_ms: float):
        """Draw FPS and status info bar at top."""
        h, w = frame.shape[:2]

        # Semi-transparent bar
        bar = frame[0:35, :].copy()
        frame[0:35, :] = cv2.addWeighted(bar, 0.4, np.zeros_like(bar), 0.6, 0)

        frame_id = state.frame_id if state else self._frame_count
        paused = state.is_paused if state else False
        n_stations = len(state.stations) if state else 0
        n_lines = len(state.lines) if state else 0

        info = (f"FPS: {self._fps:.1f} | "
                f"Frame: {frame_id} | "
                f"Vision: {processing_ms:.0f}ms | "
                f"Stations: {n_stations} | "
                f"Lines: {n_lines}")

        if paused:
            info += " | ⏸ PAUSED"

        cv2.putText(
            frame, info,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
