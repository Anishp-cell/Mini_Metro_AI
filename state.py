"""
Mini Metro AI Agent — Game State Model (Layer 3)

Combines outputs from all vision detectors into a unified
GameState dataclass. Also builds a station graph for
pathfinding and computes per-line load scores.

Usage:
    python state.py    # Live state printout from capture
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple

import config
from vision.station_detector import DetectedStation
from vision.line_detector import DetectedLine
from vision.hud_parser import HUDState


# =============================================================================
# Unified data classes (normalized coordinates, enriched)
# =============================================================================

@dataclass
class Station:
    """A station in the game state (normalized coordinates)."""
    id: int
    x: float               # normalized 0-1
    y: float               # normalized 0-1
    px: int                 # pixel x (for action execution)
    py: int                 # pixel y
    shape: str
    queue_size: int
    connected_lines: List[str] = field(default_factory=list)
    score: float = 0.0     # urgency score from scorer


@dataclass
class Line:
    """A metro line in the game state."""
    color: str
    station_ids: List[int]
    load_score: float = 0.0  # sum of queue sizes of connected stations


@dataclass
class Resources:
    """Available resources."""
    spare_trains: int = 0
    spare_carriages: int = 0
    spare_tunnels: int = 0


@dataclass
class GameState:
    """Complete game state for a single frame."""
    stations: List[Station]
    lines: List[Line]
    resources: Resources
    is_paused: bool
    is_game_over: bool
    frame_id: int
    timestamp: float

    # Derived data
    station_graph: Dict[int, Set[int]] = field(default_factory=dict)
    unconnected_stations: List[int] = field(default_factory=list)


# =============================================================================
# State Builder
# =============================================================================

class StateBuilder:
    """
    Assembles GameState from raw vision outputs.
    Maintains a history buffer for trend detection.
    """

    def __init__(self):
        self._frame_counter = 0
        self._history: deque = deque(maxlen=config.STATE_HISTORY_SIZE)

    def build(
        self,
        detected_stations: List[DetectedStation],
        detected_lines: List[DetectedLine],
        hud_state: HUDState,
    ) -> GameState:
        """
        Build a GameState from raw detector outputs.
        """
        self._frame_counter += 1

        # Convert detected stations → state stations
        stations = []
        for ds in detected_stations:
            connected = []
            for dl in detected_lines:
                if ds.id in dl.station_ids:
                    connected.append(dl.color)
            stations.append(Station(
                id=ds.id,
                x=ds.nx, y=ds.ny,
                px=ds.cx, py=ds.cy,
                shape=ds.shape,
                queue_size=ds.queue_size,
                connected_lines=connected,
            ))

        # Convert detected lines → state lines
        station_map = {s.id: s for s in stations}
        lines = []
        for dl in detected_lines:
            load = sum(
                station_map[sid].queue_size
                for sid in dl.station_ids
                if sid in station_map
            )
            lines.append(Line(
                color=dl.color,
                station_ids=dl.station_ids,
                load_score=load,
            ))

        # Resources
        resources = Resources(
            spare_trains=hud_state.spare_trains,
            spare_carriages=hud_state.spare_carriages,
            spare_tunnels=hud_state.spare_tunnels,
        )

        # Build graph
        graph = self._build_graph(lines)

        # Find unconnected stations
        all_connected = set()
        for line in lines:
            all_connected.update(line.station_ids)
        unconnected = [s.id for s in stations if s.id not in all_connected]

        state = GameState(
            stations=stations,
            lines=lines,
            resources=resources,
            is_paused=hud_state.is_paused,
            is_game_over=hud_state.is_game_over,
            frame_id=self._frame_counter,
            timestamp=time.time(),
            station_graph=graph,
            unconnected_stations=unconnected,
        )

        self._history.append(state)
        return state

    def _build_graph(self, lines: List[Line]) -> Dict[int, Set[int]]:
        """
        Build adjacency graph: nodes = station IDs, edges = direct line connections.
        Two stations are adjacent if they are consecutive on the same line.
        """
        graph: Dict[int, Set[int]] = {}

        for line in lines:
            sids = line.station_ids
            for i, sid in enumerate(sids):
                if sid not in graph:
                    graph[sid] = set()
                # Connect to previous and next station on this line
                if i > 0:
                    graph[sid].add(sids[i - 1])
                if i < len(sids) - 1:
                    graph[sid].add(sids[i + 1])
                # Ensure neighbor entries exist
                for neighbor in list(graph.get(sid, [])):
                    if neighbor not in graph:
                        graph[neighbor] = set()
                    graph[neighbor].add(sid)

        return graph

    def get_history(self) -> List[GameState]:
        """Get recent state history (oldest first)."""
        return list(self._history)

    def get_avg_queue(self, n_frames: int = 5) -> float:
        """Get average queue size across stations over last n frames."""
        if not self._history:
            return 0.0

        recent = list(self._history)[-n_frames:]
        total_q = 0
        total_s = 0
        for state in recent:
            for s in state.stations:
                total_q += s.queue_size
                total_s += 1
        return total_q / total_s if total_s > 0 else 0.0

    def get_queue_trend(self) -> str:
        """
        Detect if average queue sizes are rising, stable, or critical.
        
        Returns: "RISING", "STABLE", or "CRITICAL"
        """
        history = list(self._history)
        window = config.TREND_WINDOW

        if len(history) < window:
            return "STABLE"

        # Check for any critical station
        latest = history[-1]
        for s in latest.stations:
            if s.queue_size >= config.QUEUE_CRITICAL_THRESHOLD:
                return "CRITICAL"

        # Check if average is rising
        avgs = []
        for state in history[-window:]:
            if state.stations:
                avg = sum(s.queue_size for s in state.stations) / len(state.stations)
            else:
                avg = 0
            avgs.append(avg)

        rising_count = sum(1 for i in range(1, len(avgs)) if avgs[i] > avgs[i - 1])
        if rising_count >= window - 1:
            return "RISING"

        return "STABLE"


# =============================================================================
# Pretty printing
# =============================================================================

def print_state(state: GameState):
    """Print a human-readable summary of the game state."""
    print(f"\n{'=' * 60}")
    print(f"  Frame #{state.frame_id} | "
          f"Paused: {state.is_paused} | "
          f"GameOver: {state.is_game_over}")
    print(f"{'=' * 60}")

    print(f"\n  Stations ({len(state.stations)}):")
    for s in state.stations:
        flag = ""
        if s.queue_size >= config.QUEUE_CRITICAL_THRESHOLD:
            flag = " ⚠️ CRITICAL"
        elif s.queue_size >= config.QUEUE_WARN_THRESHOLD:
            flag = " ⚡ WARNING"
        connected = ", ".join(s.connected_lines) if s.connected_lines else "NONE"
        print(f"    #{s.id} {s.shape:10s} ({s.x:.2f},{s.y:.2f}) "
              f"Q={s.queue_size} [{connected}]{flag}")

    print(f"\n  Lines ({len(state.lines)}):")
    for l in state.lines:
        print(f"    [{l.color:8s}] stations={l.station_ids} load={l.load_score:.1f}")

    print(f"\n  Resources: trains={state.resources.spare_trains} "
          f"carriages={state.resources.spare_carriages} "
          f"tunnels={state.resources.spare_tunnels}")

    if state.unconnected_stations:
        print(f"\n  ⚠️  Unconnected stations: {state.unconnected_stations}")

    print()


# =============================================================================
# Self-test
# =============================================================================

def main():
    """Build and print game state from live capture."""
    from capture import ScreenCapture
    from vision.station_detector import StationDetector
    from vision.line_detector import LineDetector
    from vision.hud_parser import HUDParser

    print("=" * 60)
    print("  Mini Metro AI — State Builder Test")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    cap = ScreenCapture()
    station_det = StationDetector()
    line_det = LineDetector()
    hud_parser = HUDParser()
    builder = StateBuilder()

    import cv2

    while True:
        frame = cap.grab_frame()

        stations = station_det.detect(frame)
        lines = line_det.detect(frame, stations)
        hud = hud_parser.parse(frame)

        state = builder.build(stations, lines, hud)
        trend = builder.get_queue_trend()

        print_state(state)
        print(f"  Trend: {trend}")

        cv2.imshow("State - Raw Frame", frame)
        key = cv2.waitKey(500) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
