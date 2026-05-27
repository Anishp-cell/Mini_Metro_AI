"""
Mini Metro AI Agent — Rule-Based Planner (Layer 4B)

The "brain" of the agent. Evaluates the game state each frame and
outputs a prioritized list of actions for the executor.

No weekly bonus selection logic — not available in this game version.
"""

import time
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import config
from state import GameState, Station, Line
from engine.scorer import (
    score_all,
    find_critical_stations,
    find_warning_stations,
    find_busiest_line,
    find_least_busy_line,
)


# =============================================================================
# Action definition
# =============================================================================

@dataclass
class Action:
    """An action the agent wants to execute."""
    type: str          # see ACTION_TYPES below
    priority: int      # lower = more urgent (0 = top priority)
    params: dict = field(default_factory=dict)
    requires_pause: bool = False
    description: str = ""

    def __repr__(self):
        return f"Action({self.type}, p={self.priority}, {self.description})"


# Valid action types
ACTION_TYPES = {
    "pause",
    "unpause",
    "connect_station",     # Draw a line to an unconnected station
    "extend_line",         # Extend an existing line to a new station
    "add_train",           # Add spare train to a line
    "add_carriage",        # Add spare carriage to a line
    "reroute_line",        # Delete and redraw part of a line
    "noop",                # Do nothing this frame
}


# =============================================================================
# Planner
# =============================================================================

class Planner:
    """
    Rule-based decision planner.
    
    Evaluates the game state and returns an ordered list of actions
    to execute. Actions that require pausing are grouped together.
    """

    def __init__(self):
        self._last_rebalance_time = time.time()
        self._game_week = 0  # track progression (not used for bonus)

    def plan(self, state: GameState, trend: str) -> List[Action]:
        """
        Generate a list of actions based on current game state.
        
        Args:
            state: current GameState
            trend: "STABLE", "RISING", or "CRITICAL" from StateBuilder
            
        Returns:
            List of Actions, sorted by priority (lowest number first)
        """
        if state.is_game_over:
            return [Action("noop", 99, description="Game is over")]

        # Score everything first
        station_scores, line_scores = score_all(state)

        actions: List[Action] = []

        # Rule 1: Connect unconnected stations (highest priority)
        actions.extend(self._plan_connect_unconnected(state))

        # Rule 2: Handle critical stations
        actions.extend(self._plan_handle_critical(state))

        # Rule 3: Add trains to overloaded lines
        actions.extend(self._plan_add_trains(state))

        # Rule 4: Add carriages to busy lines
        actions.extend(self._plan_add_carriages(state))

        # Rule 5: Periodic rebalance
        actions.extend(self._plan_rebalance(state))

        # Rule 6: Handle rising trend
        if trend == "RISING" and not any(a.type == "reroute_line" for a in actions):
            actions.extend(self._plan_trend_response(state, trend))

        # Sort by priority
        actions.sort(key=lambda a: a.priority)

        # If nothing to do, return noop
        if not actions:
            actions.append(Action("noop", 99, description="Network stable"))

        return actions

    # -----------------------------------------------------------------
    # Rule implementations
    # -----------------------------------------------------------------

    def _plan_connect_unconnected(self, state: GameState) -> List[Action]:
        """Rule 1: Connect any station not on any line."""
        actions = []

        if not state.unconnected_stations:
            return actions

        station_map = {s.id: s for s in state.stations}

        for sid in state.unconnected_stations:
            station = station_map.get(sid)
            if station is None:
                continue

            # Find the nearest connected station
            best_target = None
            best_dist = float("inf")
            best_line = None

            for s in state.stations:
                if s.id == sid or not s.connected_lines:
                    continue
                dist = math.sqrt((station.x - s.x)**2 + (station.y - s.y)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_target = s
                    best_line = s.connected_lines[0]  # extend this line

            if best_target is not None:
                actions.append(Action(
                    type="extend_line",
                    priority=1,
                    requires_pause=True,
                    params={
                        "station_id": sid,
                        "station_pos": (station.px, station.py),
                        "target_station_id": best_target.id,
                        "target_pos": (best_target.px, best_target.py),
                        "line_color": best_line,
                    },
                    description=f"Connect #{sid} ({station.shape}) via {best_line} "
                                f"to #{best_target.id}",
                ))
            else:
                # No connected stations exist yet — this is the very first line
                # Find the next-nearest unconnected station and create a new line
                for s in state.stations:
                    if s.id == sid:
                        continue
                    dist = math.sqrt((station.x - s.x)**2 + (station.y - s.y)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_target = s

                if best_target:
                    actions.append(Action(
                        type="connect_station",
                        priority=0,
                        requires_pause=True,
                        params={
                            "from_id": sid,
                            "from_pos": (station.px, station.py),
                            "to_id": best_target.id,
                            "to_pos": (best_target.px, best_target.py),
                        },
                        description=f"Create first line: #{sid} -> #{best_target.id}",
                    ))

        return actions

    def _plan_handle_critical(self, state: GameState) -> List[Action]:
        """Rule 2: Restructure around critical stations."""
        actions = []
        critical = find_critical_stations(state)

        for station in critical:
            # Strategy: if station is only on 1 line, try to extend another line to it
            if len(station.connected_lines) < 2 and len(state.lines) > 1:
                # Find a line that doesn't already serve this station
                for line in state.lines:
                    if line.color not in station.connected_lines:
                        # Find nearest station on this line to extend from
                        station_map = {s.id: s for s in state.stations}
                        nearest = None
                        nearest_dist = float("inf")
                        for sid in line.station_ids:
                            s = station_map.get(sid)
                            if s:
                                d = math.sqrt((station.x - s.x)**2 + (station.y - s.y)**2)
                                if d < nearest_dist:
                                    nearest_dist = d
                                    nearest = s

                        if nearest:
                            actions.append(Action(
                                type="extend_line",
                                priority=2,
                                requires_pause=True,
                                params={
                                    "station_id": station.id,
                                    "station_pos": (station.px, station.py),
                                    "target_station_id": nearest.id,
                                    "target_pos": (nearest.px, nearest.py),
                                    "line_color": line.color,
                                },
                                description=f"Extend {line.color} to critical #{station.id} "
                                            f"(Q={station.queue_size})",
                            ))
                            break  # one action per critical station

        return actions

    def _plan_add_trains(self, state: GameState) -> List[Action]:
        """Rule 3: Add spare trains to overloaded lines."""
        actions = []

        if state.resources.spare_trains <= 0:
            return actions

        for line in state.lines:
            if line.load_score > config.LINE_OVERLOAD_THRESHOLD:
                actions.append(Action(
                    type="add_train",
                    priority=3,
                    requires_pause=False,
                    params={
                        "line_color": line.color,
                        "line_station_ids": line.station_ids,
                    },
                    description=f"Add train to {line.color} (load={line.load_score:.1f})",
                ))
                break  # one train per planning cycle

        return actions

    def _plan_add_carriages(self, state: GameState) -> List[Action]:
        """Rule 4: Add spare carriages to busy lines."""
        actions = []

        if state.resources.spare_carriages <= 0:
            return actions

        busiest = find_busiest_line(state)
        if busiest and busiest.load_score > config.QUEUE_WARN_THRESHOLD:
            actions.append(Action(
                type="add_carriage",
                priority=4,
                requires_pause=False,
                params={
                    "line_color": busiest.color,
                    "line_station_ids": busiest.station_ids,
                },
                description=f"Add carriage to {busiest.color} "
                            f"(load={busiest.load_score:.1f})",
            ))

        return actions

    def _plan_rebalance(self, state: GameState) -> List[Action]:
        """Rule 5: Periodic rebalance check."""
        actions = []
        now = time.time()

        if now - self._last_rebalance_time < config.REBALANCE_INTERVAL_SEC:
            return actions

        self._last_rebalance_time = now

        busiest = find_busiest_line(state)
        quietest = find_least_busy_line(state)

        if busiest and quietest and busiest.color != quietest.color:
            load_diff = busiest.load_score - quietest.load_score
            if load_diff > 4:
                actions.append(Action(
                    type="reroute_line",
                    priority=5,
                    requires_pause=True,
                    params={
                        "from_line": quietest.color,
                        "to_line": busiest.color,
                        "reason": "rebalance",
                    },
                    description=f"Rebalance: {busiest.color} "
                                f"(load={busiest.load_score:.1f}) overloaded vs "
                                f"{quietest.color} (load={quietest.load_score:.1f})",
                ))

        return actions

    def _plan_trend_response(self, state: GameState, trend: str) -> List[Action]:
        """Rule 6: React to rising queue trend."""
        actions = []

        if trend == "RISING":
            warning_stations = find_warning_stations(state)
            if warning_stations:
                worst = max(warning_stations, key=lambda s: s.queue_size)
                actions.append(Action(
                    type="reroute_line",
                    priority=6,
                    requires_pause=True,
                    params={
                        "target_station_id": worst.id,
                        "target_pos": (worst.px, worst.py),
                        "reason": "rising_trend",
                    },
                    description=f"Rising trend: restructure around #{worst.id} "
                                f"(Q={worst.queue_size})",
                ))

        return actions
