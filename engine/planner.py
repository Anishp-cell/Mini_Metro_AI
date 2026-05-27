"""
Mini Metro AI Agent — Rule-Based Planner (Layer 4B)

The "brain" of the agent. Evaluates the game state each frame and
outputs a prioritized list of actions for the executor.

No weekly bonus selection logic — not available in this game version.
"""

import time
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import config
from state import GameState, Station, Line, Resources
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

    def plan(self, state: GameState, trend: str, geo_detector = None) -> List[Action]:
        """
        Generate a list of actions based on current game state.
        
        Args:
            state: current GameState
            trend: "STABLE", "RISING", or "CRITICAL" from StateBuilder
            geo_detector: GeographyDetector to evaluate river crossings
            
        Returns:
            List of Actions, sorted by priority (lowest number first)
        """
        self.geo_detector = geo_detector

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
    # Helper to find best drag starting position
    # -----------------------------------------------------------------

    def _get_line_extend_start_pos(self, state: GameState, line_color: str, target_station: Station) -> Tuple[int, int]:
        """
        Finds the best starting pixel position (T-cap endpoint or terminal station)
        to extend a line.
        """
        # Find the line object
        line = next((l for l in state.lines if l.color == line_color), None)
        if not line:
            return (target_station.px, target_station.py) # fallback
            
        # 1. Try to use geometric endpoints (T-caps) detected by vision
        if line.endpoints:
            best_ep = None
            best_dist = float("inf")
            for ep in line.endpoints:
                d = (ep[0] - target_station.px)**2 + (ep[1] - target_station.py)**2
                if d < best_dist:
                    best_dist = d
                    best_ep = ep
            if best_ep:
                return best_ep
                
        # 2. Fallback: use the terminal station of the line that is closest to the target
        station_map = {s.id: s for s in state.stations}
        if line.station_ids:
            terminals = []
            if len(line.station_ids) > 0:
                s_first = station_map.get(line.station_ids[0])
                if s_first:
                    terminals.append(s_first)
            if len(line.station_ids) > 1:
                s_last = station_map.get(line.station_ids[-1])
                if s_last:
                    terminals.append(s_last)
                    
            if terminals:
                best_term = min(terminals, key=lambda s: (s.px - target_station.px)**2 + (s.py - target_station.py)**2)
                return (best_term.px, best_term.py)
                
        # 3. Ultimate fallback: use closest station
        return (target_station.px, target_station.py)

    # -----------------------------------------------------------------
    # Rule implementations
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # State Simulation & Look-Ahead Evaluation
    # -----------------------------------------------------------------

    def _simulate_action(self, state: GameState, action: Action) -> GameState:
        """
        Simulate the impact of an action on a lightweight copy of the state.
        """
        # Shallow copy stations
        sim_stations = []
        station_map = {}
        for s in state.stations:
            s_copy = Station(
                id=s.id,
                x=s.x, y=s.y,
                px=s.px, py=s.py,
                shape=s.shape,
                queue_size=s.queue_size,
                connected_lines=list(s.connected_lines),
                score=s.score
            )
            sim_stations.append(s_copy)
            station_map[s.id] = s_copy

        # Shallow copy lines
        sim_lines = []
        line_map = {}
        for l in state.lines:
            l_copy = Line(
                color=l.color,
                station_ids=list(l.station_ids),
                load_score=l.load_score,
                endpoints=list(l.endpoints)
            )
            sim_lines.append(l_copy)
            line_map[l.color] = l_copy

        # Copy resources
        sim_resources = Resources(
            spare_trains=state.resources.spare_trains,
            spare_carriages=state.resources.spare_carriages,
            spare_tunnels=state.resources.spare_tunnels,
            spare_train_positions=list(state.resources.spare_train_positions),
            spare_carriage_positions=list(state.resources.spare_carriage_positions)
        )

        sim_unconnected = list(state.unconnected_stations)

        # Apply action changes to simulation
        t = action.type

        if t == "connect_station":
            from_id = action.params.get("from_id")
            to_id = action.params.get("to_id")
            color = "new_sim_line"
            sim_lines.append(Line(
                color=color,
                station_ids=[from_id, to_id],
                load_score=0.0,
                endpoints=[]
            ))
            if from_id in station_map:
                station_map[from_id].connected_lines.append(color)
            if to_id in station_map:
                station_map[to_id].connected_lines.append(color)

            if from_id in sim_unconnected:
                sim_unconnected.remove(from_id)
            if to_id in sim_unconnected:
                sim_unconnected.remove(to_id)

        elif t == "extend_line":
            station_id = action.params.get("station_id")
            line_color = action.params.get("line_color")
            target_station_id = action.params.get("target_station_id")

            if line_color in line_map:
                line = line_map[line_color]
                if line.station_ids and target_station_id == line.station_ids[0]:
                    line.station_ids.insert(0, station_id)
                else:
                    line.station_ids.append(station_id)

            if station_id in station_map:
                station_map[station_id].connected_lines.append(line_color)

            if station_id in sim_unconnected:
                sim_unconnected.remove(station_id)

        elif t == "add_train":
            sim_resources.spare_trains -= 1

        elif t == "add_carriage":
            sim_resources.spare_carriages -= 1

        return GameState(
            stations=sim_stations,
            lines=sim_lines,
            resources=sim_resources,
            is_paused=state.is_paused,
            is_game_over=state.is_game_over,
            frame_id=state.frame_id,
            timestamp=state.timestamp,
            station_graph={},
            unconnected_stations=sim_unconnected
        )

    def _evaluate_state(self, state: GameState) -> float:
        """
        Evaluate the topological value/health of a GameState.
        Higher is better.
        """
        score = 0.0

        # 1. Unconnected stations penalty (extremely severe)
        score -= 1000.0 * len(state.unconnected_stations)

        # 2. Station queue congestion
        # Penalize high individual queue sizes quadratically (squared) to avoid major bottlenecks
        for s in state.stations:
            score -= (s.queue_size ** 2) * 5.0
            if s.queue_size >= config.QUEUE_CRITICAL_THRESHOLD:
                score -= 100.0
            elif s.queue_size >= config.QUEUE_WARN_THRESHOLD:
                score -= 20.0

        # 3. Line properties: shape alternation and length balance
        station_map = {s.id: s for s in state.stations}
        for line in state.lines:
            n_stations = len(line.station_ids)
            if n_stations == 0:
                continue

            # Length penalty (keep lines under 6 stations if possible)
            if n_stations > 6:
                score -= (n_stations - 6) ** 2 * 10.0

            # Shape alternation/diversity:
            # We want circles to alternate with triangles/squares, not consecutive circle-runs
            shapes = [station_map[sid].shape for sid in line.station_ids if sid in station_map]
            
            # Count consecutive duplicates (e.g. circle next to circle)
            duplicates = 0
            for i in range(len(shapes) - 1):
                if shapes[i] == shapes[i+1]:
                    duplicates += 1
            score -= duplicates * 80.0 # large penalty for duplicate shape runs

            # Reward unique/rare shapes on the line
            unique_on_line = sum(1 for shape in shapes if shape in {"pentagon", "star", "cross", "diamond", "square"})
            score += unique_on_line * 15.0

            # Reward line shape completeness ( Circle + Triangle + Square )
            line_shapes_set = set(shapes)
            if "circle" in line_shapes_set and "triangle" in line_shapes_set:
                score += 30.0
            if "square" in line_shapes_set:
                score += 20.0

        # 4. River crossing penalty
        tunnels_used = 0
        if getattr(self, "geo_detector", None) is not None:
            for line in state.lines:
                for i in range(len(line.station_ids) - 1):
                    s1 = station_map.get(line.station_ids[i])
                    s2 = station_map.get(line.station_ids[i+1])
                    if s1 and s2:
                        if self.geo_detector.crosses_river((s1.px, s1.py), (s2.px, s2.py)):
                            tunnels_used += 1

            if tunnels_used > state.resources.spare_tunnels + tunnels_used:
                score -= 10000.0 # illegal crossing exceeding spare tunnels
            else:
                score -= tunnels_used * 15.0 # minor penalty for using tunnel resources

        return score

    # -----------------------------------------------------------------
    # Rule implementations (Look-Ahead Search)
    # -----------------------------------------------------------------

    def _plan_connect_unconnected(self, state: GameState) -> List[Action]:
        """Rule 1: Connect any station not on any line using look-ahead search."""
        actions = []

        if not state.unconnected_stations:
            return actions

        station_map = {s.id: s for s in state.stations}
        candidates = []

        for sid in state.unconnected_stations:
            station = station_map.get(sid)
            if station is None:
                continue

            # Option A: Extend an existing line to this station
            for line in state.lines:
                # Find nearest station on this line to represent target connection point
                nearest = None
                nearest_dist = float("inf")
                for l_sid in line.station_ids:
                    s = station_map.get(l_sid)
                    if s:
                        d = math.sqrt((station.x - s.x)**2 + (station.y - s.y)**2)
                        if d < nearest_dist:
                            nearest_dist = d
                            nearest = s

                if nearest:
                    start_drag_pos = self._get_line_extend_start_pos(state, line.color, station)
                    
                    candidate = Action(
                        type="extend_line",
                        priority=1,
                        requires_pause=True,
                        params={
                            "station_id": sid,
                            "station_pos": (station.px, station.py),
                            "target_station_id": nearest.id,
                            "target_pos": start_drag_pos,
                            "line_color": line.color,
                        },
                        description=f"Connect #{sid} ({station.shape}) via {line.color}",
                    )

                    # River/Tunnel constraint check
                    if getattr(self, "geo_detector", None) is not None:
                        if self.geo_detector.crosses_river((station.px, station.py), start_drag_pos):
                            if state.resources.spare_tunnels <= 0:
                                continue # Invalid candidate

                    # Simulate and evaluate
                    sim_state = self._simulate_action(state, candidate)
                    candidate_score = self._evaluate_state(sim_state)
                    candidate.score = candidate_score
                    candidates.append(candidate)

            # Option B: Create the very first line (if no lines exist yet)
            if not state.lines:
                for s in state.stations:
                    if s.id == sid:
                        continue

                    # River crossing check for first line
                    if getattr(self, "geo_detector", None) is not None:
                        if self.geo_detector.crosses_river((station.px, station.py), (s.px, s.py)):
                            if state.resources.spare_tunnels <= 0:
                                continue

                    candidate = Action(
                        type="connect_station",
                        priority=0,
                        requires_pause=True,
                        params={
                            "from_id": sid,
                            "from_pos": (station.px, station.py),
                            "to_id": s.id,
                            "to_pos": (s.px, s.py),
                        },
                        description=f"Create first line: #{sid} -> #{s.id}",
                    )
                    sim_state = self._simulate_action(state, candidate)
                    candidate_score = self._evaluate_state(sim_state)
                    candidate.score = candidate_score
                    candidates.append(candidate)

        # Promote the highest-evaluated search candidate
        if candidates:
            candidates.sort(key=lambda c: getattr(c, "score", -999999.0), reverse=True)
            best_cand = candidates[0]
            if best_cand.type == "extend_line":
                best_cand.description = f"Connect #{best_cand.params['station_id']} ({station_map[best_cand.params['station_id']].shape}) via {best_cand.params['line_color']} to #{best_cand.params['target_station_id']} (Sim Score={best_cand.score:.1f})"
            elif best_cand.type == "connect_station":
                best_cand.description = f"Create first line: #{best_cand.params['from_id']} -> #{best_cand.params['to_id']} (Sim Score={best_cand.score:.1f})"
            actions.append(best_cand)

        return actions

    def _plan_handle_critical(self, state: GameState) -> List[Action]:
        """Rule 2: Restructure around critical stations using look-ahead search."""
        actions = []
        critical = find_critical_stations(state)

        if not critical:
            return actions

        station_map = {s.id: s for s in state.stations}
        candidates = []

        for station in critical:
            if len(station.connected_lines) < 2 and len(state.lines) > 1:
                # Evaluate connecting other active lines to the critical station
                for line in state.lines:
                    if line.color not in station.connected_lines:
                        nearest = None
                        nearest_dist = float("inf")
                        for l_sid in line.station_ids:
                            s = station_map.get(l_sid)
                            if s:
                                d = math.sqrt((station.x - s.x)**2 + (station.y - s.y)**2)
                                if d < nearest_dist:
                                    nearest_dist = d
                                    nearest = s

                        if nearest:
                            start_drag_pos = self._get_line_extend_start_pos(state, line.color, station)
                            
                            # River/Tunnel constraint check
                            if getattr(self, "geo_detector", None) is not None:
                                if self.geo_detector.crosses_river((station.px, station.py), start_drag_pos):
                                    if state.resources.spare_tunnels <= 0:
                                        continue

                            candidate = Action(
                                type="extend_line",
                                priority=2,
                                requires_pause=True,
                                params={
                                    "station_id": station.id,
                                    "station_pos": (station.px, station.py),
                                    "target_station_id": nearest.id,
                                    "target_pos": start_drag_pos,
                                    "line_color": line.color,
                                },
                                description=f"Extend {line.color} to critical #{station.id}",
                            )
                            sim_state = self._simulate_action(state, candidate)
                            candidate_score = self._evaluate_state(sim_state)
                            candidate.score = candidate_score
                            candidates.append(candidate)

        if candidates:
            candidates.sort(key=lambda c: getattr(c, "score", -999999.0), reverse=True)
            best_cand = candidates[0]
            best_cand.description = f"Extend {best_cand.params['line_color']} to critical #{best_cand.params['station_id']} (Sim Score={best_cand.score:.1f}, Q={station_map[best_cand.params['station_id']].queue_size})"
            actions.append(best_cand)

        return actions

    def _plan_add_trains(self, state: GameState) -> List[Action]:
        """Rule 3: Add spare trains to overloaded lines."""
        actions = []

        if state.resources.spare_trains <= 0 or not state.resources.spare_train_positions:
            return actions

        station_map = {s.id: s for s in state.stations}

        for line in state.lines:
            if line.load_score > config.LINE_OVERLOAD_THRESHOLD:
                # Find busiest station on this line to drop the train on
                line_stations = [station_map[sid] for sid in line.station_ids if sid in station_map]
                if not line_stations:
                    continue
                busiest_station = max(line_stations, key=lambda s: s.queue_size)

                actions.append(Action(
                    type="add_train",
                    priority=3,
                    requires_pause=False,
                    params={
                        "line_color": line.color,
                        "hud_pos": state.resources.spare_train_positions[0],
                        "station_pos": (busiest_station.px, busiest_station.py),
                    },
                    description=f"Add train to {line.color} at station #{busiest_station.id} "
                                f"(load={line.load_score:.1f}, Q={busiest_station.queue_size})",
                ))
                break  # one train per planning cycle

        return actions

    def _plan_add_carriages(self, state: GameState) -> List[Action]:
        """Rule 4: Add spare carriages to busy lines."""
        actions = []

        if state.resources.spare_carriages <= 0 or not state.resources.spare_carriage_positions:
            return actions

        busiest = find_busiest_line(state)
        if busiest and busiest.load_score > config.QUEUE_WARN_THRESHOLD:
            # Find busiest station on this line to drop the carriage on
            station_map = {s.id: s for s in state.stations}
            line_stations = [station_map[sid] for sid in busiest.station_ids if sid in station_map]
            if line_stations:
                busiest_station = max(line_stations, key=lambda s: s.queue_size)
                actions.append(Action(
                    type="add_carriage",
                    priority=4,
                    requires_pause=False,
                    params={
                        "line_color": busiest.color,
                        "hud_pos": state.resources.spare_carriage_positions[0],
                        "station_pos": (busiest_station.px, busiest_station.py),
                    },
                    description=f"Add carriage to {busiest.color} at station #{busiest_station.id} "
                                f"(load={busiest.load_score:.1f}, Q={busiest_station.queue_size})",
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
