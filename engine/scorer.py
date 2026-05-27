"""
Mini Metro AI Agent — Priority Scorer (Layer 4A)

Scores urgency of each station and line to drive decision making.
Also detects network-level trends (rising queues, critical stations).
"""

import math
from typing import List, Tuple

import config
from state import GameState, Station, Line, StateBuilder


# Shapes that are "unique" (appear rarely, high demand)
UNIQUE_SHAPES = {"pentagon", "star", "cross", "diamond"}


def score_station(station: Station, lines: List[Line], all_stations: List[Station]) -> float:
    """
    Score a station's urgency.
    
    Higher score = more urgent / closer to overflow.
    
    Factors:
        - Base: queue_size
        - ×1.5 if station has a unique shape with no matching destination reachable
        - ×1.3 if station is on only 1 line (single point of failure)
        - ×1.2 if station queue is above warning threshold
    """
    score = float(station.queue_size)

    # Unique shape penalty: if this station's shape is rare and
    # no other station on any connected line shares the shape,
    # passengers can't reach their destination
    if station.shape in UNIQUE_SHAPES:
        has_match = False
        for line in lines:
            if any(
                s.shape == station.shape and s.id != station.id
                for s in all_stations
                if s.id in line.station_ids
            ):
                has_match = True
                break
        if not has_match:
            score *= 1.5

    # Single-line vulnerability
    if len(station.connected_lines) <= 1:
        score *= 1.3

    # Warning escalation
    if station.queue_size >= config.QUEUE_WARN_THRESHOLD:
        score *= 1.2

    return score


def score_line(line: Line, all_stations: List[Station]) -> float:
    """
    Score a line's overall load/urgency.
    
    Factors:
        - Sum of station scores on this line
        - Penalty for very long lines (> LONG_LINE_PENALTY_STATIONS)
    """
    station_map = {s.id: s for s in all_stations}
    total = 0.0

    for sid in line.station_ids:
        if sid in station_map:
            total += station_map[sid].queue_size

    # Long line penalty
    n_stations = len(line.station_ids)
    if n_stations > config.LONG_LINE_PENALTY_STATIONS:
        total *= 1.0 + 0.1 * (n_stations - config.LONG_LINE_PENALTY_STATIONS)

    return total


def score_all(state: GameState) -> Tuple[List[Tuple[int, float]], List[Tuple[str, float]]]:
    """
    Score all stations and lines in the game state.
    
    Returns:
        (station_scores, line_scores) — sorted by score descending
        station_scores: [(station_id, score), ...]
        line_scores: [(line_color, score), ...]
    """
    station_scores = []
    for s in state.stations:
        sc = score_station(s, state.lines, state.stations)
        s.score = sc  # store back on station
        station_scores.append((s.id, sc))

    line_scores = []
    for l in state.lines:
        sc = score_line(l, state.stations)
        l.load_score = sc
        line_scores.append((l.color, sc))

    station_scores.sort(key=lambda x: x[1], reverse=True)
    line_scores.sort(key=lambda x: x[1], reverse=True)

    return station_scores, line_scores


def find_critical_stations(state: GameState) -> List[Station]:
    """Return stations with queue >= critical threshold."""
    return [s for s in state.stations
            if s.queue_size >= config.QUEUE_CRITICAL_THRESHOLD]


def find_warning_stations(state: GameState) -> List[Station]:
    """Return stations with queue >= warning threshold."""
    return [s for s in state.stations
            if s.queue_size >= config.QUEUE_WARN_THRESHOLD]


def find_busiest_line(state: GameState) -> Line:
    """Return the line with the highest load score."""
    if not state.lines:
        return None
    return max(state.lines, key=lambda l: l.load_score)


def find_least_busy_line(state: GameState) -> Line:
    """Return the line with the lowest load score."""
    if not state.lines:
        return None
    return min(state.lines, key=lambda l: l.load_score)
