"""
Mini Metro AI — Planner Unit Tests

Tests the planner with mock GameState objects to verify
correct action selection for each scenario.

Usage:
    python -m pytest tests/test_planner.py -v
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import GameState, Station, Line, Resources
from engine.planner import Planner
from engine.scorer import score_all


def make_state(
    stations=None, lines=None, resources=None,
    is_paused=False, is_game_over=False, unconnected=None,
):
    """Helper to create a GameState for testing."""
    state = GameState(
        stations=stations or [],
        lines=lines or [],
        resources=resources or Resources(),
        is_paused=is_paused,
        is_game_over=is_game_over,
        frame_id=1,
        timestamp=0,
        station_graph={},
        unconnected_stations=unconnected or [],
    )
    return state


def test_noop_on_empty_state():
    """When there's nothing to do, planner returns noop."""
    planner = Planner()
    state = make_state()
    actions = planner.plan(state, "STABLE")
    assert len(actions) >= 1
    assert actions[0].type == "noop"


def test_noop_on_game_over():
    """Game over → noop."""
    planner = Planner()
    state = make_state(is_game_over=True)
    actions = planner.plan(state, "STABLE")
    assert actions[0].type == "noop"


def test_connect_unconnected_station():
    """Unconnected station should trigger connect/extend action."""
    planner = Planner()

    s1 = Station(id=0, x=0.2, y=0.5, px=200, py=500,
                 shape="circle", queue_size=0, connected_lines=["red"])
    s2 = Station(id=1, x=0.8, y=0.5, px=800, py=500,
                 shape="triangle", queue_size=0, connected_lines=[])

    line = Line(color="red", station_ids=[0])

    state = make_state(
        stations=[s1, s2],
        lines=[line],
        unconnected=[1],
    )

    actions = planner.plan(state, "STABLE")
    connect_actions = [a for a in actions if a.type in ("connect_station", "extend_line")]
    assert len(connect_actions) >= 1
    assert connect_actions[0].params["station_id"] == 1


def test_critical_station_triggers_restructure():
    """Station with queue >= critical threshold should trigger action."""
    planner = Planner()

    s1 = Station(id=0, x=0.2, y=0.5, px=200, py=500,
                 shape="circle", queue_size=8, connected_lines=["red"])
    s2 = Station(id=1, x=0.5, y=0.5, px=500, py=500,
                 shape="triangle", queue_size=0, connected_lines=["red", "blue"])

    red_line = Line(color="red", station_ids=[0, 1])
    blue_line = Line(color="blue", station_ids=[1])

    state = make_state(
        stations=[s1, s2],
        lines=[red_line, blue_line],
    )

    score_all(state)
    actions = planner.plan(state, "CRITICAL")

    # Should want to extend another line to the critical station
    extend_actions = [a for a in actions if a.type == "extend_line"]
    assert len(extend_actions) >= 1


def test_add_train_when_available():
    """Spare trains + overloaded line → add train action."""
    planner = Planner()

    s1 = Station(id=0, x=0.2, y=0.5, px=200, py=500,
                 shape="circle", queue_size=4, connected_lines=["red"])
    s2 = Station(id=1, x=0.5, y=0.5, px=500, py=500,
                 shape="triangle", queue_size=4, connected_lines=["red"])

    red_line = Line(color="red", station_ids=[0, 1], load_score=8)

    state = make_state(
        stations=[s1, s2],
        lines=[red_line],
        resources=Resources(spare_trains=2),
    )

    score_all(state)
    actions = planner.plan(state, "STABLE")

    train_actions = [a for a in actions if a.type == "add_train"]
    assert len(train_actions) >= 1
    assert train_actions[0].params["line_color"] == "red"


def test_add_carriage_to_busiest_line():
    """Spare carriages → add to busiest line."""
    planner = Planner()

    s1 = Station(id=0, x=0.2, y=0.5, px=200, py=500,
                 shape="circle", queue_size=2, connected_lines=["red"])
    s2 = Station(id=1, x=0.5, y=0.5, px=500, py=500,
                 shape="triangle", queue_size=5, connected_lines=["blue"])

    red_line = Line(color="red", station_ids=[0])
    blue_line = Line(color="blue", station_ids=[1])

    state = make_state(
        stations=[s1, s2],
        lines=[red_line, blue_line],
        resources=Resources(spare_carriages=1),
    )

    score_all(state)
    actions = planner.plan(state, "STABLE")

    carriage_actions = [a for a in actions if a.type == "add_carriage"]
    assert len(carriage_actions) >= 1
    assert carriage_actions[0].params["line_color"] == "blue"


def test_first_line_creation():
    """Two unconnected stations with no existing lines → create first line."""
    planner = Planner()

    s1 = Station(id=0, x=0.2, y=0.5, px=200, py=500,
                 shape="circle", queue_size=0, connected_lines=[])
    s2 = Station(id=1, x=0.8, y=0.5, px=800, py=500,
                 shape="triangle", queue_size=0, connected_lines=[])

    state = make_state(
        stations=[s1, s2],
        lines=[],
        unconnected=[0, 1],
    )

    actions = planner.plan(state, "STABLE")
    connect_actions = [a for a in actions
                       if a.type in ("connect_station", "extend_line")]
    assert len(connect_actions) >= 1


# =============================================================================
# Run tests directly
# =============================================================================

if __name__ == "__main__":
    tests = [
        test_noop_on_empty_state,
        test_noop_on_game_over,
        test_connect_unconnected_station,
        test_critical_station_triggers_restructure,
        test_add_train_when_available,
        test_add_carriage_to_busiest_line,
        test_first_line_creation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  FAIL {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
