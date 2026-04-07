"""Predefined task protocols based on Vertechi et al."""

from src.config import TaskParams

# Default max_turns value (can be adjusted after piloting)
DEFAULT_MAX_TURNS = 300

# Predefined protocols from the paper
PROTOCOLS: dict[str, TaskParams] = {
    "easy": TaskParams(
        p_reward=0.9,
        p_switch=0.9,
        travel_cost=2,
        max_turns=DEFAULT_MAX_TURNS,
        max_points=250,
        points_per_hit=3,
    ),
    "medium": TaskParams(
        p_reward=0.9,
        p_switch=0.3,
        travel_cost=2,
        max_turns=DEFAULT_MAX_TURNS,
        max_points=250,
        points_per_hit=3,
    ),
    "hard": TaskParams(
        p_reward=0.3,
        p_switch=0.3,
        travel_cost=2,
        max_turns=DEFAULT_MAX_TURNS,
        max_points=250,
        points_per_hit=3,
    ),
    "recent": TaskParams(
        p_reward=0.45,
        p_switch=0.15,
        travel_cost=2,
        max_turns=DEFAULT_MAX_TURNS,
        max_points=1000,
        points_per_hit=4,
    ),
    "human": TaskParams(
        p_reward=0.45,
        p_switch=0.15,
        travel_cost=0,
        max_turns=DEFAULT_MAX_TURNS,
        max_points=1000,
        points_per_hit=4,
    ),
    "human_no_feedback": TaskParams(
        p_reward=0.45,
        p_switch=0.15,
        travel_cost=0,
        max_turns=DEFAULT_MAX_TURNS,
        max_points=1000,
        points_per_hit=4,
        immediate_feedback=False,
    ),
    "protocol-1": TaskParams(
        p_reward=0.45,
        p_switch=0.15,
        travel_cost=0,
        max_turns=500,
        max_points=400,
        points_per_hit=10,
        immediate_feedback=False,
    ),
}


def get_protocol(name: str) -> TaskParams:
    """Get a predefined protocol by name.

    Args:
        name: Protocol name (easy, medium, hard, recent)

    Returns:
        TaskParams for the protocol

    Raises:
        KeyError: If protocol name not found
    """
    name_lower = name.lower()
    if name_lower not in PROTOCOLS:
        raise KeyError(
            f"Unknown protocol '{name}'. Available: {list(PROTOCOLS.keys())}"
        )
    return PROTOCOLS[name_lower]


def list_protocols() -> list[str]:
    """Return list of available protocol names."""
    return list(PROTOCOLS.keys())
