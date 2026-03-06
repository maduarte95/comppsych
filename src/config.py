"""Configuration and data classes for the LLM Foraging Task."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TaskParams:
    """Parameters controlling task behavior."""

    p_reward: float  # Probability of hit when firing at active tower
    p_switch: float  # Probability monster switches after each fire
    travel_cost: int  # Turns spent traveling between towers
    max_turns: int  # Turn budget
    max_points: int  # Monster starting life (goal: reduce to 0)
    points_per_hit: int  # Damage per successful hit

    def __post_init__(self) -> None:
        """Validate parameters."""
        if not 0 <= self.p_reward <= 1:
            raise ValueError(f"p_reward must be between 0 and 1, got {self.p_reward}")
        if not 0 <= self.p_switch <= 1:
            raise ValueError(f"p_switch must be between 0 and 1, got {self.p_switch}")
        if self.travel_cost < 1:
            raise ValueError(f"travel_cost must be >= 1, got {self.travel_cost}")
        if self.max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {self.max_turns}")
        if self.max_points < 1:
            raise ValueError(f"max_points must be >= 1, got {self.max_points}")
        if self.points_per_hit < 1:
            raise ValueError(f"points_per_hit must be >= 1, got {self.points_per_hit}")


@dataclass
class LLMParams:
    """Parameters for LLM configuration."""

    provider: Literal["anthropic", "openrouter"]
    model: str
    temperature: float = 0.0
    max_tokens: int = 64

    def __post_init__(self) -> None:
        """Validate parameters."""
        if not 0 <= self.temperature <= 1:
            raise ValueError(
                f"temperature must be between 0 and 1, got {self.temperature}"
            )
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")


# Side type for type hints
Side = Literal["left", "right", "center"]
