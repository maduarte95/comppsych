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
    immediate_feedback: bool = True  # Whether agent is told if monster is at arrived tower

    def __post_init__(self) -> None:
        """Validate parameters."""
        if not 0 <= self.p_reward <= 1:
            raise ValueError(f"p_reward must be between 0 and 1, got {self.p_reward}")
        if not 0 <= self.p_switch <= 1:
            raise ValueError(f"p_switch must be between 0 and 1, got {self.p_switch}")
        if self.travel_cost < 0:
            raise ValueError(f"travel_cost must be >= 0, got {self.travel_cost}")
        if self.max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {self.max_turns}")
        if self.max_points < 1:
            raise ValueError(f"max_points must be >= 1, got {self.max_points}")
        if self.points_per_hit < 1:
            raise ValueError(f"points_per_hit must be >= 1, got {self.points_per_hit}")


# Thinking mode types
ThinkingMode = Literal["disabled", "enabled", "adaptive"]


@dataclass
class LLMParams:
    """Parameters for LLM configuration."""

    provider: Literal["anthropic", "openrouter"]
    model: str
    temperature: float = 0.0
    max_tokens: int = 64
    thinking_mode: ThinkingMode = "disabled"
    thinking_budget: int = 10000  # For "enabled" mode on older models

    def __post_init__(self) -> None:
        """Validate parameters."""
        if not 0 <= self.temperature <= 1:
            raise ValueError(
                f"temperature must be between 0 and 1, got {self.temperature}"
            )
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.thinking_mode not in ("disabled", "enabled", "adaptive"):
            raise ValueError(
                f"thinking_mode must be 'disabled', 'enabled', or 'adaptive', "
                f"got {self.thinking_mode}"
            )
        if self.thinking_budget < 1:
            raise ValueError(
                f"thinking_budget must be >= 1, got {self.thinking_budget}"
            )


# Side type for type hints
Side = Literal["left", "right", "center"]
