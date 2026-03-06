"""Prompt builder for dynamic prompt generation."""

from pathlib import Path

import yaml

from src.config import TaskParams
from src.task_engine import TaskEngine, TurnEvent


class PromptBuilder:
    """Builds dynamic prompts for the LLM based on game state."""

    def __init__(self, template_path: Path | str | None = None) -> None:
        """Initialize the prompt builder.

        Args:
            template_path: Path to YAML template file. If None, uses default.
        """
        if template_path is None:
            template_path = Path(__file__).parent.parent / "prompts" / "default.yaml"
        else:
            template_path = Path(template_path)

        with open(template_path) as f:
            self.templates = yaml.safe_load(f)

        self.system_template = self.templates["system_prompt"]
        self.user_template = self.templates["user_prompt"]

    def build_system_prompt(self) -> str:
        """Build the system prompt."""
        return self.system_template

    def build_user_prompt(self, engine: TaskEngine) -> str:
        """Build the user prompt based on current game state.

        Args:
            engine: TaskEngine instance with current state

        Returns:
            Formatted user prompt
        """
        state = engine.get_state()
        params = engine.params

        # Build dynamic sections
        history_block = self._build_history_block(engine.history)
        location_line = self._build_location_line(state["current_side"])
        decision_block = self._build_decision_block(state)

        # Substitute template variables
        prompt = self.user_template.format(
            points_per_hit=params.points_per_hit,
            travel_cost=params.travel_cost,
            max_turns=params.max_turns,
            max_points=params.max_points,
            current_life=state["monster_life"],
            current_turn=state["current_turn"],
            history_block=history_block,
            location_line=location_line,
            decision_block=decision_block,
        )

        return prompt

    def _build_history_block(self, history: list[TurnEvent]) -> str:
        """Build the game history block.

        Args:
            history: List of turn events

        Returns:
            Formatted history string
        """
        if not history:
            return "=== GAME LOG ===\nNo actions yet. This is your first turn!"

        lines = ["=== GAME LOG ==="]

        for event in history:
            line = self._format_event(event)
            if line:  # Skip empty lines (traveling mid-turns)
                lines.append(line)

        return "\n".join(lines)

    def _format_event(self, event: TurnEvent) -> str:
        """Format a single turn event.

        Args:
            event: Turn event to format

        Returns:
            Formatted string for the event
        """
        side_name = event.side.upper() if event.side != "center" else "CENTER"

        if event.action == "travel":
            if event.outcome == "spotted":
                return f"Turn {event.turn}: Traveled to {side_name} tower. Monster spotted!"
            elif event.outcome == "not_spotted":
                return f"Turn {event.turn}: Traveled to {side_name} tower. Monster not spotted! You must go back."
            elif event.outcome == "return":
                return f"Turn {event.turn}: Traveled back to {side_name} tower."
            elif event.outcome == "traveling":
                # Mid-travel turn, don't show (or show a brief indicator)
                return ""
        elif event.action == "fire":
            if event.outcome == "hit":
                return (
                    f"Turn {event.turn}: Fired at {side_name} tower. "
                    f"Monster appeared — hit! (Monster life: {event.monster_life}/{event.monster_life + 3})"
                )
            elif event.outcome == "miss":
                return f"Turn {event.turn}: Fired at {side_name} tower. Nothing happened."

        return ""

    def _build_location_line(self, current_side: str) -> str:
        """Build the location description line.

        Args:
            current_side: Current side (left, right, or center)

        Returns:
            Location description string
        """
        if current_side == "center":
            return "You are at the center of the castle."
        else:
            return f"You are at the {current_side.upper()} tower."

    def _build_decision_block(self, state: dict) -> str:
        """Build the decision options block.

        Args:
            state: Current game state dict

        Returns:
            Decision options string
        """
        turn = state["current_turn"]

        if state["is_first_turn"]:
            return (
                f"What do you do? (Turn {turn})\n"
                "A: Go to the LEFT tower\n"
                "B: Go to the RIGHT tower"
            )
        else:
            current = state["current_side"].upper()
            other = "LEFT" if state["current_side"] == "right" else "RIGHT"
            return (
                f"What do you do? (Turn {turn})\n"
                f"A: Fire again at the {current} tower\n"
                f"B: Travel to the {other} tower"
            )


def load_prompt_templates(template_path: Path | str | None = None) -> dict:
    """Load prompt templates from YAML file.

    Args:
        template_path: Path to YAML template file

    Returns:
        Dictionary with 'system_prompt' and 'user_prompt' keys
    """
    if template_path is None:
        template_path = Path(__file__).parent.parent / "prompts" / "default.yaml"
    else:
        template_path = Path(template_path)

    with open(template_path) as f:
        return yaml.safe_load(f)
