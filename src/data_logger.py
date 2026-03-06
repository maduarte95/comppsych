"""Data logging for saving task run data."""

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.config import LLMParams, TaskParams
from src.task_engine import TaskEngine, TurnEvent


class DataLogger:
    """Handles saving task run data to CSV and JSON files."""

    def __init__(
        self,
        output_dir: Path | str = "data",
        model: str = "unknown",
        protocol: str = "custom",
    ) -> None:
        """Initialize the data logger.

        Args:
            output_dir: Directory to save output files
            model: Model name for filename
            protocol: Protocol name for filename
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.protocol = protocol
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sanitize model name for filename
        model_safe = model.replace("/", "-").replace(":", "-")
        self.base_filename = f"{model_safe}_{protocol}_{self.timestamp}"

        self.csv_path = self.output_dir / f"{self.base_filename}.csv"
        self.meta_path = self.output_dir / f"{self.base_filename}_meta.json"

        # Track parse errors
        self.parse_errors: list[dict] = []

    def _side_to_int(self, side: str) -> int:
        """Convert side name to int (1=left, 0=right)."""
        return 1 if side == "left" else 0

    def save_turn(
        self,
        event: TurnEvent,
        task_params: TaskParams,
        write_header: bool = False,
    ) -> None:
        """Save a single turn event to CSV.

        Args:
            event: Turn event to save
            task_params: Task parameters for the run
            write_header: Whether to write CSV header
        """
        row = {
            "true_log_time": event.timestamp.isoformat(),
            "prob_flip": task_params.p_switch,
            "prob_rwd": task_params.p_reward,
            "rand_flip": event.rand_switch if event.rand_switch is not None else "",
            "rand_reward": event.rand_reward if event.rand_reward is not None else "",
            "model": self.model,
            "score": event.score,
            "active_side": self._side_to_int(event.active_side),
            "side": self._side_to_int(event.side) if event.side != "center" else -1,
            "active_side_poke": 1 if event.side == event.active_side else 0,
            "reward": 1 if event.reward else 0,
            "protocol": self.protocol,
            "streak": event.streak,
            "turn": event.turn,
            "action": event.action,
            "outcome": event.outcome,
            "monster_life": event.monster_life,
        }

        mode = "w" if write_header else "a"
        with open(self.csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def save_all_turns(
        self,
        events: list[TurnEvent],
        task_params: TaskParams,
    ) -> None:
        """Save all turn events to CSV.

        Args:
            events: List of turn events
            task_params: Task parameters for the run
        """
        if not events:
            return

        for i, event in enumerate(events):
            self.save_turn(event, task_params, write_header=(i == 0))

    def add_parse_error(self, raw_response: str, error_message: str, turn: int) -> None:
        """Record a parse error.

        Args:
            raw_response: Raw LLM response that failed to parse
            error_message: Error description
            turn: Turn number when error occurred
        """
        self.parse_errors.append({
            "turn": turn,
            "raw_response": raw_response[:500],  # Truncate long responses
            "error_message": error_message,
            "timestamp": datetime.now().isoformat(),
        })

    def save_metadata(
        self,
        task_params: TaskParams,
        llm_params: LLMParams,
        engine: TaskEngine,
        outcome: str,
        prompts_used: dict | None = None,
    ) -> None:
        """Save run metadata to JSON.

        Args:
            task_params: Task parameters
            llm_params: LLM parameters
            engine: Task engine (for summary stats)
            outcome: Run outcome ("win", "turn_limit", "error")
            prompts_used: Prompt templates used (optional)
        """
        summary = engine.get_summary_stats()

        metadata = {
            "run_timestamp": self.timestamp,
            "outcome": outcome,
            "task_params": asdict(task_params),
            "llm_params": asdict(llm_params),
            "summary_stats": summary,
            "parse_error_count": len(self.parse_errors),
            "parse_errors": self.parse_errors,
            "prompts_used": prompts_used,
            "files": {
                "csv": str(self.csv_path),
                "metadata": str(self.meta_path),
            },
        }

        with open(self.meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def get_file_paths(self) -> tuple[Path, Path]:
        """Get paths to output files.

        Returns:
            Tuple of (csv_path, metadata_path)
        """
        return self.csv_path, self.meta_path
