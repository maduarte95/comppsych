"""Core task engine implementing the Witch Foraging Task state machine."""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from src.config import Side, TaskParams


@dataclass
class TurnEvent:
    """Record of a single turn event."""

    turn: int
    action: Literal["travel", "fire"]
    side: Side
    active_side: Side
    outcome: str  # "hit", "miss", "spotted", "not_spotted", "return", "arrived", "traveling"
    reward: bool
    rand_reward: float | None  # Random number used for reward roll
    rand_switch: float | None  # Random number used for switch roll
    monster_life: int
    score: int
    streak: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TurnResult:
    """Result of processing a choice."""

    events: list[TurnEvent]
    game_over: bool
    game_over_reason: str | None  # "win", "turn_limit", None


class TaskEngine:
    """State machine for the Witch Foraging Task."""

    def __init__(self, params: TaskParams, seed: int | None = None) -> None:
        """Initialize the task engine.

        Args:
            params: Task parameters
            seed: Random seed for reproducibility (optional)
        """
        self.params = params
        self.rng = random.Random(seed)

        # State
        self.current_turn: int = 1
        self.monster_life: int = params.max_points
        self.current_side: Side = "center"
        self.active_side: Side = self.rng.choice(["left", "right"])
        self.history: list[TurnEvent] = []
        self.streak: int = 0  # Current poke streak (resets on switch)
        self.bout_rewards: int = 0  # Rewards in current bout

    @property
    def score(self) -> int:
        """Current score (damage dealt to monster)."""
        return self.params.max_points - self.monster_life

    def get_state(self) -> dict:
        """Return current game state as a dictionary."""
        return {
            "current_turn": self.current_turn,
            "monster_life": self.monster_life,
            "max_points": self.params.max_points,
            "current_side": self.current_side,
            "active_side": self.active_side,
            "score": self.score,
            "streak": self.streak,
            "is_first_turn": self.current_side == "center",
            "history": self.history,
        }

    def is_game_over(self) -> tuple[bool, str | None]:
        """Check if game is over.

        Returns:
            Tuple of (is_over, reason). Reason is "win" or "turn_limit" or None.
        """
        if self.monster_life <= 0:
            return True, "win"
        if self.current_turn > self.params.max_turns:
            return True, "turn_limit"
        return False, None

    def _roll_reward(self) -> tuple[bool, float]:
        """Roll for reward.

        Returns:
            Tuple of (got_reward, random_value_used)
        """
        rand_val = self.rng.random()
        return rand_val < self.params.p_reward, rand_val

    def _roll_switch(self) -> tuple[bool, float]:
        """Roll for monster switch.

        Returns:
            Tuple of (did_switch, random_value_used)
        """
        rand_val = self.rng.random()
        return rand_val < self.params.p_switch, rand_val

    def _other_side(self, side: Side) -> Side:
        """Get the opposite side."""
        if side == "left":
            return "right"
        elif side == "right":
            return "left"
        else:
            raise ValueError("Cannot get other side from center")

    def process_first_turn(self, choice: str) -> TurnResult:
        """Handle the first turn from center.

        Args:
            choice: "A" for left, "B" for right

        Returns:
            TurnResult with events
        """
        events = []

        # Determine chosen side
        chosen_side: Side = "left" if choice.upper() == "A" else "right"

        # First move always costs 1 turn (symmetric start)
        is_correct = chosen_side == self.active_side

        if not self.params.immediate_feedback:
            # No feedback mode: agent arrives at chosen tower with no location info
            event = TurnEvent(
                turn=self.current_turn,
                action="travel",
                side=chosen_side,
                active_side=self.active_side,
                outcome="arrived",
                reward=False,
                rand_reward=None,
                rand_switch=None,
                monster_life=self.monster_life,
                score=self.score,
                streak=0,
            )
            events.append(event)
            self.history.append(event)
            self.current_turn += 1
            self.current_side = chosen_side
            self.streak = 0
            self.bout_rewards = 0
            game_over, reason = self.is_game_over()
            return TurnResult(events=events, game_over=game_over, game_over_reason=reason)

        if is_correct:
            # Correct choice - monster spotted
            event = TurnEvent(
                turn=self.current_turn,
                action="travel",
                side=chosen_side,
                active_side=self.active_side,
                outcome="spotted",
                reward=False,
                rand_reward=None,
                rand_switch=None,
                monster_life=self.monster_life,
                score=self.score,
                streak=0,
            )
            events.append(event)
            self.history.append(event)
            self.current_turn += 1
            self.current_side = chosen_side
            self.streak = 0
            self.bout_rewards = 0
        else:
            # Incorrect choice - forced return
            # First: travel to wrong tower
            event1 = TurnEvent(
                turn=self.current_turn,
                action="travel",
                side=chosen_side,
                active_side=self.active_side,
                outcome="not_spotted",
                reward=False,
                rand_reward=None,
                rand_switch=None,
                monster_life=self.monster_life,
                score=self.score,
                streak=0,
            )
            events.append(event1)
            self.history.append(event1)
            self.current_turn += 1

            # Second: forced return to correct tower
            correct_side = self._other_side(chosen_side)
            event2 = TurnEvent(
                turn=self.current_turn,
                action="travel",
                side=correct_side,
                active_side=self.active_side,
                outcome="return",
                reward=False,
                rand_reward=None,
                rand_switch=None,
                monster_life=self.monster_life,
                score=self.score,
                streak=0,
            )
            events.append(event2)
            self.history.append(event2)
            self.current_turn += 1
            self.current_side = correct_side
            self.streak = 0
            self.bout_rewards = 0

        game_over, reason = self.is_game_over()
        return TurnResult(events=events, game_over=game_over, game_over_reason=reason)

    def process_fire(self) -> TurnResult:
        """Handle a fire action at current tower.

        Returns:
            TurnResult with events
        """
        events = []

        is_active = self.current_side == self.active_side

        if is_active:
            # At active tower - roll for reward
            got_reward, rand_reward = self._roll_reward()

            if got_reward:
                self.monster_life -= self.params.points_per_hit
                self.bout_rewards += 1
                outcome = "hit"
            else:
                outcome = "miss"

            event = TurnEvent(
                turn=self.current_turn,
                action="fire",
                side=self.current_side,
                active_side=self.active_side,
                outcome=outcome,
                reward=got_reward,
                rand_reward=rand_reward,
                rand_switch=None,
                monster_life=self.monster_life,
                score=self.score,
                streak=self.streak,
            )
            events.append(event)
            self.history.append(event)

            # Roll for monster switch after fire
            did_switch, rand_switch = self._roll_switch()
            # Update the last event with switch info
            event.rand_switch = rand_switch

            if did_switch:
                self.active_side = self._other_side(self.active_side)

        else:
            # At inactive tower - always miss
            event = TurnEvent(
                turn=self.current_turn,
                action="fire",
                side=self.current_side,
                active_side=self.active_side,
                outcome="miss",
                reward=False,
                rand_reward=None,
                rand_switch=None,
                monster_life=self.monster_life,
                score=self.score,
                streak=self.streak,
            )
            events.append(event)
            self.history.append(event)
            # No switch roll when at inactive tower (based on paper mechanics)

        self.current_turn += 1
        self.streak += 1

        game_over, reason = self.is_game_over()
        return TurnResult(events=events, game_over=game_over, game_over_reason=reason)

    def process_switch(self) -> TurnResult:
        """Handle a switch action to the other tower.

        Returns:
            TurnResult with events
        """
        events = []

        target_side = self._other_side(self.current_side)
        is_correct = target_side == self.active_side

        if not self.params.immediate_feedback:
            # No feedback mode: agent travels to target with no location info, no forced return
            if self.params.travel_cost == 0:
                event = TurnEvent(
                    turn=self.current_turn,
                    action="travel",
                    side=target_side,
                    active_side=self.active_side,
                    outcome="arrived",
                    reward=False,
                    rand_reward=None,
                    rand_switch=None,
                    monster_life=self.monster_life,
                    score=self.score,
                    streak=self.streak,
                )
                events.append(event)
                self.history.append(event)
                self.current_turn += 1
            else:
                for i in range(self.params.travel_cost):
                    outcome = "arrived" if i == self.params.travel_cost - 1 else "traveling"
                    event = TurnEvent(
                        turn=self.current_turn,
                        action="travel",
                        side=target_side if i == self.params.travel_cost - 1 else self.current_side,
                        active_side=self.active_side,
                        outcome=outcome,
                        reward=False,
                        rand_reward=None,
                        rand_switch=None,
                        monster_life=self.monster_life,
                        score=self.score,
                        streak=self.streak,
                    )
                    events.append(event)
                    self.history.append(event)
                    self.current_turn += 1

                    if self.current_turn > self.params.max_turns:
                        game_over, reason = self.is_game_over()
                        return TurnResult(events=events, game_over=game_over, game_over_reason=reason)

            self.current_side = target_side
            self.streak = 0
            self.bout_rewards = 0
            game_over, reason = self.is_game_over()
            return TurnResult(events=events, game_over=game_over, game_over_reason=reason)

        if self.params.travel_cost == 0:
            # Instant switch: no travel turns, but still costs 1 turn to prevent infinite loops
            outcome = "spotted" if is_correct else "not_spotted"
            event = TurnEvent(
                turn=self.current_turn,
                action="travel",
                side=target_side,
                active_side=self.active_side,
                outcome=outcome,
                reward=False,
                rand_reward=None,
                rand_switch=None,
                monster_life=self.monster_life,
                score=self.score,
                streak=self.streak,
            )
            events.append(event)
            self.history.append(event)
            self.current_turn += 1

            if is_correct:
                self.current_side = target_side
                self.streak = 0
                self.bout_rewards = 0
            # If incorrect, player stays put (no forced return with 0 cost)

            game_over, reason = self.is_game_over()
            return TurnResult(events=events, game_over=game_over, game_over_reason=reason)

        # Spend travel_cost turns
        for i in range(self.params.travel_cost):
            if i == self.params.travel_cost - 1:
                # Final travel turn - we arrive
                if is_correct:
                    outcome = "spotted"
                else:
                    outcome = "not_spotted"
            else:
                outcome = "traveling"

            event = TurnEvent(
                turn=self.current_turn,
                action="travel",
                side=target_side if i == self.params.travel_cost - 1 else self.current_side,
                active_side=self.active_side,
                outcome=outcome,
                reward=False,
                rand_reward=None,
                rand_switch=None,
                monster_life=self.monster_life,
                score=self.score,
                streak=self.streak,
            )
            events.append(event)
            self.history.append(event)
            self.current_turn += 1

            # Check turn limit during travel
            if self.current_turn > self.params.max_turns:
                game_over, reason = self.is_game_over()
                return TurnResult(
                    events=events, game_over=game_over, game_over_reason=reason
                )

        if is_correct:
            # Arrived at correct tower
            self.current_side = target_side
            self.streak = 0
            self.bout_rewards = 0
        else:
            # Incorrect switch - forced return
            # Spend travel_cost turns returning
            original_side = self.current_side
            for i in range(self.params.travel_cost):
                if i == self.params.travel_cost - 1:
                    outcome = "return"
                else:
                    outcome = "traveling"

                event = TurnEvent(
                    turn=self.current_turn,
                    action="travel",
                    side=original_side if i == self.params.travel_cost - 1 else target_side,
                    active_side=self.active_side,
                    outcome=outcome,
                    reward=False,
                    rand_reward=None,
                    rand_switch=None,
                    monster_life=self.monster_life,
                    score=self.score,
                    streak=self.streak,
                )
                events.append(event)
                self.history.append(event)
                self.current_turn += 1

                # Check turn limit during return travel
                if self.current_turn > self.params.max_turns:
                    game_over, reason = self.is_game_over()
                    return TurnResult(
                        events=events, game_over=game_over, game_over_reason=reason
                    )

            # Back at original tower, streak continues
            self.current_side = original_side

        game_over, reason = self.is_game_over()
        return TurnResult(events=events, game_over=game_over, game_over_reason=reason)

    def process_choice(self, choice: str) -> TurnResult:
        """Process player choice.

        Args:
            choice: "A" or "B"
                - First turn: A=left, B=right
                - Normal play: A=fire, B=switch

        Returns:
            TurnResult with events
        """
        choice = choice.upper()
        if choice not in ("A", "B"):
            raise ValueError(f"Invalid choice: {choice}. Must be 'A' or 'B'.")

        if self.current_side == "center":
            return self.process_first_turn(choice)
        elif choice == "A":
            return self.process_fire()
        else:
            return self.process_switch()

    def get_summary_stats(self) -> dict:
        """Calculate summary statistics for the completed game."""
        fire_events = [e for e in self.history if e.action == "fire"]
        switch_events = [
            e
            for e in self.history
            if e.action == "travel" and e.outcome in ("spotted", "not_spotted", "arrived")
        ]

        hits = sum(1 for e in fire_events if e.reward)
        total_fires = len(fire_events)

        return {
            "total_turns": self.current_turn - 1,
            "final_score": self.score,
            "monster_life_remaining": self.monster_life,
            "total_hits": hits,
            "total_fires": total_fires,
            "hit_rate": hits / total_fires if total_fires > 0 else 0,
            "total_switches": len(switch_events),
            "correct_switches": sum(1 for e in switch_events if e.side == e.active_side),
            "incorrect_switches": sum(1 for e in switch_events if e.side != e.active_side),
        }
