"""Standalone visualization script for task run data."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load CSV data from a task run."""
    df = pd.read_csv(csv_path)
    return df


def plot_score(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot score over turns."""
    score_df = df[["turn", "score"]].drop_duplicates()
    ax.plot(score_df["turn"], score_df["score"], color="blue", linewidth=1.5)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Score")
    ax.set_title("Score Over Turns")
    ax.grid(True, alpha=0.3)


def plot_raster(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot task raster with monster location, witch position, and attack outcomes."""
    position_df = df[df["side"] >= 0][["turn", "side", "active_side"]].drop_duplicates(
        subset=["turn"]
    )
    fire_df = df[df["action"] == "fire"][["turn", "side", "reward"]].copy()

    if position_df.empty:
        ax.text(0.5, 0.5, "No position data", ha="center", va="center")
        return

    turns = position_df["turn"].values
    player = position_df["side"].values
    monster = position_df["active_side"].values

    # Shaded background for monster location
    # Blue when LEFT (1), Red when RIGHT (0)
    for i in range(len(turns) - 1):
        color = "blue" if monster[i] == 1 else "red"
        ax.axvspan(turns[i], turns[i + 1], alpha=0.2, color=color)
    # Last segment
    if len(turns) > 0:
        color = "blue" if monster[-1] == 1 else "red"
        ax.axvspan(turns[-1], turns[-1] + 1, alpha=0.2, color=color)

    # Black line for witch position
    ax.step(turns, player, where="post", color="black", linewidth=1.5)

    # Raster ticks for attack outcomes
    for _, row in fire_df.iterrows():
        turn = row["turn"]
        side = row["side"]
        reward = row["reward"]
        color = "green" if reward == 1 else "grey"
        y_pos = 1.08 if side == 1 else -0.08
        ax.plot(turn, y_pos, marker="|", markersize=8, color=color, markeredgewidth=1.5)

    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["RIGHT", "LEFT"])
    ax.set_xlabel("Turn")
    ax.set_ylabel("Position")
    ax.set_title("Task Raster Plot")

    # Custom legend
    legend_elements = [
        Patch(facecolor="blue", alpha=0.2, label="Monster LEFT"),
        Patch(facecolor="red", alpha=0.2, label="Monster RIGHT"),
        Line2D([0], [0], color="black", linewidth=1.5, label="Witch Position"),
        Line2D(
            [0],
            [0],
            marker="|",
            color="green",
            linestyle="None",
            markersize=8,
            markeredgewidth=1.5,
            label="Hit",
        ),
        Line2D(
            [0],
            [0],
            marker="|",
            color="grey",
            linestyle="None",
            markersize=8,
            markeredgewidth=1.5,
            label="Miss",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)


def plot_summary(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot summary statistics as text."""
    ax.axis("off")

    # Calculate stats
    fire_df = df[df["action"] == "fire"]
    total_turns = df["turn"].max()
    total_fires = len(fire_df)
    total_hits = fire_df["reward"].sum()
    hit_rate = total_hits / total_fires if total_fires > 0 else 0
    final_score = df["score"].max()

    switch_df = df[df["outcome"].isin(["spotted", "not_spotted"])]
    total_switches = len(switch_df)
    correct_switches = len(switch_df[switch_df["outcome"] == "spotted"])

    text = f"""Summary Statistics
─────────────────
Total Turns: {total_turns}
Final Score: {final_score}

Attacks:
  Total Fires: {total_fires}
  Hits: {total_hits}
  Hit Rate: {hit_rate:.1%}

Switches:
  Total: {total_switches}
  Correct: {correct_switches}
  Incorrect: {total_switches - correct_switches}
"""
    ax.text(
        0.1, 0.9, text, transform=ax.transAxes, fontsize=10, verticalalignment="top",
        fontfamily="monospace"
    )


def create_visualization(
    csv_path: str | Path, output_path: str | Path | None = None, show: bool = True
) -> None:
    """Create full visualization from a task run CSV.

    Args:
        csv_path: Path to the CSV file
        output_path: Optional path to save the figure (PNG, PDF, etc.)
        show: Whether to display the figure
    """
    df = load_data(csv_path)

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 8))

    # Score plot (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    plot_score(df, ax1)

    # Summary stats (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    plot_summary(df, ax2)

    # Raster plot (bottom, full width)
    ax3 = fig.add_subplot(2, 1, 2)
    plot_raster(df, ax3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize task run data from CSV files"
    )
    parser.add_argument("csv_path", type=str, help="Path to the CSV file")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output path for the figure"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Don't display the figure"
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    create_visualization(csv_path, output_path=args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
