"""Plot individual-behavior figures for all Qwen run configs.

Reproduces the two figures in `human_data/`:
  - `individualbehavior_example.png`  — mean consecutive failures before leaving
    per bout (x=streak) with error-shaded area.
  - `individualbehavior_example2.png` — one line per individual (per game),
    consecutive failures before leaving per bout.

Both figures are rendered as a 2x3 grid over the six Qwen configurations:
  rows: reasoning / no_reasoning
  cols: maxturns / nomaxturns / nomaxturns_strict

Bout-level statistics follow the Vertechi convention used in
`figure4a.ipynb`: trailing rewards are stripped from each bout, then
trailing consecutive failures are counted; all-reward bouts and the final
(terminal) bout of each game are dropped.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "qwen"
FIGS_DIR = REPO_ROOT / "analysis" / "figs"

CONFIGS = [
    ("reasoning",    "qwen_reasoning_maxturns",           "Reasoning | max turns",           "qwen_reasoning_max"),
    ("reasoning",    "qwen_reasoning_nomaxturns",         "Reasoning | no max turns",        "qwen_reasoning_no_max"),
    ("reasoning",    "qwen_reasoning_nomaxturns_strict",  "Reasoning | no max turns, strict","qwen_reasoning_no_max_strict"),
    ("no_reasoning", "qwen_noreasoning_maxturns",         "No-reasoning | max turns",           "qwen_no_reasoning_max"),
    ("no_reasoning", "qwen_noreasoning_nomaxturns",       "No-reasoning | no max turns",        "qwen_no_reasoning_no_max"),
    ("no_reasoning", "qwen_noreasoning_nomaxturns_strict","No-reasoning | no max turns, strict","qwen_no_reasoning_no_max_strict"),
]


def compute_bout_stats(csv_path: str | Path) -> pd.DataFrame:
    """Return one row per bout: streak, rewards_within_streak, consec_fail_before_leaving.

    `streak` is 1-indexed over the bouts that survive filtering (dropping
    the terminal bout and all-reward bouts), matching the x-axis used in
    the human example figures.
    """
    df = pd.read_csv(csv_path)
    fires = df[df["action"] == "fire"].copy()
    if fires.empty:
        return pd.DataFrame(columns=["streak", "rewards_within_streak", "consec_fail_before_leaving"])

    last_side = None
    bout_id = 0
    bout_ids = []
    for _, row in fires.iterrows():
        if last_side is not None and row["side"] != last_side:
            bout_id += 1
        last_side = row["side"]
        bout_ids.append(bout_id)
    fires["bout_id"] = bout_ids

    max_bout = fires["bout_id"].max()
    bouts = []
    for bid, bout_df in fires.groupby("bout_id"):
        if bid == max_bout:
            continue
        rewards = int(bout_df["reward"].sum())
        outcomes = list(bout_df["reward"].values)
        while outcomes and outcomes[-1] == 1:
            outcomes.pop()
        if not outcomes:
            continue
        consec_fail = 0
        for v in reversed(outcomes):
            if v == 0:
                consec_fail += 1
            else:
                break
        bouts.append({
            "rewards_within_streak": rewards,
            "consec_fail_before_leaving": consec_fail,
        })

    out = pd.DataFrame(bouts)
    if not out.empty:
        out.insert(0, "streak", range(1, len(out) + 1))
    return out


def load_config(subdir: Path) -> pd.DataFrame:
    """Load every game CSV in `subdir` and concatenate bout stats with `game` and `game_stem` columns."""
    csv_files = sorted(glob.glob(str(subdir / "*.csv")))
    if not csv_files:
        return pd.DataFrame(columns=["game", "game_stem", "streak", "rewards_within_streak", "consec_fail_before_leaving"])
    pieces = []
    for i, f in enumerate(csv_files):
        bouts = compute_bout_stats(f)
        bouts["game"] = i
        bouts["game_stem"] = Path(f).stem
        pieces.append(bouts)
    return pd.concat(pieces, ignore_index=True)


def plot_mean_per_streak(ax: plt.Axes, df: pd.DataFrame, title: str, error: str) -> None:
    """Mean consec_fail_before_leaving per streak, with SEM or SD shaded band."""
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        return

    grouped = df.groupby("streak")["consec_fail_before_leaving"]
    mean = grouped.mean()
    if error == "sem":
        err = grouped.sem()
        err_label = "SEM"
    else:
        err = grouped.std()
        err_label = "SD"

    ax.fill_between(mean.index, (mean - err).values, (mean + err).values,
                    alpha=0.25, color="tab:blue", linewidth=0)
    ax.plot(mean.index, mean.values, "-", color="tab:blue", linewidth=2,
            label=f"mean ± {err_label}")

    n_games = df["game"].nunique()
    ax.set_title(f"{title}\nN={n_games} games, total bouts={len(df)}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def plot_individual_lines(ax: plt.Axes, df: pd.DataFrame, title: str,
                          n_lines: int, seed: int) -> None:
    """One line per game (up to `n_lines`): consec_fail_before_leaving vs. streak."""
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        return

    games_all = sorted(df["game"].unique())
    rng = np.random.default_rng(seed)
    if len(games_all) > n_lines:
        games = sorted(rng.choice(games_all, size=n_lines, replace=False).tolist())
    else:
        games = games_all

    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(games):
        sub = df[df["game"] == g].sort_values("streak")
        ax.plot(sub["streak"], sub["consec_fail_before_leaving"],
                "-o", color=cmap(i % cmap.N), linewidth=1.2, markersize=3,
                alpha=0.9, label=f"game {g}")

    ax.set_title(f"{title}\nshowing {len(games)} of {len(games_all)} games",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=1, loc="upper right")


def save_per_game_plots(df: pd.DataFrame, config_label: str, out_dir: Path) -> None:
    """For each game in `df`, save a single-line consec-fail-vs-streak plot.

    Filename uses the original CSV stem plus a `_failures` suffix so it
    doesn't collide with other plots already in the directory.
    """
    if df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for game_id in sorted(df["game"].unique()):
        sub = df[df["game"] == game_id].sort_values("streak")
        stem = sub["game_stem"].iloc[0]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sub["streak"], sub["consec_fail_before_leaving"],
                "-o", color="tab:blue", linewidth=1.5, markersize=4)
        ax.set_xlabel("Streak", fontsize=11)
        ax.set_ylabel("Consecutive failures\nbefore leaving", fontsize=11)
        ax.set_title(f"{config_label} — {stem}\n{len(sub)} bouts", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_failures.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {df['game'].nunique()} per-game plots to {out_dir.relative_to(REPO_ROOT)}")


def make_grid_figure(
    configs_data: dict[str, pd.DataFrame],
    plot_fn,
    suptitle: str,
    out_path: Path,
    share_y: bool,
    **plot_kwargs,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=share_y)
    for ax, (_category, name, label, _fig_dir) in zip(axes.flat, CONFIGS):
        df = configs_data[name]
        plot_fn(ax, df, label, **plot_kwargs)
    for ax in axes[-1, :]:
        ax.set_xlabel("Streak", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("Consecutive failures\nbefore leaving", fontsize=11)

    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--error", choices=["sem", "sd"], default="sem",
                        help="Error band for the mean plot (default: sem)")
    parser.add_argument("--n-lines", type=int, default=5,
                        help="Number of individual-game lines per panel (default: 5)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for sampling which games to show")
    parser.add_argument("--out-dir", type=Path, default=FIGS_DIR,
                        help="Directory to write figures into")
    args = parser.parse_args()

    configs_data: dict[str, pd.DataFrame] = {}
    for category, name, label, _fig_dir in CONFIGS:
        subdir = DATA_ROOT / category / name
        if not subdir.exists():
            print(f"WARNING: missing directory {subdir}")
            configs_data[name] = pd.DataFrame(
                columns=["game", "streak", "rewards_within_streak", "consec_fail_before_leaving"]
            )
            continue
        df = load_config(subdir)
        configs_data[name] = df
        n_games = df["game"].nunique() if not df.empty else 0
        print(f"{label:40s} {n_games:>2} games, {len(df):>4} bouts")

    make_grid_figure(
        configs_data,
        plot_mean_per_streak,
        suptitle=f"Qwen individual behavior — mean consecutive failures before leaving per streak (± {args.error.upper()})",
        out_path=args.out_dir / "individual_behavior_mean.png",
        share_y=True,
        error=args.error,
    )
    make_grid_figure(
        configs_data,
        plot_individual_lines,
        suptitle=f"Qwen individual behavior — one line per game (up to {args.n_lines} sampled per config)",
        out_path=args.out_dir / "individual_behavior_lines.png",
        share_y=False,
        n_lines=args.n_lines,
        seed=args.seed,
    )

    for _category, name, label, fig_dir in CONFIGS:
        save_per_game_plots(configs_data[name], label, args.out_dir / fig_dir)


if __name__ == "__main__":
    main()
