"""Batch-generate per-game visualizations for all qwen runs."""

from pathlib import Path

from visualize import create_visualization

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "qwen"
FIGS_ROOT = ROOT / "analysis" / "figs"

# Map each data folder to its corresponding figs folder
FOLDER_MAP = {
    DATA_ROOT / "reasoning" / "qwen_reasoning_maxturns": FIGS_ROOT / "qwen_reasoning_max",
    DATA_ROOT / "reasoning" / "qwen_reasoning_nomaxturns": FIGS_ROOT / "qwen_reasoning_no_max",
    DATA_ROOT / "reasoning" / "qwen_reasoning_nomaxturns_strict": FIGS_ROOT / "qwen_reasoning_no_max_strict",
    DATA_ROOT / "no_reasoning" / "qwen_noreasoning_maxturns": FIGS_ROOT / "qwen_no_reasoning_max",
    DATA_ROOT / "no_reasoning" / "qwen_noreasoning_nomaxturns": FIGS_ROOT / "qwen_no_reasoning_no_max",
    DATA_ROOT / "no_reasoning" / "qwen_noreasoning_nomaxturns_strict": FIGS_ROOT / "qwen_no_reasoning_no_max_strict",
}


def main() -> None:
    for data_dir, figs_dir in FOLDER_MAP.items():
        if not data_dir.exists():
            print(f"Skipping missing data dir: {data_dir}")
            continue
        figs_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(data_dir.glob("*.csv"))
        print(f"\n{data_dir.name}: {len(csv_files)} games -> {figs_dir}")

        for csv_path in csv_files:
            output_path = figs_dir / f"{csv_path.stem}.png"
            create_visualization(csv_path, output_path=output_path, show=False)


if __name__ == "__main__":
    main()
