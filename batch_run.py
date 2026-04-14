"""CLI for running concurrent batch experiments."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.config import LLMParams, TaskParams
from src.llm_client import AnthropicClient, OpenRouterClient
from src.prompt_builder import PromptBuilder
from src.protocols import PROTOCOLS, list_protocols
from src.runner import run_batch

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run concurrent LLM foraging experiments"
    )
    parser.add_argument(
        "-n", "--n-games", type=int, default=10, help="Number of concurrent games"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="medium",
        choices=list_protocols(),
        help="Task protocol",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openrouter",
        choices=["anthropic", "openrouter"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: first model for the provider)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="default",
        help="Prompt template name from prompts/ folder",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--thinking-mode",
        type=str,
        default="disabled",
        choices=["disabled", "adaptive", "enabled"],
    )
    parser.add_argument(
        "--thinking-budget", type=int, default=1500, help="Thinking token budget"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent games (default: unlimited)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Base seed for reproducibility (game i gets seed base_seed + i)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data", help="Output directory"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # Resolve API key
    if args.provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        env_var = (
            "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENROUTER_API_KEY"
        )
        print(f"Error: {env_var} not set in environment or .env file")
        sys.exit(1)

    # Resolve model
    if args.model is None:
        if args.provider == "anthropic":
            args.model = AnthropicClient.MODELS[0]
        else:
            args.model = OpenRouterClient.MODELS[0]

    # Build params
    task_params = PROTOCOLS[args.protocol]

    max_tokens = 16000 if args.thinking_mode != "disabled" else 64
    llm_params = LLMParams(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=max_tokens,
        thinking_mode=args.thinking_mode,
        thinking_budget=args.thinking_budget,
    )

    template_path = Path("prompts") / f"{args.template}.yaml"
    if not template_path.exists():
        print(f"Error: template not found: {template_path}")
        sys.exit(1)

    prompt_builder = PromptBuilder(template_path=template_path)
    system_prompt = prompt_builder.system_template

    print(f"Running {args.n_games} games concurrently")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model}")
    print(f"  Protocol: {args.protocol}")
    print(f"  Thinking: {args.thinking_mode}")
    if args.max_concurrent:
        print(f"  Max concurrent: {args.max_concurrent}")
    if args.base_seed is not None:
        print(f"  Seeds: {args.base_seed} .. {args.base_seed + args.n_games - 1}")
    print()

    results = asyncio.run(
        run_batch(
            n_games=args.n_games,
            task_params=task_params,
            llm_params=llm_params,
            api_key=api_key,
            template_path=template_path,
            system_prompt=system_prompt,
            protocol_name=args.protocol,
            output_dir=args.output_dir,
            max_concurrent=args.max_concurrent,
            base_seed=args.base_seed,
        )
    )

    # Summary
    print("\n" + "=" * 60)
    print("BATCH RESULTS")
    print("=" * 60)

    wins = sum(1 for r in results if r.get("outcome") == "win")
    errors = sum(1 for r in results if r.get("outcome") == "error")
    turn_limits = sum(1 for r in results if r.get("outcome") == "turn_limit")

    print(f"  Wins: {wins}/{args.n_games}")
    print(f"  Turn limits: {turn_limits}/{args.n_games}")
    print(f"  Errors: {errors}/{args.n_games}")

    for r in results:
        stats = r.get("stats", {})
        print(
            f"  Game {r['game_id']:>2}: {r.get('outcome', '?'):>10} | "
            f"score={stats.get('final_score', '?')} | "
            f"turns={stats.get('total_turns', '?')} | "
            f"hits={stats.get('total_hits', '?')} | "
            f"csv={r.get('csv_path', 'N/A')}"
        )


if __name__ == "__main__":
    main()
