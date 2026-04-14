"""Game runner for concurrent batch execution."""

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.config import LLMParams, TaskParams
from src.data_logger import DataLogger
from src.llm_client import create_client
from src.prompt_builder import PromptBuilder
from src.task_engine import TaskEngine

logger = logging.getLogger(__name__)


def run_single_game_sync(
    game_id: int,
    task_params: TaskParams,
    llm_params: LLMParams,
    api_key: str,
    template_path: Path,
    system_prompt: str,
    protocol_name: str,
    output_dir: str = "data",
    seed: int | None = None,
    on_turn: Callable[[int, int, int], None] | None = None,
) -> dict:
    """Run a single game synchronously (for use in threads).

    Args:
        game_id: Identifier for this game (for logging)
        task_params: Task parameters
        llm_params: LLM parameters
        api_key: API key for the provider
        template_path: Path to prompt template YAML
        system_prompt: System prompt string
        protocol_name: Protocol name (for file naming)
        output_dir: Output directory for data files
        seed: Random seed for reproducibility
        on_turn: Optional callback(game_id, current_turn, max_turns) called each turn

    Returns:
        Dict with game results (outcome, stats, file paths, errors)
    """
    engine = TaskEngine(task_params, seed=seed)
    prompt_builder = PromptBuilder(template_path=template_path)
    client = create_client(llm_params, api_key)
    game_logger = DataLogger(
        output_dir=output_dir,
        model=llm_params.model,
        protocol=protocol_name,
        suffix=f"game{game_id}",
    )

    total_parse_errors = 0
    first_write = True
    reason = None

    while True:
        game_over, reason = engine.is_game_over()
        if game_over:
            break

        if on_turn:
            on_turn(game_id, engine.current_turn, task_params.max_turns)

        user_prompt = prompt_builder.build_user_prompt(engine)

        response, errors = client.complete_with_retry(
            system=system_prompt,
            user=user_prompt,
            max_retries=3,
        )

        for error in errors:
            game_logger.add_parse_error(
                error.raw_response, error.error_message, engine.current_turn
            )
            total_parse_errors += 1

        if response is None:
            reason = "error"
            break

        game_logger.save_llm_response(
            turn=engine.current_turn,
            choice=response.choice,
            text=response.raw_response,
            thinking=response.thinking,
            prompt=user_prompt,
        )

        result = engine.process_choice(response.choice)

        for event in result.events:
            game_logger.save_turn(event, task_params, write_header=first_write)
            first_write = False

        if result.game_over:
            reason = result.game_over_reason
            break

    prompts_used = {
        "template_file": template_path.stem,
        "system": system_prompt,
        "user_template": prompt_builder.user_template,
    }
    game_logger.save_metadata(
        task_params=task_params,
        llm_params=llm_params,
        engine=engine,
        outcome=reason or "unknown",
        prompts_used=prompts_used,
    )

    stats = engine.get_summary_stats()
    csv_path, meta_path, llm_log_path = game_logger.get_file_paths()

    logger.info(
        f"Game {game_id} done: {reason} | "
        f"score={engine.score}/{task_params.max_points} | "
        f"turns={engine.current_turn - 1}"
    )

    return {
        "game_id": game_id,
        "seed": seed,
        "outcome": reason,
        "stats": stats,
        "parse_errors": total_parse_errors,
        "csv_path": str(csv_path),
        "meta_path": str(meta_path),
        "llm_log_path": str(llm_log_path),
    }


def run_batch_threaded(
    n_games: int,
    task_params: TaskParams,
    llm_params: LLMParams,
    api_key: str,
    template_path: Path,
    system_prompt: str,
    protocol_name: str,
    output_dir: str = "data",
    max_concurrent: int = 10,
    base_seed: int | None = None,
    on_game_done: Callable[[int, int, dict], None] | None = None,
    on_turn: Callable[[int, int, int], None] | None = None,
) -> list[dict]:
    """Run multiple games concurrently using threads.

    Works reliably inside Streamlit (which has its own asyncio event loop).

    Args:
        n_games: Number of games to run
        task_params: Task parameters (shared across games)
        llm_params: LLM parameters
        api_key: API key
        template_path: Path to prompt template YAML
        system_prompt: System prompt string
        protocol_name: Protocol name
        output_dir: Output directory
        max_concurrent: Max concurrent games (thread pool size)
        base_seed: If set, game i gets seed base_seed + i for reproducibility
        on_game_done: Optional callback(game_id, n_games, result) when a game finishes
        on_turn: Optional callback(game_id, current_turn, max_turns) each turn

    Returns:
        List of result dicts, one per game
    """
    results: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {}
        for i in range(n_games):
            seed = (base_seed + i) if base_seed is not None else None
            future = pool.submit(
                run_single_game_sync,
                game_id=i,
                task_params=task_params,
                llm_params=llm_params,
                api_key=api_key,
                template_path=template_path,
                system_prompt=system_prompt,
                protocol_name=protocol_name,
                output_dir=output_dir,
                seed=seed,
                on_turn=on_turn,
            )
            futures[future] = i

        for future in as_completed(futures):
            game_id = futures[future]
            try:
                result = future.result()
                results[game_id] = result
                if on_game_done:
                    on_game_done(game_id, n_games, result)
            except Exception as e:
                logger.error(f"Game {game_id} failed with exception: {e}")
                error_result = {
                    "game_id": game_id,
                    "outcome": "error",
                    "error": str(e),
                }
                results[game_id] = error_result
                if on_game_done:
                    on_game_done(game_id, n_games, error_result)

    # Return in game_id order
    return [results[i] for i in range(n_games)]


# --- Async versions (for CLI / non-Streamlit use) ---


async def run_single_game(
    game_id: int,
    task_params: TaskParams,
    llm_params: LLMParams,
    api_key: str,
    template_path: Path,
    system_prompt: str,
    protocol_name: str,
    output_dir: str = "data",
    seed: int | None = None,
    on_turn: Callable[[int, int, int], None] | None = None,
) -> dict:
    """Run a single game asynchronously (for CLI use)."""
    engine = TaskEngine(task_params, seed=seed)
    prompt_builder = PromptBuilder(template_path=template_path)
    client = create_client(llm_params, api_key)
    game_logger = DataLogger(
        output_dir=output_dir,
        model=llm_params.model,
        protocol=protocol_name,
        suffix=f"game{game_id}",
    )

    total_parse_errors = 0
    first_write = True
    reason = None

    while True:
        game_over, reason = engine.is_game_over()
        if game_over:
            break

        if on_turn:
            on_turn(game_id, engine.current_turn, task_params.max_turns)

        user_prompt = prompt_builder.build_user_prompt(engine)

        response, errors = await client.acomplete_with_retry(
            system=system_prompt,
            user=user_prompt,
            max_retries=3,
        )

        for error in errors:
            game_logger.add_parse_error(
                error.raw_response, error.error_message, engine.current_turn
            )
            total_parse_errors += 1

        if response is None:
            reason = "error"
            break

        game_logger.save_llm_response(
            turn=engine.current_turn,
            choice=response.choice,
            text=response.raw_response,
            thinking=response.thinking,
            prompt=user_prompt,
        )

        result = engine.process_choice(response.choice)

        for event in result.events:
            game_logger.save_turn(event, task_params, write_header=first_write)
            first_write = False

        if result.game_over:
            reason = result.game_over_reason
            break

    prompts_used = {
        "template_file": template_path.stem,
        "system": system_prompt,
        "user_template": prompt_builder.user_template,
    }
    game_logger.save_metadata(
        task_params=task_params,
        llm_params=llm_params,
        engine=engine,
        outcome=reason or "unknown",
        prompts_used=prompts_used,
    )

    stats = engine.get_summary_stats()
    csv_path, meta_path, llm_log_path = game_logger.get_file_paths()

    logger.info(
        f"Game {game_id} done: {reason} | "
        f"score={engine.score}/{task_params.max_points} | "
        f"turns={engine.current_turn - 1}"
    )

    return {
        "game_id": game_id,
        "seed": seed,
        "outcome": reason,
        "stats": stats,
        "parse_errors": total_parse_errors,
        "csv_path": str(csv_path),
        "meta_path": str(meta_path),
        "llm_log_path": str(llm_log_path),
    }


async def run_batch(
    n_games: int,
    task_params: TaskParams,
    llm_params: LLMParams,
    api_key: str,
    template_path: Path,
    system_prompt: str,
    protocol_name: str,
    output_dir: str = "data",
    max_concurrent: int | None = None,
    base_seed: int | None = None,
    on_game_done: Callable[[int, int, dict], None] | None = None,
    on_turn: Callable[[int, int, int], None] | None = None,
) -> list[dict]:
    """Run multiple games concurrently using asyncio (for CLI use)."""
    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    async def _run_one(game_id: int, seed: int | None) -> dict:
        async def _inner() -> dict:
            return await run_single_game(
                game_id=game_id,
                task_params=task_params,
                llm_params=llm_params,
                api_key=api_key,
                template_path=template_path,
                system_prompt=system_prompt,
                protocol_name=protocol_name,
                output_dir=output_dir,
                seed=seed,
                on_turn=on_turn,
            )

        if semaphore:
            async with semaphore:
                result = await _inner()
        else:
            result = await _inner()

        if on_game_done:
            on_game_done(game_id, n_games, result)

        return result

    tasks = []
    for i in range(n_games):
        seed = (base_seed + i) if base_seed is not None else None
        tasks.append(_run_one(i, seed))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    final_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"Game {i} failed with exception: {r}")
            final_results.append({
                "game_id": i,
                "outcome": "error",
                "error": str(r),
            })
        else:
            final_results.append(r)

    return final_results
