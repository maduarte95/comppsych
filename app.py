"""Streamlit GUI for the LLM Foraging Task."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.config import LLMParams, TaskParams
from src.data_logger import DataLogger
from src.llm_client import AnthropicClient, OpenRouterClient, create_client
from src.prompt_builder import PromptBuilder
from src.protocols import PROTOCOLS, list_protocols
from src.runner import run_batch_threaded
from src.task_engine import TaskEngine

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="LLM Foraging Task",
    page_icon="🧙‍♀️",
    layout="wide",
)

st.title("🧙‍♀️ LLM Foraging Task Engine")
st.markdown("Adapting the Witch Foraging Task (Vertechi et al.) for LLM evaluation")



def init_session_state() -> None:
    """Initialize session state variables."""
    if "run_complete" not in st.session_state:
        st.session_state.run_complete = False
    if "run_results" not in st.session_state:
        st.session_state.run_results = None
    if "is_running" not in st.session_state:
        st.session_state.is_running = False


init_session_state()

# Sidebar - Configuration
st.sidebar.header("Configuration")

# LLM Settings
st.sidebar.subheader("LLM Settings")

provider = st.sidebar.selectbox(
    "Provider",
    options=["anthropic", "openrouter"],
    index=0,
)

# Get available models based on provider
if provider == "anthropic":
    available_models = AnthropicClient.MODELS
else:
    available_models = OpenRouterClient.MODELS

model = st.sidebar.selectbox(
    "Model",
    options=available_models,
    index=0,
)

# Thinking/Reasoning mode settings
# Must be before temperature since Anthropic thinking disables temperature control
if provider == "anthropic":
    supports_adaptive = model in AnthropicClient.ADAPTIVE_THINKING_MODELS
    if supports_adaptive:
        thinking_options = ["disabled", "adaptive", "enabled"]
        thinking_help = (
            "disabled: No thinking (default).\n"
            "adaptive: Claude dynamically decides per-request whether to think "
            "and how much, based on prompt complexity. Also enables interleaved "
            "thinking between tool calls. Truly dynamic. Requires Opus 4.6 or "
            "Sonnet 4.6.\n"
            "enabled: Always think with a fixed token budget (deprecated on 4.6 "
            "models; prefer adaptive)."
        )
    else:
        thinking_options = ["disabled", "enabled"]
        thinking_help = (
            "disabled: No thinking (default).\n"
            "enabled: Always think with a fixed token budget.\n"
            "(Note: adaptive mode requires Opus/Sonnet 4.6.)"
        )
elif provider == "openrouter":
    if model in OpenRouterClient.REASONING_MODELS:
        thinking_options = ["disabled", "adaptive", "enabled"]
        thinking_help = (
            "disabled: No reasoning (default).\n"
            "adaptive: NOT truly adaptive for these models. Sends "
            "reasoning.effort='high', which OpenRouter translates to a fixed "
            "thinking_budget of ~80% of max_tokens (roughly 12,800 tokens with "
            "our current max_tokens=16000). The 'Thinking Budget' field below "
            "is ignored in this mode.\n"
            "enabled: Reasoning with the fixed token budget set below "
            "(mapped to Qwen's thinking_budget / equivalent)."
        )
    else:
        thinking_options = ["disabled"]
        thinking_help = "This model does not support reasoning."

thinking_mode = st.sidebar.selectbox(
    "Thinking Mode" if provider == "anthropic" else "Reasoning Mode",
    options=thinking_options,
    index=0,
    help=thinking_help,
)

if thinking_mode == "enabled":
    thinking_budget = st.sidebar.number_input(
        "Thinking Budget (tokens)",
        min_value=1000,
        max_value=100000,
        value=1500,
        step=500,
        help="Token budget for thinking/reasoning (only used in 'enabled' mode)",
    )
else:
    thinking_budget = 1500  # Default, not used in adaptive/disabled

# Temperature control - locked to 1.0 for Anthropic when thinking is enabled
thinking_enabled = thinking_mode != "disabled"

if thinking_enabled and provider == "anthropic":
    st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.1,
        disabled=True,
        help="Temperature is fixed at 1.0 when thinking mode is enabled (Anthropic requirement)",
    )
    temperature = 1.0
else:
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
    )

# API Key
api_key_env = (
    os.getenv("ANTHROPIC_API_KEY")
    if provider == "anthropic"
    else os.getenv("OPENROUTER_API_KEY")
)
api_key_input = st.sidebar.text_input(
    "API Key (or use .env)",
    value="",
    type="password",
    help="Leave empty to use API key from .env file",
)
api_key = api_key_input if api_key_input else api_key_env

# Task Settings
st.sidebar.subheader("Task Settings")

protocol_options = list_protocols() + ["custom"]
protocol_name = st.sidebar.selectbox(
    "Protocol",
    options=protocol_options,
    index=1,  # Default to medium
    format_func=lambda x: x.capitalize(),
)

if protocol_name == "custom":
    p_reward = st.sidebar.slider(
        "P(reward)",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
    )
    p_switch = st.sidebar.slider(
        "P(switch)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )
    travel_cost = st.sidebar.slider(
        "Travel Cost (turns)",
        min_value=0,
        max_value=5,
        value=2,
        help="Set to 0 to disable travel penalty (instant switching)",
    )
    max_points = st.sidebar.number_input(
        "Max Points (monster life)",
        min_value=50,
        max_value=2000,
        value=250,
        step=50,
    )
    points_per_hit = st.sidebar.number_input(
        "Points per Hit",
        min_value=1,
        max_value=10,
        value=3,
    )
    max_turns = st.sidebar.number_input(
        "Max Turns",
        min_value=50,
        max_value=1000,
        value=300,
        step=50,
    )

    immediate_feedback = st.sidebar.checkbox(
        "Immediate Feedback",
        value=True,
        help="When enabled, agent is told if the monster is at the tower they traveled to. When disabled, agent must infer monster location from hit/miss patterns.",
    )

    task_params = TaskParams(
        p_reward=p_reward,
        p_switch=p_switch,
        travel_cost=travel_cost,
        max_turns=max_turns,
        max_points=max_points,
        points_per_hit=points_per_hit,
        immediate_feedback=immediate_feedback,
    )
else:
    task_params = PROTOCOLS[protocol_name]
    st.sidebar.markdown(f"""
    **Protocol Parameters:**
    - P(reward): {task_params.p_reward}
    - P(switch): {task_params.p_switch}
    - Travel cost: {task_params.travel_cost}
    - Max points: {task_params.max_points}
    - Points/hit: {task_params.points_per_hit}
    - Max turns: {task_params.max_turns}
    - Immediate feedback: {task_params.immediate_feedback}
    """)

# Prompt Settings
with st.sidebar.expander("Prompt Settings"):
    # Get available prompt templates from prompts folder
    prompts_dir = Path("prompts")
    available_templates = sorted([f.stem for f in prompts_dir.glob("*.yaml")])

    if not available_templates:
        st.error("No prompt templates found in prompts/ folder")
        selected_template = "default"
    else:
        selected_template = st.selectbox(
            "Prompt Template",
            options=available_templates,
            index=available_templates.index("default") if "default" in available_templates else 0,
            help="Select a prompt template from the prompts/ folder"
        )

    # Load the selected template
    template_path = prompts_dir / f"{selected_template}.yaml"
    prompt_builder = PromptBuilder(template_path=template_path)

    system_prompt = st.text_area(
        "System Prompt",
        value=prompt_builder.system_template,
        height=100,
    )

    # Show user prompt template in an expander (read-only since it uses placeholders)
    with st.expander("View User Prompt Template"):
        st.code(prompt_builder.user_template, language="text")
        st.caption("*User prompt is generated dynamically by filling in placeholders based on game state*")

# Main area — run mode toggle
run_mode = st.radio("Run Mode", ["Single Run", "Batch Run"], horizontal=True)

# Batch-specific controls
if run_mode == "Batch Run":
    batch_col1, batch_col2, batch_col3 = st.columns(3)
    with batch_col1:
        n_games = st.number_input("Number of Games", min_value=2, max_value=100, value=10, step=1)
    with batch_col2:
        max_concurrent = st.number_input(
            "Max Concurrent", min_value=1, max_value=50, value=10, step=1,
            help="Limit concurrent API requests to avoid rate limits",
        )
    with batch_col3:
        base_seed = st.number_input(
            "Base Seed", min_value=0, max_value=999999, value=42, step=1,
            help="Game i gets seed = base_seed + i for reproducibility",
        )

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Run Controls")

    # Check for API key
    if not api_key:
        st.warning(
            f"⚠️ No API key found. Please enter an API key or set "
            f"{'ANTHROPIC_API_KEY' if provider == 'anthropic' else 'OPENROUTER_API_KEY'} "
            f"in your .env file."
        )

    run_button = st.button(
        "🚀 Start Run",
        disabled=not api_key or st.session_state.is_running,
        use_container_width=True,
    )

    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.run_complete = False
        st.session_state.run_results = None
        st.session_state.is_running = False
        st.rerun()

with col2:
    st.subheader("Status")
    if st.session_state.is_running:
        st.info("🔄 Running...")
    elif st.session_state.run_complete:
        st.success("✅ Run complete!")
    else:
        st.info("Ready to start")

# Run the task
if run_button and api_key:
    st.session_state.is_running = True

    # Shared LLM params
    max_tokens = 16000 if thinking_mode != "disabled" else 64

    llm_params = LLMParams(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking_mode=thinking_mode,
        thinking_budget=thinking_budget,
    )

    template_path = Path("prompts") / f"{selected_template}.yaml"

    if run_mode == "Batch Run":
        # --- Batch run (async concurrent) ---
        prompt_builder = PromptBuilder(template_path=template_path)

        # Progress elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"Running {n_games} games ({max_concurrent} concurrent)...")

        progress_state = {"games_done": 0}

        def on_game_done(game_id: int, n_total: int, result: dict) -> None:
            progress_state["games_done"] += 1
            done = progress_state["games_done"]
            outcome = result.get("outcome", "?")
            score = result.get("stats", {}).get("final_score", "?")
            status_text.text(
                f"Game {game_id} finished: {outcome} (score: {score}) "
                f"[{done}/{n_total} done]"
            )
            progress_bar.progress(min(done / n_total, 1.0))

        batch_results = run_batch_threaded(
            n_games=n_games,
            task_params=task_params,
            llm_params=llm_params,
            api_key=api_key,
            template_path=template_path,
            system_prompt=system_prompt,
            protocol_name=protocol_name,
            output_dir="data",
            max_concurrent=max_concurrent,
            base_seed=base_seed,
            on_game_done=on_game_done,
        )

        st.session_state.run_results = {"mode": "batch", "batch_results": batch_results}
        st.session_state.run_complete = True
        st.session_state.is_running = False
        progress_bar.progress(1.0)
        status_text.text("Batch complete!")
        st.rerun()

    else:
        # --- Single run (original sync loop) ---
        engine = TaskEngine(task_params)
        prompt_builder = PromptBuilder(template_path=template_path)
        client = create_client(llm_params, api_key)
        logger = DataLogger(
            output_dir="data",
            model=model,
            protocol=protocol_name,
        )

        # Progress display
        progress_bar = st.progress(0)
        status_text = st.empty()
        current_turn_display = st.empty()

        # Run loop
        total_parse_errors = 0
        first_write = True

        while True:
            # Check game over
            game_over, reason = engine.is_game_over()
            if game_over:
                break

            # Update progress
            progress = min(engine.current_turn / task_params.max_turns, 1.0)
            progress_bar.progress(progress)
            status_text.text(
                f"Turn {engine.current_turn}/{task_params.max_turns} | "
                f"Monster Life: {engine.monster_life}/{task_params.max_points}"
            )

            # Build prompts
            user_prompt = prompt_builder.build_user_prompt(engine)

            # Get LLM response
            response, errors = client.complete_with_retry(
                system=system_prompt,
                user=user_prompt,
                max_retries=3,
            )

            # Log parse errors
            for error in errors:
                logger.add_parse_error(
                    error.raw_response, error.error_message, engine.current_turn
                )
                total_parse_errors += 1

            if response is None:
                # All retries failed
                reason = "error"
                break

            # Log LLM response (including thinking if enabled)
            logger.save_llm_response(
                turn=engine.current_turn,
                choice=response.choice,
                text=response.raw_response,
                thinking=response.thinking,
                prompt=user_prompt,
            )

            # Process choice
            result = engine.process_choice(response.choice)

            # Log turn events
            for event in result.events:
                logger.save_turn(event, task_params, write_header=first_write)
                first_write = False

            if result.game_over:
                reason = result.game_over_reason
                break

        # Save metadata
        prompts_used = {
            "template_file": selected_template,
            "system": system_prompt,
            "user_template": prompt_builder.user_template,
        }
        logger.save_metadata(
            task_params=task_params,
            llm_params=llm_params,
            engine=engine,
            outcome=reason or "unknown",
            prompts_used=prompts_used,
        )

        # Store results
        st.session_state.run_results = {
            "mode": "single",
            "engine": engine,
            "logger": logger,
            "outcome": reason,
            "parse_errors": total_parse_errors,
        }
        st.session_state.run_complete = True
        st.session_state.is_running = False

        # Clear progress displays
        progress_bar.progress(1.0)
        status_text.text("Run complete!")

        st.rerun()

# Display results
if st.session_state.run_complete and st.session_state.run_results:
    results = st.session_state.run_results

    st.divider()
    st.header("Results")

    if results.get("mode") == "batch":
        # --- Batch results display ---
        batch_results = results["batch_results"]
        n_total = len(batch_results)
        wins = sum(1 for r in batch_results if r.get("outcome") == "win")
        turn_limits = sum(1 for r in batch_results if r.get("outcome") == "turn_limit")
        errors = sum(1 for r in batch_results if r.get("outcome") == "error")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Wins", f"{wins}/{n_total}")
        with col2:
            st.metric("Turn Limits", f"{turn_limits}/{n_total}")
        with col3:
            st.metric("Errors", f"{errors}/{n_total}")

        # Per-game summary table
        st.subheader("Per-Game Summary")
        rows = []
        for r in batch_results:
            stats = r.get("stats", {})
            rows.append({
                "Game": r.get("game_id", "?"),
                "Seed": r.get("seed", "?"),
                "Outcome": r.get("outcome", "?"),
                "Score": stats.get("final_score", "?"),
                "Turns": stats.get("total_turns", "?"),
                "Hits": stats.get("total_hits", "?"),
                "Hit Rate": f"{stats['hit_rate']:.1%}" if "hit_rate" in stats else "?",
                "Switches": stats.get("total_switches", "?"),
                "Parse Errors": r.get("parse_errors", "?"),
                "CSV": r.get("csv_path", ""),
            })
        summary_df = pd.DataFrame(rows)
        summary_df["Seed"] = pd.to_numeric(summary_df["Seed"], errors="coerce")
        st.dataframe(summary_df, width="stretch", height=400)

        # Aggregate stats
        valid_results = [r for r in batch_results if "stats" in r]
        if valid_results:
            st.subheader("Aggregate Statistics")
            hit_rates = [r["stats"]["hit_rate"] for r in valid_results]
            total_turns = [r["stats"]["total_turns"] for r in valid_results]
            scores = [r["stats"]["final_score"] for r in valid_results]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Hit Rate", f"{sum(hit_rates) / len(hit_rates):.1%}")
            with col2:
                st.metric("Mean Turns", f"{sum(total_turns) / len(total_turns):.0f}")
            with col3:
                st.metric("Mean Score", f"{sum(scores) / len(scores):.0f}")

    else:
        # --- Single run results display ---
        engine = results["engine"]
        logger = results["logger"]
        outcome = results["outcome"]

        # Outcome and summary
        col1, col2, col3 = st.columns(3)

        with col1:
            if outcome == "win":
                st.success("🎉 Monster Defeated!")
            elif outcome == "turn_limit":
                st.warning("⏰ Turn Limit Reached")
            else:
                st.error("❌ Run Error")

        with col2:
            st.metric("Final Score", f"{engine.score}/{task_params.max_points}")

        with col3:
            st.metric("Parse Errors", results["parse_errors"])

        # Summary stats
        stats = engine.get_summary_stats()

        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Turns", stats["total_turns"])
        with col2:
            st.metric("Total Hits", stats["total_hits"])
        with col3:
            st.metric("Hit Rate", f"{stats['hit_rate']:.1%}")
        with col4:
            st.metric("Total Switches", stats["total_switches"])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Correct Switches", stats["correct_switches"])
        with col2:
            st.metric("Incorrect Switches", stats["incorrect_switches"])

        # Game log
        st.subheader("Game Log")
        csv_path, meta_path, llm_log_path = logger.get_file_paths()

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            st.dataframe(df, use_container_width=True, height=300)

            # Download button
            st.download_button(
                "📥 Download CSV",
                data=df.to_csv(index=False),
                file_name=csv_path.name,
                mime="text/csv",
            )

        # Visualizations
        st.subheader("Visualizations")

        if csv_path.exists():
            df = pd.read_csv(csv_path)

            # Check if required columns exist
            if "turn" in df.columns and "score" in df.columns:
                col1, col2 = st.columns(2)

                with col1:
                    # Score over turns
                    st.markdown("**Score Over Turns**")
                    score_df = df[["turn", "score"]].drop_duplicates()
                    st.line_chart(score_df.set_index("turn"))

                with col2:
                    # Raster plot with monster location, witch position, and attack outcomes
                    st.markdown("**Task Raster Plot**")
                    if "side" in df.columns and "active_side" in df.columns:
                        position_df = df[df["side"] >= 0][["turn", "side", "active_side"]].drop_duplicates(subset=["turn"])
                        # Get fire events with outcomes
                        fire_df = df[df["action"] == "fire"][["turn", "side", "reward"]].copy()

                        if not position_df.empty:
                            fig, ax = plt.subplots(figsize=(10, 4))

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
                            ax.step(turns, player, where="post", color="black", linewidth=1.5,
                                   label="Witch Position")

                            # Raster ticks for attack outcomes
                            # Top raster (LEFT side attacks, y ~ 1.05)
                            # Bottom raster (RIGHT side attacks, y ~ -0.05)
                            for _, row in fire_df.iterrows():
                                turn = row["turn"]
                                side = row["side"]
                                reward = row["reward"]
                                color = "green" if reward == 1 else "grey"
                                y_pos = 1.08 if side == 1 else -0.08  # LEFT at top, RIGHT at bottom
                                ax.plot(turn, y_pos, marker="|", markersize=8, color=color,
                                       markeredgewidth=1.5)

                            ax.set_ylim(-0.2, 1.2)
                            ax.set_yticks([0, 1])
                            ax.set_yticklabels(["RIGHT", "LEFT"])
                            ax.set_xlabel("Turn")
                            ax.set_ylabel("Position")

                            # Custom legend
                            from matplotlib.patches import Patch
                            from matplotlib.lines import Line2D
                            legend_elements = [
                                Patch(facecolor="blue", alpha=0.2, label="Monster LEFT"),
                                Patch(facecolor="red", alpha=0.2, label="Monster RIGHT"),
                                Line2D([0], [0], color="black", linewidth=1.5, label="Witch Position"),
                                Line2D([0], [0], marker="|", color="green", linestyle="None",
                                      markersize=8, markeredgewidth=1.5, label="Hit"),
                                Line2D([0], [0], marker="|", color="grey", linestyle="None",
                                      markersize=8, markeredgewidth=1.5, label="Miss"),
                            ]
                            ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

                            st.pyplot(fig)
                            plt.close(fig)
            else:
                st.warning("CSV columns not found. Available columns: " + ", ".join(df.columns.tolist()))

        # File locations
        st.subheader("Output Files")
        st.markdown(f"- CSV: `{csv_path}`")
        st.markdown(f"- Metadata: `{meta_path}`")
        st.markdown(f"- LLM Log: `{llm_log_path}`")
