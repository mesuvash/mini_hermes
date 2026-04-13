#!/usr/bin/env python3
"""
Mini-Hermes CLI (Chapter 15)

Terminal interface that wires all components together:
- Interactive prompt with history, arrow keys, auto-suggest
- Frozen system prompt built once at session start
- SQLite + FTS5 session persistence
- Persistent memory (MEMORY.md / USER.md)
- Skill system with progressive disclosure
- Learning loop with nudge counters
"""

import sys
import select
import termios
import threading
import tty
import yaml
import logging
from pick import pick
from pathlib import Path
from openai import OpenAI
from tool_calling import strategy_for_model

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML

# Ensure mini_hermes directory is on path
sys.path.insert(0, str(Path(__file__).parent))

from tool_registry import registry
from agent import Agent
from prompt_builder import PromptBuilder
from memory.session_db import SessionDB
from memory.persistent import PersistentMemory
from memory.recall import SessionRecall
from skills.loader import SkillLoader
from context_compression import ContextCompressor

# Import tool modules to trigger registration
import tools.terminal
import tools.file_tools
import tools.memory_tool
import tools.tool_creator
import tools.introspect
import tools.lmstudio_logs
import skills.manager

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("mini-hermes")


def _run_with_esc(agent, user_input: str) -> str:
    """Run agent in a background thread. Press ESC to cancel."""
    result = [None]
    error = [None]

    def _target():
        try:
            result[0] = agent.run(user_input)
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    # Save terminal settings and switch to raw mode to detect ESC
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)  # cbreak: read char-by-char, no echo
        while thread.is_alive():
            thread.join(timeout=0.1)
            # Check for keypress without blocking
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == '\x1b':  # ESC
                    agent._abort = True
                    # Wait briefly for thread to notice
                    thread.join(timeout=2.0)
                    return "[Cancelled]"
    finally:
        # Restore terminal to normal mode
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    if error[0]:
        return f"[Error: {error[0]}]"
    return result[0]


def main():
    # ── Load config ──
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)
    config = yaml.safe_load(config_path.read_text())

    # ── Initialize OpenAI client (pointed at LM Studio) ──
    client = OpenAI(
        api_key=config["model"]["api_key"],
        base_url=config["model"]["base_url"],
    )
    model = config["model"]["model"]
    max_tokens = config["model"].get("max_tokens", 400)

    # ── Data directory ──
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    skills_dir = data_dir / "skills"
    skills_dir.mkdir(exist_ok=True)

    # ── Initialize components ──
    session_db = SessionDB(data_dir / "state.db")
    persistent = PersistentMemory(data_dir)
    skill_loader = SkillLoader(skills_dir)
    recall = SessionRecall(
        session_db, client, model,
        max_tokens=config.get("aux_model", {}).get("max_tokens", 300),
    )

    # Wire memory + skills tools
    tools.memory_tool.set_memory(persistent, recall)
    skills.manager.set_skill_loader(skill_loader, skills_dir)

    # Wire custom tool creator + load existing custom tools
    custom_tools_dir = data_dir / "custom_tools"
    tools.tool_creator.set_custom_tools_dir(custom_tools_dir)
    custom_tool_count = tools.tool_creator.load_custom_tools()

    # ── Build system prompt ONCE (frozen snapshot) ──
    builder = PromptBuilder()
    system_prompt = builder.build(
        memory_block=persistent.load(),
        skills_index=skill_loader.build_skills_index(),
    )

    # ── Create session ──
    session_id = session_db.create_session(
        source="cli", system_prompt=system_prompt,
    )

    # ── Create agent ──
    agent = Agent(
        client=client,
        model=model,
        system_prompt=system_prompt,
        tools=registry.get_schemas(),
        max_iterations=config.get("agent", {}).get("max_iterations", 15),
        max_tokens=max_tokens,
    )
    agent.set_registry(registry)
    agent.set_handlers(registry.get_handlers())
    agent.session_db = session_db
    agent.session_id = session_id

    # Configure learning loop
    learning = config.get("learning", {})
    agent.configure_learning(
        memory_nudge=learning.get("memory_nudge_interval", 5),
        skill_nudge=learning.get("skill_nudge_interval", 8),
    )

    # Context compression (Chapter 14)
    compressor = ContextCompressor(
        client=client, model=model,
        max_context_tokens=32000,
        max_tokens=max_tokens,
    )
    agent.set_compressor(compressor)

    # ── Interactive prompt with history ──
    history_file = data_dir / ".prompt_history"
    session = PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
        enable_history_search=True,  # Ctrl-R reverse search
    )

    # ── REPL ──
    print("╔══════════════════════════════════════╗")
    print("║       Mini-Hermes Agent v0.1         ║")
    print("║  /help for commands                  ║")
    print("╚══════════════════════════════════════╝")
    print(f"  Model: {model}")
    print(f"  Session: {session_id[:8]}...")
    mem_status = "loaded" if persistent.load() else "empty"
    skill_count = len(skill_loader.load_all())
    print(f"  Memory: {mem_status} | Skills: {skill_count} | Custom Tools: {custom_tool_count}")
    print()

    while True:
        try:
            user_input = session.prompt(
                HTML("<ansigreen><b>you &gt;</b></ansigreen> ")
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
            print("Goodbye!")
            break

        # Slash commands
        if user_input == "/help":
            print("\n\033[1;33m── Commands ──\033[0m")
            print("  /help       Show this help message")
            print("  /clear      Clear conversation context (start fresh)")
            print("  /mem        Show persistent memory and user profile")
            print("  /skills     List learned skills")
            print("  /model      Switch between available models")
            print("  /sessions   Search past sessions by keyword")
            print("  /tools      List all registered tools")
            print("  exit        Quit and save session")
            print()
            print("\033[1;33m── Shortcuts ──\033[0m")
            print("  ESC         Cancel ongoing request")
            print("  ↑/↓         Navigate command history")
            print("  Ctrl-R      Reverse search history")
            print("  Ctrl-A/E    Jump to start/end of line")
            print("  Tab         Accept auto-suggestion")
            print()
            continue

        if user_input == "/clear":
            # Reset conversation to just the system prompt
            agent.messages = [
                {"role": "system", "content": agent.system_prompt}
            ]
            msg_count = len(agent.messages)
            print(f"  Context cleared. ({msg_count} system message kept)\n")
            continue

        if user_input == "/tools":
            tool_names = sorted(registry._tools.keys())
            print(f"\n\033[1;33m── {len(tool_names)} Tools ──\033[0m")
            for name in tool_names:
                entry = registry._tools[name]
                print(f"  {name}: {entry.description[:70]}")
            print()
            continue

        if user_input == "/mem":
            print("\n\033[1;33m── Memory ──\033[0m")
            print(persistent.read_memory())
            print("\n\033[1;33m── User Profile ──\033[0m")
            print(persistent.read_user())
            print()
            continue

        if user_input == "/skills":
            all_skills = skill_loader.load_all()
            if not all_skills:
                print("No skills yet.\n")
            else:
                print(f"\n\033[1;33m── {len(all_skills)} Skills ──\033[0m")
                for s in all_skills:
                    print(f"  {s['name']}: {s['description'][:60]}")
                print()
            continue

        if user_input == "/model":
            print("\033[2m  fetching models...\033[0m", end="", flush=True)
            try:
                models = client.models.list()
                model_ids = [m.id for m in models.data]
            except Exception as e:
                print(f"\r  Error listing models: {e}\n")
                continue
            print("\r" + " " * 40 + "\r", end="")
            options = [f"{mid}  ← active" if mid == agent.model else mid
                       for mid in model_ids]
            try:
                default = model_ids.index(agent.model)
            except ValueError:
                default = 0
            try:
                _, idx = pick(options, "Select model (↑↓ to move, Enter to select, q to cancel):",
                              default_index=default)
                new_model = model_ids[idx]
                agent.model = new_model
                agent._strategy = strategy_for_model(new_model)
                strategy_name = type(agent._strategy).__name__
                print(f"  Switched to \033[1m{new_model}\033[0m ({strategy_name})\n")
            except (KeyboardInterrupt, EOFError):
                print("  Cancelled.\n")
            continue

        if user_input == "/sessions":
            try:
                query = session.prompt("Search query: ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if query:
                result = recall.recall(query)
                print(result if result else "No results.\n")
            continue

        # Normal agent turn
        print("\033[2m  thinking... (ESC to cancel)\033[0m",
              end="", flush=True)
        response = _run_with_esc(agent, user_input)
        # Clear the "thinking..." line
        print("\r" + " " * 60 + "\r", end="")

        print(f"\033[1;36mhermes >\033[0m {response}\n")

    # End session
    session_db.end_session(session_id)
    print(f"Session {session_id[:8]} saved.")


if __name__ == "__main__":
    main()
