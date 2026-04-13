"""
LM Studio log reader — lets the agent inspect server logs to debug issues.

Reads the LM Studio application log and extracts API-relevant entries:
predictions, token usage, tool calls, errors, model loading, etc.

The agent can use this to diagnose:
- Context window overflow (prompt_tokens stops growing)
- Tool calls landing in reasoning_content vs tool_calls
- Model loading errors or GPU config issues
- Why a response was empty or unexpected
"""

import json
import re
from pathlib import Path
from tool_registry import registry

# LM Studio log locations on macOS
_APP_LOG = Path.home() / "Library" / "Logs" / "LM Studio" / "main.log"
# Server/API logs are displayed in LM Studio's UI console but not written
# to a file by default. Users can save them here for analysis:
_SERVER_LOG = Path(__file__).parent.parent / "data" / "lmstudio_server.log"


def lmstudio_logs(action: str = "recent", lines: int = 50,
                  search: str = None, text: str = None) -> str:
    """Read and analyze LM Studio logs.

    Args:
        action: What to do:
            "recent"    — show last N app log lines (filtered to useful entries)
            "errors"    — show recent errors and warnings from app log
            "requests"  — show API request/response summaries (from server log)
            "search"    — search logs for a pattern
            "context"   — analyze token usage for context overflow (server log)
            "raw"       — show last N raw app log lines (unfiltered)
            "save"      — save server log text (paste from LM Studio UI console)
            "status"    — show what log files are available and their sizes
        lines: Number of entries to return (default 50)
        search: Search pattern (for action="search")
        text: Server log text to save (for action="save")
    """
    if action == "save":
        if not text:
            return (
                "Error: 'text' parameter required. Paste the server log "
                "text from LM Studio's UI console."
            )
        return _save_server_log(text)

    if action == "status":
        return _log_status()

    if action == "requests":
        return _extract_requests(lines)
    elif action == "context":
        return _analyze_context()

    # App log actions
    if not _APP_LOG.exists():
        return (
            f"LM Studio app log not found at {_APP_LOG}\n"
            "Expected: ~/Library/Logs/LM Studio/main.log"
        )

    if action == "recent":
        return _recent_entries(lines)
    elif action == "errors":
        return _filter_entries(lines, level="error")
    elif action == "search":
        if not search:
            return "Error: 'search' parameter required for action='search'"
        return _search_logs(search, lines)
    elif action == "raw":
        return _raw_tail(lines)
    else:
        return (
            f"Unknown action '{action}'. Use one of:\n"
            "  recent   — filtered app log entries\n"
            "  errors   — errors and warnings only\n"
            "  requests — API request/response summaries (needs server log)\n"
            "  search   — search for a pattern\n"
            "  context  — analyze token usage for context overflow\n"
            "  raw      — raw tail of app log\n"
            "  save     — save server log text (from LM Studio UI)\n"
            "  status   — show available log files"
        )


def _save_server_log(text: str) -> str:
    """Save server log text pasted from LM Studio UI console."""
    _SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    # Append to existing log with a separator
    mode = "a" if _SERVER_LOG.exists() else "w"
    with open(_SERVER_LOG, mode) as f:
        f.write(f"\n{'='*60}\n")
        f.write(text)
        f.write("\n")
    line_count = text.count("\n") + 1
    return (
        f"Saved {line_count} lines to {_SERVER_LOG.name}. "
        f"Use action='requests' or action='context' to analyze."
    )


def _log_status() -> str:
    """Show available log files, sizes, and detected model config."""
    lines = ["# LM Studio Log Files\n"]
    for name, path in [("App log", _APP_LOG), ("Server log", _SERVER_LOG)]:
        if path.exists():
            size = path.stat().st_size
            lines.append(f"  {name}: {path} ({size:,} bytes)")
        else:
            lines.append(f"  {name}: {path} (not found)")

    # Extract model config from app log
    if _APP_LOG.exists():
        app_lines = _read_log_lines(_APP_LOG, max_bytes=100_000)
        ctx_lines = [l.strip() for l in app_lines if "context length" in l]
        gpu_lines = [l.strip() for l in app_lines
                     if "Num Offload Layers" in l]
        if ctx_lines:
            lines.append(f"\n# Last loaded model config")
            lines.append(f"  {ctx_lines[-1]}")
        if gpu_lines:
            lines.append(f"  {gpu_lines[-1]}")

    lines.append(
        "\nNote: Server logs (API predictions, token counts) are shown in "
        "LM Studio's UI console. Use action='save' with text parameter "
        "to save them here, or copy them to: " + str(_SERVER_LOG)
    )
    return "\n".join(lines)


def _read_log_lines(log_path: Path = None,
                    max_bytes: int = 500_000) -> list[str]:
    """Read the tail of a log file (up to max_bytes)."""
    path = log_path or _APP_LOG
    if not path.exists():
        return []
    size = path.stat().st_size
    offset = max(0, size - max_bytes)
    with open(path, "r", errors="replace") as f:
        if offset > 0:
            f.seek(offset)
            f.readline()  # skip partial line
        return f.readlines()


def _read_server_log(max_bytes: int = 500_000) -> list[str]:
    """Read server log lines."""
    return _read_log_lines(_SERVER_LOG, max_bytes)


def _recent_entries(n: int) -> str:
    """Show recent log entries, filtering out noise."""
    noise = {
        "blob_storage", "DawnGraphite", "GPUCache", "Session Storage",
        "processTicksAndRejections", "webpack", "clientPortOnClose",
    }
    all_lines = _read_log_lines()
    filtered = []
    for line in all_lines:
        if any(skip in line for skip in noise):
            continue
        filtered.append(line.rstrip())
    return "\n".join(filtered[-n:]) or "No relevant entries found."


def _filter_entries(n: int, level: str = "error") -> str:
    """Show entries matching a log level."""
    pattern = f"[{level}]"
    all_lines = _read_log_lines()
    matches = [l.rstrip() for l in all_lines if pattern in l.lower()]
    return "\n".join(matches[-n:]) or f"No {level} entries found."


def _extract_requests(n: int) -> str:
    """Extract API request summaries: model, tokens, tool calls, finish reason."""
    server_lines = _read_server_log()
    app_lines = _read_log_lines() if not server_lines else []
    all_lines = server_lines or app_lines
    if not all_lines:
        return (
            "No server logs found. The API prediction logs are shown in "
            "LM Studio's UI console (not in the app log file). To analyze:\n"
            "1. Copy the log text from LM Studio's server console\n"
            "2. Use action='save' with text=<pasted log> to save it\n"
            "3. Then use action='requests' to analyze"
        )
    full_text = "".join(all_lines)

    # Find all "Generated prediction" JSON blocks
    pattern = re.compile(
        r'Generated prediction:\s*(\{.*?\})\s*(?=\d{4}-|\Z)',
        re.DOTALL
    )

    summaries = []
    for m in pattern.finditer(full_text):
        try:
            pred = json.loads(m.group(1))
            choice = pred["choices"][0]
            msg = choice["message"]
            usage = pred.get("usage", {})

            summary_parts = [
                f"model={pred.get('model', '?')}",
                f"prompt={usage.get('prompt_tokens', '?')}",
                f"completion={usage.get('completion_tokens', '?')}",
                f"finish={choice.get('finish_reason', '?')}",
            ]

            # Tool calls
            tc = msg.get("tool_calls", [])
            if tc:
                names = [t["function"]["name"] for t in tc]
                summary_parts.append(f"tools=[{','.join(names)}]")

            # Content preview
            content = (msg.get("content") or "").strip()
            if content:
                preview = content[:80].replace("\n", " ")
                summary_parts.append(f'content="{preview}..."')
            elif not tc:
                # Check reasoning_content
                rc = (msg.get("reasoning_content") or "").strip()
                if rc:
                    preview = rc[:80].replace("\n", " ")
                    summary_parts.append(f'reasoning="{preview}..."')

            # Reasoning tokens
            cd = usage.get("completion_tokens_details", {})
            if cd.get("reasoning_tokens"):
                summary_parts.append(
                    f"reasoning_tokens={cd['reasoning_tokens']}")

            summaries.append("  ".join(summary_parts))
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    if not summaries:
        return "No API requests found in recent logs."

    header = f"# Recent API Requests ({len(summaries)} total)\n"
    return header + "\n".join(
        f"{i+1}. {s}" for i, s in enumerate(summaries[-n:])
    )


def _analyze_context() -> str:
    """Analyze token usage across recent requests to detect context overflow."""
    server_lines = _read_server_log()
    app_lines = _read_log_lines() if not server_lines else []
    all_lines = server_lines or app_lines
    if not all_lines:
        return (
            "No server logs found. Save server log first with "
            "action='save' and text=<pasted log from LM Studio UI>."
        )
    full_text = "".join(all_lines)

    pattern = re.compile(
        r'Generated prediction:\s*(\{.*?\})\s*(?=\d{4}-|\Z)',
        re.DOTALL
    )

    token_history = []
    for m in pattern.finditer(full_text):
        try:
            pred = json.loads(m.group(1))
            usage = pred.get("usage", {})
            choice = pred["choices"][0]
            msg = choice["message"]
            tc = msg.get("tool_calls", [])
            content = (msg.get("content") or "").strip()
            rc = (msg.get("reasoning_content") or "").strip()

            token_history.append({
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0),
                "finish": choice.get("finish_reason", "?"),
                "has_tool_calls": bool(tc),
                "has_content": bool(content),
                "has_reasoning_tc": bool(
                    rc and ("<function=" in rc or "call:" in rc)),
            })
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    if not token_history:
        return "No token usage data found."

    # Take last 20 requests
    recent = token_history[-20:]

    lines = ["# Context Usage Analysis (last {} requests)\n".format(
        len(recent))]
    lines.append(
        f"{'#':>3}  {'prompt':>7}  {'compl':>6}  {'total':>6}  "
        f"{'finish':>10}  {'tools':>5}  notes"
    )
    lines.append("-" * 70)

    prev_prompt = 0
    issues = []
    for i, entry in enumerate(recent, 1):
        pt = entry["prompt"]
        notes = []

        if prev_prompt > 0 and pt < prev_prompt * 0.9:
            notes.append("CONTEXT OVERFLOW - tokens dropped!")
            issues.append(i)
        if entry["has_reasoning_tc"]:
            notes.append("tool call in reasoning")
        if not entry["has_content"] and not entry["has_tool_calls"]:
            notes.append("empty response")

        lines.append(
            f"{i:>3}  {pt:>7}  {entry['completion']:>6}  "
            f"{entry['total']:>6}  {entry['finish']:>10}  "
            f"{'yes' if entry['has_tool_calls'] else 'no':>5}  "
            f"{'; '.join(notes)}"
        )
        prev_prompt = pt

    if issues:
        lines.append(f"\nWARNING: Context overflow detected at request(s): "
                      f"{', '.join(str(i) for i in issues)}")
        lines.append("The model's context window is too small. "
                      "Increase it in LM Studio model settings.")
    else:
        lines.append("\nNo context overflow detected.")

    return "\n".join(lines)


def _search_logs(pattern: str, n: int) -> str:
    """Search log for a pattern (case-insensitive)."""
    all_lines = _read_log_lines()
    pat = re.compile(pattern, re.IGNORECASE)
    matches = [l.rstrip() for l in all_lines if pat.search(l)]
    if not matches:
        return f"No matches for '{pattern}'."
    return "\n".join(matches[-n:])


def _raw_tail(n: int) -> str:
    """Raw tail of the log file."""
    all_lines = _read_log_lines()
    return "\n".join(l.rstrip() for l in all_lines[-n:])


# ── Registration ──

registry.register(
    name="lmstudio_logs",
    description=(
        "Read LM Studio logs to debug issues. "
        "Actions: status, recent, errors, requests, search, context, raw, save. "
        "Use 'save' to store server log text from LM Studio UI. "
        "Use 'context' to detect context window overflow. "
        "Use 'requests' to see API call summaries."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "recent", "errors", "requests", "search",
                         "context", "raw", "save"],
                "description": "What to do with the logs",
            },
            "lines": {
                "type": "integer",
                "description": "Number of entries to return (default 50)",
                "default": 50,
            },
            "search": {
                "type": "string",
                "description": "Search pattern (for action='search')",
            },
            "text": {
                "type": "string",
                "description": "Server log text to save (for action='save')",
            },
        },
        "required": ["action"],
    },
    handler=lmstudio_logs,
    category="debug",
)
