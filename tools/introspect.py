"""
Introspection tool — lets the agent inspect its own tools and source code.

Useful when the agent needs to:
- Understand how existing tools are implemented before creating new ones
- Debug tool registration or handler issues
- Navigate its own codebase
"""

import inspect
import json
from pathlib import Path
from tool_registry import registry

_PROJECT_ROOT = Path(__file__).parent.parent


def introspect(target: str, name: str = None) -> str:
    """Inspect mini_hermes internals.

    Args:
        target: What to inspect. One of:
            "tools"     — list all registered tools (or detail for one if name given)
            "source"    — show project file tree (or file contents if name given)
            "config"    — show current config.yaml
            "custom"    — list custom tool files and their code
        name: Optional. Tool name (for tools), file path (for source),
              or tool name (for custom).
    """
    if target == "tools":
        return _inspect_tools(name)
    elif target == "source":
        return _inspect_source(name)
    elif target == "config":
        return _inspect_config()
    elif target == "custom":
        return _inspect_custom(name)
    else:
        return (
            f"Unknown target '{target}'. Use one of:\n"
            "  tools   — registered tools and their schemas\n"
            "  source  — project file tree or file contents\n"
            "  config  — current configuration\n"
            "  custom  — custom tool files"
        )


def _inspect_tools(name: str = None) -> str:
    """List all tools or show full detail for one."""
    if name:
        entry = registry._tools.get(name)
        if not entry:
            return f"Tool '{name}' not found. Use introspect(target='tools') to list all."
        # Full detail: schema + handler source
        lines = [
            f"# Tool: {entry.name}",
            f"Category: {entry.category}",
            f"Description: {entry.description}",
            f"\n## Parameters Schema",
            json.dumps(entry.parameters, indent=2),
            f"\n## Handler: {entry.handler.__module__}.{entry.handler.__qualname__}",
        ]
        try:
            source = inspect.getsource(entry.handler)
            lines.append(source)
        except (OSError, TypeError):
            lines.append("(source not available)")
        return "\n".join(lines)

    # List all tools grouped by category
    by_category: dict[str, list] = {}
    for entry in registry._tools.values():
        by_category.setdefault(entry.category, []).append(entry)

    lines = [f"# Registered Tools ({len(registry._tools)} total)\n"]
    for cat in sorted(by_category):
        lines.append(f"## {cat}")
        for entry in by_category[cat]:
            params = entry.parameters.get("properties", {})
            param_names = ", ".join(params.keys()) if params else "(none)"
            lines.append(f"  {entry.name}({param_names})")
            lines.append(f"    {entry.description[:120]}")
        lines.append("")
    return "\n".join(lines)


def _inspect_source(path: str = None) -> str:
    """Show project structure or read a specific source file."""
    if path:
        # Resolve relative to project root
        target = _PROJECT_ROOT / path
        if not target.exists():
            return f"File not found: {path}\nUse introspect(target='source') to see the file tree."
        if not target.is_file():
            return f"Not a file: {path}"
        # Safety: only allow reading files within the project
        try:
            target.resolve().relative_to(_PROJECT_ROOT.resolve())
        except ValueError:
            return f"Cannot read files outside the project directory."
        content = target.read_text()
        if len(content) > 50000:
            content = content[:50000] + "\n... [truncated]"
        return f"# {path}\n\n{content}"

    # Build file tree (exclude .venv, __pycache__, .git, data/)
    skip = {".venv", "__pycache__", ".git", ".claude", "data"}
    lines = ["# mini_hermes project structure\n"]
    _walk_tree(_PROJECT_ROOT, lines, skip, prefix="")
    return "\n".join(lines)


def _walk_tree(directory: Path, lines: list, skip: set, prefix: str):
    """Recursively build a file tree."""
    entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
    for i, entry in enumerate(entries):
        if entry.name in skip:
            continue
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        if entry.is_dir():
            lines.append(f"{prefix}{connector}{entry.name}/")
            extension = "    " if is_last else "│   "
            _walk_tree(entry, lines, skip, prefix + extension)
        else:
            size = entry.stat().st_size
            lines.append(f"{prefix}{connector}{entry.name}  ({size:,} bytes)")


def _inspect_config() -> str:
    """Show current config.yaml."""
    config_path = _PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        return "config.yaml not found."
    return f"# config.yaml\n\n{config_path.read_text()}"


def _inspect_custom(name: str = None) -> str:
    """List custom tools or show one's source code."""
    custom_dir = _PROJECT_ROOT / "data" / "custom_tools"
    if not custom_dir.exists():
        return "No custom tools directory yet."

    py_files = sorted(f for f in custom_dir.glob("*.py") if not f.name.startswith("_"))
    if not py_files:
        return "No custom tools created yet."

    if name:
        target = custom_dir / f"{name}.py"
        if not target.exists():
            names = [f.stem for f in py_files]
            return f"Custom tool '{name}' not found. Available: {', '.join(names)}"
        return f"# custom_tools/{name}.py\n\n{target.read_text()}"

    # List all custom tools with first line of their description
    lines = [f"# Custom Tools ({len(py_files)} total)\n"]
    for f in py_files:
        content = f.read_text()
        # Extract description from docstring
        desc = ""
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith('"""') and "Custom tool:" in line:
                continue
            if line and not line.startswith('"""') and not line.startswith("#"):
                desc = line[:100]
                break
        lines.append(f"  {f.stem}: {desc}")
    return "\n".join(lines)


# ── Registration ──

registry.register(
    name="introspect",
    description=(
        "Inspect mini_hermes internals: registered tools, source code, "
        "config, and custom tools. Use before creating new tools to "
        "understand existing patterns.\n"
        "Targets: tools, source, config, custom.\n"
        "Pass name for detail (tool name or file path)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "enum": ["tools", "source", "config", "custom"],
                "description": "What to inspect: tools, source, config, or custom",
            },
            "name": {
                "type": "string",
                "description": (
                    "Optional detail selector. For tools: tool name. "
                    "For source: file path (e.g. 'tools/terminal.py'). "
                    "For custom: tool name."
                ),
            },
        },
        "required": ["target"],
    },
    handler=introspect,
    category="meta",
)
