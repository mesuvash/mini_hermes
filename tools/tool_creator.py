"""
Tool Creator (Dynamic Tool Registration)

Meta-tool that lets the agent create new callable tools at runtime.
Custom tools are saved as Python files in data/custom_tools/ and
auto-loaded on startup.

Each custom tool file follows a standard template:
  - A handler function
  - A TOOL_DEF dict with name, description, parameters, category

The create_tool handler:
  1. Validates the tool definition
  2. Writes the Python file to data/custom_tools/
  3. Dynamically imports and registers it in the ToolRegistry
  4. The tool is immediately available in the current session

Security: Custom tools run with the same permissions as the agent.
The terminal tool already has shell access, so this doesn't expand
the attack surface.
"""

import importlib.util
import json
import logging
import sys
from pathlib import Path
from tool_registry import registry

logger = logging.getLogger(__name__)

# Set by cli.py at startup
_custom_tools_dir: Path = None

TOOL_TEMPLATE = '''"""
Custom tool: {name}
{description}
"""

{imports}


def handler({params_signature}):
    """{description}"""
{handler_body}


TOOL_DEF = {{
    "name": "{name}",
    "description": """{description}""",
    "parameters": {parameters_json},
    "category": "custom",
}}
'''


def set_custom_tools_dir(path: Path):
    global _custom_tools_dir
    _custom_tools_dir = path
    _custom_tools_dir.mkdir(parents=True, exist_ok=True)


def load_custom_tools():
    """Load all custom tools from data/custom_tools/ on startup."""
    if _custom_tools_dir is None or not _custom_tools_dir.exists():
        return 0
    count = 0
    for py_file in sorted(_custom_tools_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        try:
            _load_tool_file(py_file)
            count += 1
        except Exception as e:
            logger.warning("Failed to load custom tool %s: %s", py_file.name, e)
    return count


def _load_tool_file(py_file: Path):
    """Dynamically import a tool file and register it."""
    module_name = f"custom_tool_{py_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    tool_def = getattr(module, "TOOL_DEF", None)
    handler_fn = getattr(module, "handler", None)
    if not tool_def or not handler_fn:
        raise ValueError(f"{py_file.name} missing TOOL_DEF or handler")

    registry.register(
        name=tool_def["name"],
        description=tool_def["description"],
        parameters=tool_def["parameters"],
        handler=handler_fn,
        category=tool_def.get("category", "custom"),
    )
    logger.info("Loaded custom tool: %s", tool_def["name"])


def create_tool(name: str, description: str, parameters: str,
                imports: str, handler_code: str) -> str:
    """Create a new tool, save it as a Python file, and register it immediately.

    Args:
        name: Tool name (lowercase, underscores). Must be unique.
        description: What the tool does.
        parameters: JSON string of the OpenAI parameters schema.
        imports: Python import statements (one per line).
        handler_code: The function body (will be indented under def handler(...)).
    """
    if _custom_tools_dir is None:
        return "Error: custom tools directory not initialized"

    # Validate name
    if not name.replace("_", "").isalnum():
        return "Error: name must be alphanumeric with underscores only"
    if name in registry._tools:
        return f"Error: tool '{name}' already exists. Use update_tool to modify it."

    # Parse parameters schema
    try:
        params = json.loads(parameters)
    except json.JSONDecodeError as e:
        return f"Error: invalid parameters JSON: {e}"

    # Build parameter signature from schema
    props = params.get("properties", {})
    required = set(params.get("required", []))
    sig_parts = []
    for pname in props:
        ptype = props[pname].get("type", "string")
        default = props[pname].get("default")
        if pname in required:
            sig_parts.append(pname)
        elif default is not None:
            sig_parts.append(f"{pname}={repr(default)}")
        else:
            sig_parts.append(f"{pname}=None")
    params_signature = ", ".join(sig_parts)

    # Indent handler code
    lines = handler_code.strip().split("\n")
    indented = "\n".join("    " + line for line in lines)

    # Generate file content
    file_content = TOOL_TEMPLATE.format(
        name=name,
        description=description,
        imports=imports.strip(),
        params_signature=params_signature,
        handler_body=indented,
        parameters_json=json.dumps(params, indent=4),
    )

    # Write file
    py_file = _custom_tools_dir / f"{name}.py"
    py_file.write_text(file_content)

    # Load and register immediately
    try:
        _load_tool_file(py_file)
    except Exception as e:
        py_file.unlink()  # clean up on failure
        return f"Error: tool file written but failed to load: {e}"

    # Auto-test: call the tool with default/sample args to verify it runs
    test_result = None
    test_error = None
    has_required_no_default = bool(
        required - set(p for p in props if "default" in props[p])
    )
    try:
        entry = registry._tools.get(name)
        if entry and not has_required_no_default:
            # Build args from defaults in schema
            test_args = {}
            for pname, pdef in props.items():
                if "default" in pdef:
                    test_args[pname] = pdef["default"]
            result = entry.handler(**test_args)
            result_str = str(result)
            test_result = result_str[:500] + ("..." if len(result_str) > 500 else "")
    except Exception as e:
        test_error = str(e)

    response = {
        "success": True,
        "message": f"Tool '{name}' created and registered.",
        "file": str(py_file),
    }
    if has_required_no_default:
        response["test"] = "skipped (has required params, call it to test)"
    elif test_error:
        response["test"] = "FAILED"
        response["test_error"] = test_error
    else:
        response["test"] = "PASSED"
        response["test_output"] = test_result
    return json.dumps(response)


def update_tool(name: str, description: str = None, parameters: str = None,
                imports: str = None, handler_code: str = None) -> str:
    """Update an existing custom tool. Only provided fields are changed."""
    if _custom_tools_dir is None:
        return "Error: custom tools directory not initialized"

    py_file = _custom_tools_dir / f"{name}.py"
    if not py_file.exists():
        return f"Error: custom tool '{name}' not found. Use create_tool first."

    # For simplicity, recreate the tool with new values
    # First, load existing to get current values
    try:
        module_name = f"custom_tool_{name}_temp"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        old_def = module.TOOL_DEF
    except Exception as e:
        return f"Error reading existing tool: {e}"

    # Read the current file to extract imports and handler code
    current_content = py_file.read_text()

    # Unregister old version
    if name in registry._tools:
        del registry._tools[name]

    # Use provided values or fall back to old
    new_desc = description or old_def["description"]
    new_params_str = parameters or json.dumps(old_def["parameters"])

    if imports is None or handler_code is None:
        # If not fully replacing, just rewrite the whole file
        if imports is not None and handler_code is not None:
            pass
        else:
            # Need to provide both imports and handler_code for a full rewrite
            return (
                "Error: for update_tool, provide both 'imports' and 'handler_code' "
                "together (the full replacement), or use delete_tool + create_tool."
            )

    # Delete and recreate
    py_file.unlink()
    return create_tool(name, new_desc, new_params_str, imports, handler_code)


def delete_tool(name: str) -> str:
    """Delete a custom tool."""
    if _custom_tools_dir is None:
        return "Error: custom tools directory not initialized"

    py_file = _custom_tools_dir / f"{name}.py"
    if not py_file.exists():
        return f"Error: custom tool '{name}' not found"

    # Unregister
    if name in registry._tools:
        del registry._tools[name]

    py_file.unlink()
    return json.dumps({
        "success": True,
        "message": f"Tool '{name}' deleted and unregistered.",
    })


def list_custom_tools() -> str:
    """List all custom tools."""
    if _custom_tools_dir is None:
        return "No custom tools directory configured."
    tools = []
    for py_file in sorted(_custom_tools_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        name = py_file.stem
        entry = registry._tools.get(name)
        if entry:
            tools.append({"name": name, "description": entry.description[:200]})
        else:
            tools.append({"name": name, "status": "not loaded"})
    return json.dumps(tools, indent=2) if tools else "No custom tools yet."


# ── Register the meta-tools ──

registry.register(
    name="create_tool",
    description=(
        "Create a new callable tool, register it, and auto-test it. "
        "Returns test status and output. Persists across restarts."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Tool name (lowercase, underscores). e.g. 'fetch_nepali_news'",
            },
            "description": {
                "type": "string",
                "description": "What the tool does. Shown to the LLM for tool selection.",
            },
            "parameters": {
                "type": "string",
                "description": (
                    "JSON string of the OpenAI function parameters schema. "
                    'e.g. \'{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}\''
                ),
            },
            "imports": {
                "type": "string",
                "description": "Python import statements, one per line. e.g. 'import requests\\nfrom bs4 import BeautifulSoup'",
            },
            "handler_code": {
                "type": "string",
                "description": (
                    "The Python function body (without 'def' or indentation). "
                    "Parameters match the schema properties. Return a string result."
                ),
            },
        },
        "required": ["name", "description", "parameters", "imports", "handler_code"],
    },
    handler=create_tool,
    category="meta",
)

registry.register(
    name="update_tool",
    description="Update an existing custom tool. Provide the full new imports and handler_code.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Tool name to update"},
            "description": {"type": "string", "description": "New description (optional)"},
            "parameters": {"type": "string", "description": "New parameters JSON (optional)"},
            "imports": {"type": "string", "description": "New import statements"},
            "handler_code": {"type": "string", "description": "New function body"},
        },
        "required": ["name", "imports", "handler_code"],
    },
    handler=update_tool,
    category="meta",
)

registry.register(
    name="delete_tool",
    description="Delete a custom tool and unregister it.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Tool name to delete"},
        },
        "required": ["name"],
    },
    handler=delete_tool,
    category="meta",
)

registry.register(
    name="list_custom_tools",
    description="List all custom tools that have been created.",
    parameters={"type": "object", "properties": {}},
    handler=list_custom_tools,
    category="meta",
)
