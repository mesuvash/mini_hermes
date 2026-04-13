"""
Tool-Calling Strategies (Chapter 3b)

Two strategies for tool calling depending on model capabilities:
- StructuredStrategy: for models with native function-calling (Qwen, Mistral, etc.)
- TextStrategy: for models that emit tool calls as text (Gemma, LLaMA, etc.)

The agent delegates three decisions to the strategy:
  1. How to present tools to the model (API param vs. system prompt text)
  2. How to parse tool calls from the response
  3. How to feed tool results back into the conversation

Text parsing handles five formats (tried in order):
  1. XML:          <function=name><parameter=key>value</parameter>...
  2. call:colon:   call:tool_name{...json...}
  3. Tagged JSON:  <tool_call>{"name":...}</tool_call>
  4. Fenced JSON:  ```json {"name":...} ```
  5. Bare JSON:    {"name":..., "arguments":...}
"""

import json
import re
import uuid
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data type — uniform representation both strategies produce
# ---------------------------------------------------------------------------

class ParsedToolCall:
    """A tool call extracted from a model response, regardless of format."""
    __slots__ = ("id", "name", "arguments", "from_reasoning")

    def __init__(self, name: str, arguments: dict, call_id: str = None,
                 from_reasoning: bool = False):
        self.id = call_id or f"call_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.arguments = arguments
        self.from_reasoning = from_reasoning


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------

class ToolCallingStrategy:
    """Interface that the Agent delegates to."""

    def prepare_kwargs(self, kwargs: dict, tools: list[dict]) -> dict:
        raise NotImplementedError

    def parse_response(self, msg) -> tuple[str, list[ParsedToolCall]]:
        raise NotImplementedError

    def build_assistant_msg(self, content: str,
                            tool_calls: list[ParsedToolCall]) -> dict:
        raise NotImplementedError

    def build_tool_result_msg(self, call: ParsedToolCall,
                              result: str) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Strategy 1: Structured (OpenAI-compatible function calling)
# ---------------------------------------------------------------------------

class StructuredStrategy(ToolCallingStrategy):
    """For models that support the `tools` API parameter."""

    def prepare_kwargs(self, kwargs, tools):
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return kwargs

    def parse_response(self, msg):
        content = msg.content or ""
        raw_calls = getattr(msg, "tool_calls", None)
        if raw_calls:
            try:
                parsed = [
                    ParsedToolCall(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                        if tc.function.arguments else {},
                        call_id=tc.id,
                    )
                    for tc in raw_calls
                ]
                return content, parsed
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning("Failed to parse structured tool calls: %s", e)
        # Fallback: some servers (LM Studio) return tool calls as text
        # even when the model supposedly supports structured calling.
        if content:
            clean, calls = parse_tool_calls(content)
            if calls:
                return clean, calls
        # Last resort: check reasoning_content — reasoning models (Qwen3,
        # DeepSeek-R1) sometimes emit tool calls in the thinking trace.
        # Tag these so the agent knows to use text-style result messages
        # (role="tool" gets dropped by servers when there's no matching
        # tool_calls in the assistant message).
        reasoning = getattr(msg, "reasoning_content", "") or ""
        if reasoning:
            _, calls = parse_tool_calls(reasoning)
            if calls:
                for c in calls:
                    c.from_reasoning = True
                return content, calls
        return content, []

    def build_assistant_msg(self, content, tool_calls):
        # Don't put tool_calls in the message dict for history.
        # Many local servers (LM Studio) drop role="tool" results and
        # assistant tool_calls from subsequent prompts, causing loops.
        # The structured tools API param still tells the model what's
        # available — we just keep the history in plain text.
        return {"role": "assistant", "content": content}

    def build_tool_result_msg(self, call, result):
        # Use role="user" so local servers always include it in the prompt.
        return {
            "role": "user",
            "content": f"[Tool Result: {call.name}]\n{result}",
        }


# ---------------------------------------------------------------------------
# Strategy 2: Text-based (parse tool calls from model output)
# ---------------------------------------------------------------------------

class TextStrategy(ToolCallingStrategy):
    """For models that emit tool calls as text (Gemma, LLaMA, etc.)."""

    def __init__(self):
        self._tools_text: str = ""

    def prepare_kwargs(self, kwargs, tools):
        if tools and not self._tools_text:
            self._tools_text = _format_tools_as_text(tools)
        if self._tools_text:
            msgs = kwargs.get("messages", [])
            if msgs and msgs[0]["role"] == "system":
                msgs = list(msgs)
                msgs[0] = dict(msgs[0])
                msgs[0]["content"] += "\n\n" + self._tools_text
                kwargs["messages"] = msgs
        return kwargs

    def parse_response(self, msg):
        content = msg.content or ""
        if content:
            clean, calls = parse_tool_calls(content)
            if calls:
                return clean, calls
        # Fallback: check reasoning_content for reasoning models.
        reasoning = getattr(msg, "reasoning_content", "") or ""
        if reasoning:
            _, calls = parse_tool_calls(reasoning)
            if calls:
                for c in calls:
                    c.from_reasoning = True
                return content, calls
        return content, []

    def build_assistant_msg(self, content, tool_calls):
        return {"role": "assistant", "content": content}

    def build_tool_result_msg(self, call, result):
        return {
            "role": "user",
            "content": f"[Tool Result: {call.name}]\n{result}",
        }


# ---------------------------------------------------------------------------
# Unified text parser — single entry point for all text formats
# ---------------------------------------------------------------------------

def parse_tool_calls(text: str) -> tuple[str, list[ParsedToolCall]]:
    """Try every known text format in priority order.
    Returns (cleaned_text, tool_calls). If nothing matched, returns (text, []).
    """
    for extractor in _EXTRACTORS:
        clean, calls = extractor(text)
        if calls:
            return clean, calls
    return text, []


# ---------------------------------------------------------------------------
# Format 1: XML — <function=name><parameter=key>value</parameter>...
# Used by LM Studio / Qwen in some configurations.
# ---------------------------------------------------------------------------

_XML_FUNC_RE = re.compile(
    r"<function=(\w+)>(.*?)(?:</function>|$)", re.DOTALL)
_XML_PARAM_RE = re.compile(
    r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_TOOL_CALL_TAG_RE = re.compile(r"<\|?/?tool_call\|?>")


def _try_xml_format(text: str) -> tuple[str, list[ParsedToolCall]]:
    calls = []
    for m in _XML_FUNC_RE.finditer(text):
        name = m.group(1)
        args = {pm.group(1): pm.group(2).strip()
                for pm in _XML_PARAM_RE.finditer(m.group(2))}
        calls.append(ParsedToolCall(name=name, arguments=args))
    if not calls:
        return text, []
    clean = _XML_FUNC_RE.sub("", text)
    clean = _TOOL_CALL_TAG_RE.sub("", clean).strip()
    return clean, calls


# ---------------------------------------------------------------------------
# Format 2: call:name{...} — brace-balanced JSON after a prefix
# Used by Gemma and some fine-tuned models.
#   Examples:  call:list_custom_tools{}
#              call:create_tool{"name":"test","handler_code":"return 1"}
# ---------------------------------------------------------------------------

_CALL_COLON_RE = re.compile(r"call:(\w+)\{")


def _try_call_colon(text: str) -> tuple[str, list[ParsedToolCall]]:
    calls = []
    clean = text
    for m in _CALL_COLON_RE.finditer(text):
        name = m.group(1)
        brace_start = m.end() - 1  # position of the '{'
        inner = _extract_braced(text, brace_start)
        if inner is None:
            continue
        args = {} if not inner.strip() else _lenient_json(inner)
        calls.append(ParsedToolCall(name=name, arguments=args))
    if not calls:
        return text, []
    # Strip the call:...{...} and surrounding tool_call tags
    clean = _CALL_COLON_RE.sub("", clean)
    clean = re.sub(r"\{[^{}]*\}", "", clean, count=len(calls))
    clean = _TOOL_CALL_TAG_RE.sub("", clean).strip()
    return clean, calls


# ---------------------------------------------------------------------------
# Format 3: Tagged JSON — <tool_call>{"name":...}</tool_call>
# ChatML and variants: <|tool_call|>, <tool_call>, etc.
# ---------------------------------------------------------------------------

_TAGGED_JSON_RE = re.compile(
    r"<\|?tool_call\|?>(.+?)<\|?/?tool_call\|?>", re.DOTALL)


def _try_tagged_json(text: str) -> tuple[str, list[ParsedToolCall]]:
    matches = _TAGGED_JSON_RE.findall(text)
    if not matches:
        return text, []
    calls = []
    for raw in matches:
        calls.extend(_parse_json_tool_call(raw.strip()))
    if not calls:
        return text, []
    clean = _TAGGED_JSON_RE.sub("", text).strip()
    return clean, calls


# ---------------------------------------------------------------------------
# Format 4: Fenced JSON — ```json {...} ``` or ```tool_call {...} ```
# ---------------------------------------------------------------------------

_FENCED_JSON_RE = re.compile(
    r"```(?:tool_call|json)?\s*\n?({.+?})\s*\n?```", re.DOTALL)


def _try_fenced_json(text: str) -> tuple[str, list[ParsedToolCall]]:
    matches = _FENCED_JSON_RE.findall(text)
    if not matches:
        return text, []
    calls = []
    for raw in matches:
        calls.extend(_parse_json_tool_call(raw.strip()))
    if not calls:
        return text, []
    clean = _FENCED_JSON_RE.sub("", text).strip()
    return clean, calls


# ---------------------------------------------------------------------------
# Format 5: Bare JSON — {"name": "tool", "arguments": {...}}
# Last resort: a raw JSON object with a "name" key.
# ---------------------------------------------------------------------------

def _try_bare_json(text: str) -> tuple[str, list[ParsedToolCall]]:
    """Find a JSON object containing "name" by scanning for { and brace-balancing."""
    # Find candidate positions: { before "name"
    for m in re.finditer(r'\{', text):
        inner = _extract_braced(text, m.start())
        if inner is None:
            continue
        full = "{" + inner + "}"
        if '"name"' not in full:
            continue
        calls = _parse_json_tool_call(full)
        if calls:
            clean = text[:m.start()] + text[m.start() + len(full):]
            return clean.strip(), calls
    return text, []


# Extractor priority order
_EXTRACTORS = [
    _try_xml_format,
    _try_call_colon,
    _try_tagged_json,
    _try_fenced_json,
    _try_bare_json,
]


# ---------------------------------------------------------------------------
# JSON tool call parser — shared by formats 3/4/5
# ---------------------------------------------------------------------------

def _parse_json_tool_call(raw: str) -> list[ParsedToolCall]:
    """Parse a JSON string into one or more ParsedToolCalls.
    Accepts:
      {"name": "x", "arguments": {...}}
      [{"name": "x", ...}, ...]
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(obj, dict) and "name" in obj:
        args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        return [ParsedToolCall(name=obj["name"], arguments=args)]

    if isinstance(obj, list):
        return [
            ParsedToolCall(
                name=item["name"],
                arguments=item.get("arguments", item.get("args", {})),
            )
            for item in obj
            if isinstance(item, dict) and "name" in item
        ]

    return []


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extract_braced(text: str, start: int) -> str | None:
    """Extract content between balanced braces starting at text[start]='{'.
    Returns the inner content (without outer braces), or None on failure."""
    if start >= len(text) or text[start] != '{':
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    # Unbalanced — return everything after opening brace
    return text[start + 1:]


def _lenient_json(s: str) -> dict:
    """Parse JSON from model output, tolerating common escaping issues.

    Tries (in order):
      1. Direct json.loads
      2. Wrap bare key-values in braces: "k":"v" -> {"k":"v"}
      3. Fix double-escaping: \\" -> "
      4. Regex key-value extraction (last resort)
    """
    s = s.replace('<|"|>', '"')  # some models emit this

    # Build candidates: original, then wrapped in braces
    candidates = [s]
    if not s.startswith("{"):
        candidates.append("{" + s + "}")

    for c in candidates:
        for attempt in (c, c.replace('\\"', '"')):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

    # Last resort: regex key-value extraction
    return _extract_kv_pairs(s) or {"raw": s}


def _extract_kv_pairs(s: str) -> dict | None:
    """Pull "key": "value" pairs from malformed JSON via state machine.
    Handles escaped quotes and newlines inside values."""
    result = {}
    i = 0
    while i < len(s):
        # Find next "key":
        km = re.search(r'"(\w+)"\s*:\s*', s[i:])
        if not km:
            break
        key = km.group(1)
        vstart = i + km.end()
        if vstart >= len(s):
            break

        if s[vstart] == '"':
            val, end = _scan_string(s, vstart)
            result[key] = val
            i = end
        else:
            # Non-string value: grab up to next comma or end
            nk = re.search(r',\s*"', s[vstart:])
            if nk:
                result[key] = s[vstart:vstart + nk.start()].strip().strip('"')
                i = vstart + nk.start() + 1
            else:
                result[key] = s[vstart:].strip().strip('"')
                break
    return result or None


def _scan_string(s: str, start: int) -> tuple[str, int]:
    """Scan a JSON-ish quoted string from s[start]='"'.
    Returns (decoded_value, position_after_close_quote)."""
    chars = []
    i = start + 1
    while i < len(s):
        ch = s[i]
        if ch == '\\' and i + 1 < len(s):
            nxt = s[i + 1]
            chars.append({'n': '\n', 't': '\t', '"': '"',
                          '\\': '\\'}.get(nxt, nxt))
            i += 2
            continue
        if ch == '"':
            # Real end-of-string if followed by , } ] whitespace or EOF
            rest = s[i + 1:].lstrip()
            if not rest or rest[0] in (',', '}', ']'):
                return ''.join(chars), i + 1
            # Unescaped quote mid-string — include it and keep going
            chars.append(ch)
            i += 1
            continue
        chars.append(ch)
        i += 1
    return ''.join(chars), i


# ---------------------------------------------------------------------------
# Tool schema formatter (for TextStrategy system prompt injection)
# ---------------------------------------------------------------------------

def _format_tools_as_text(tools: list[dict]) -> str:
    """Render tool schemas as plain-text instructions for the system prompt."""
    lines = [
        "## Available Tools",
        "Call tools by responding with a JSON block inside <tool_call> tags:",
        '<tool_call>{"name": "tool_name", "arguments": {"arg": "value"}}</tool_call>',
        "",
        "You may call multiple tools by using multiple <tool_call> blocks.",
        "After each tool call, you will receive the result and can continue.",
        "",
        "Tools:",
    ]
    for t in tools:
        fn = t.get("function", t)
        name = fn["name"]
        desc = fn.get("description", "")
        props = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])

        lines.append(f"\n### {name}")
        lines.append(desc)
        if props:
            lines.append("  Parameters:")
            for pname, pdef in props.items():
                req = " (required)" if pname in required else ""
                ptype = pdef.get("type", "string")
                pdesc = pdef.get("description", "")
                lines.append(f"    - {pname} ({ptype}{req}): {pdesc}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_STRUCTURED_MODELS = {
    "qwen", "mistral", "hermes", "functionary", "firefunction",
    "gorilla", "nexusraven", "command-r",
}


def strategy_for_model(model_name: str) -> ToolCallingStrategy:
    """Return the right strategy based on model name heuristics."""
    name_lower = model_name.lower()
    for keyword in _STRUCTURED_MODELS:
        if keyword in name_lower:
            return StructuredStrategy()
    return TextStrategy()
