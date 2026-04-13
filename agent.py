"""
Mini-Hermes Agent (Chapters 3, 6, 13, 14)

Core agent loop with:
- Tool calling (Chapter 3)
- Prompt caching (Chapter 6)
- Learning loop with nudge counters + background review (Chapter 13)
- Context compression (Chapter 14)
"""

import copy
import json
import threading
import time
import logging
from typing import Optional
from openai import OpenAI
from tool_calling import ToolCallingStrategy, strategy_for_model

logger = logging.getLogger(__name__)


class Agent:
    """Core agent loop: message -> tools -> response."""

    def __init__(self, client: OpenAI, model: str, system_prompt: str,
                 tools: list[dict], max_iterations: int = 15,
                 max_tokens: int = 400):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt  # frozen per session
        self.tools = tools                  # fallback if no registry set
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]

        # Tool-calling strategy (structured vs text-based)
        self._strategy: ToolCallingStrategy = strategy_for_model(model)

        # Tool registry -- if set, schemas and handlers are read live
        # so newly created tools are available immediately.
        self._registry = None

        # Tool handlers -- used as fallback if no registry set
        self._tool_handlers: dict = {}

        # Session DB -- set externally
        self.session_db = None
        self.session_id: Optional[str] = None

        # Nudge counters (Chapter 13)
        self._memory_nudge_interval = 5
        self._skill_nudge_interval = 8
        self._turns_since_memory = 0
        self._iters_since_skill = 0
        self._user_turn_count = 0

        # Abort flag — set by CLI when user presses ESC
        self._abort = False

        # Compression (Chapter 14)
        self._compressor = None  # set via set_compressor()
        self._enable_prompt_caching = False  # set via configure_caching()

    def set_registry(self, registry):
        """Set the tool registry for live schema/handler lookups.
        When set, newly registered tools (e.g. via create_tool) are
        available to the LLM on the very next turn."""
        self._registry = registry

    def set_handlers(self, handlers: dict):
        self._tool_handlers = handlers

    def configure_learning(self, memory_nudge: int = 5,
                           skill_nudge: int = 8):
        self._memory_nudge_interval = memory_nudge
        self._skill_nudge_interval = skill_nudge

    def set_compressor(self, compressor):
        """Set the ContextCompressor (Chapter 14)."""
        self._compressor = compressor

    def configure_caching(self, enable: bool = False):
        """Enable prompt caching (Chapter 6). Only useful for Anthropic models."""
        self._enable_prompt_caching = enable

    def run(self, user_input: str) -> str:
        """Run the agent loop for a single user turn. Returns final text."""
        self._user_turn_count += 1
        self._turns_since_memory += 1

        # Check memory nudge BEFORE the turn
        should_review_memory = (
            self._memory_nudge_interval > 0
            and self._turns_since_memory >= self._memory_nudge_interval
        )
        if should_review_memory:
            self._turns_since_memory = 0

        # Add user message
        self.messages.append({"role": "user", "content": user_input})
        self._persist_message("user", user_input)

        # Agent loop
        self._abort = False
        tool_iters_this_turn = 0
        for i in range(self.max_iterations):
            if self._abort:
                return "[Cancelled]"

            response = self._call_llm()
            if self._abort:
                return "[Cancelled]"
            if response is None:
                return "[Error: LLM call failed]"

            msg = response.choices[0].message

            # Strategy parses the response uniformly
            content, tool_calls = self._strategy.parse_response(msg)

            # Build and append assistant message
            assistant_msg = self._strategy.build_assistant_msg(
                content, tool_calls)
            self.messages.append(assistant_msg)

            # If no tool calls, we're done
            if not tool_calls:
                self._persist_message("assistant", content)
                break

            # Execute tool calls
            tool_iters_this_turn += 1
            self._iters_since_skill += 1

            for tc in tool_calls:
                if self._abort:
                    break

                # Reset nudge counter if agent uses the tool voluntarily
                if tc.name == "memory":
                    self._turns_since_memory = 0
                elif tc.name == "skill_manage":
                    self._iters_since_skill = 0

                result = self._execute_tool(tc.name, tc.arguments)

                result_msg = self._strategy.build_tool_result_msg(tc, result)
                self.messages.append(result_msg)
                self._persist_message("tool", result, tool_name=tc.name,
                                      tool_call_id=tc.id)
        else:
            content = "[Max iterations reached]"

        # Check skill nudge AFTER the turn
        should_review_skills = (
            self._skill_nudge_interval > 0
            and self._iters_since_skill >= self._skill_nudge_interval
        )
        if should_review_skills:
            self._iters_since_skill = 0

        # Spawn background review if nudges fired (Chapter 13)
        if should_review_memory or should_review_skills:
            self._spawn_background_review(
                review_memory=should_review_memory,
                review_skills=should_review_skills,
            )

        return content

    def _call_llm(self):
        """Make one API call with tool schemas."""
        try:
            # Compression check (Chapter 14)
            if self._compressor:
                est = self._compressor._estimate_tokens(self.messages)
                threshold = self._compressor.max_context_tokens * self._compressor.THRESHOLD
                if est >= threshold:
                    # Flush memories before compression (user-role sentinel)
                    from context_compression import flush_memories
                    flush_memories(self, self.messages)
                    self.messages = self._compressor.maybe_compress(self.messages)

            # Prepare messages (strip internal fields for API)
            api_messages = self._prepare_api_messages()

            # Prompt caching (Chapter 6)
            if self._enable_prompt_caching:
                from prompt_caching import apply_prompt_caching
                api_messages = apply_prompt_caching(api_messages)

            kwargs = {
                "model": self.model,
                "messages": api_messages,
                "max_tokens": self.max_tokens,
            }
            # Strategy decides whether to pass tools as API param
            # or inject them into the system prompt.
            # Read live from registry so newly created tools appear immediately.
            tools = self._registry.get_schemas() if self._registry else self.tools
            kwargs = self._strategy.prepare_kwargs(kwargs, tools)

            return self.client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return None

    def _prepare_api_messages(self) -> list[dict]:
        """Prepare messages for API call, stripping internal fields."""
        api_msgs = []
        for msg in self.messages:
            api_msg = {}
            api_msg["role"] = msg["role"]
            if msg.get("content") is not None:
                api_msg["content"] = msg["content"]
            if msg.get("tool_calls"):
                api_msg["tool_calls"] = msg["tool_calls"]
            if msg.get("tool_call_id"):
                api_msg["tool_call_id"] = msg["tool_call_id"]
            # For assistant messages with tool_calls but empty content,
            # some APIs need content to be empty string not None
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                api_msg["content"] = msg.get("content") or ""
            api_msgs.append(api_msg)
        return api_msgs

    def _execute_tool(self, name: str, args: dict) -> str:
        # Look up live from registry (picks up newly created tools),
        # fall back to static handlers dict for backward compat.
        if self._registry:
            handler = self._registry.get_handlers().get(name)
        else:
            handler = self._tool_handlers.get(name)
        if not handler:
            return f"Error: unknown tool '{name}'"
        try:
            result = handler(**args)
            result_str = str(result)
            if len(result_str) > 50000:
                result_str = result_str[:50000] + "\n... [truncated]"
            return result_str
        except Exception as e:
            return f"Error executing {name}: {e}"

    def _persist_message(self, role: str, content: str,
                         tool_name: str = None, tool_call_id: str = None):
        if self.session_db and self.session_id:
            try:
                self.session_db.append_message(
                    self.session_id, role, content,
                    tool_name=tool_name, tool_call_id=tool_call_id,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Background review (Chapter 13)
    # ------------------------------------------------------------------

    _MEMORY_REVIEW = (
        "Review the conversation above. Has the user revealed preferences, "
        "habits, or personal details worth remembering? Has the user expressed "
        "expectations about how you should behave? If so, save using the "
        "memory tool. If nothing is worth saving, say 'Nothing to save.' and stop."
    )

    _SKILL_REVIEW = (
        "Review the conversation above. Was a non-trivial approach used that "
        "required trial and error or course corrections? If a relevant skill "
        "exists, update it. Otherwise, create a new one if reusable. If nothing "
        "is worth saving, say 'Nothing to save.' and stop."
    )

    _COMBINED_REVIEW = (
        "Review the conversation above.\n\n"
        "**Memory**: User preferences, habits, corrections? Save with memory tool.\n\n"
        "**Skills**: Non-trivial approach worth reusing? Create or update a skill.\n\n"
        "Only act if genuinely worth saving. Say 'Nothing to save.' if not."
    )

    def _spawn_background_review(self, review_memory: bool,
                                 review_skills: bool):
        """Fork a review agent on a background thread (Chapter 13)."""
        if review_memory and review_skills:
            prompt = self._COMBINED_REVIEW
        elif review_memory:
            prompt = self._MEMORY_REVIEW
        else:
            prompt = self._SKILL_REVIEW

        # Snapshot current conversation
        messages_snapshot = [m.copy() for m in self.messages]

        def _run():
            try:
                review_agent = Agent(
                    client=self.client,
                    model=self.model,
                    system_prompt=self.system_prompt,
                    tools=self.tools,
                    max_iterations=6,
                    max_tokens=self.max_tokens,
                )
                review_agent.set_handlers(self._tool_handlers)
                # Disable nudges on review agent (no recursion)
                review_agent._memory_nudge_interval = 0
                review_agent._skill_nudge_interval = 0
                # Load conversation snapshot
                review_agent.messages = messages_snapshot
                # Run review
                review_agent.run(prompt)
            except Exception as e:
                logger.debug("Background review failed: %s", e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        logger.info("Background review spawned (memory=%s, skills=%s)",
                     review_memory, review_skills)
