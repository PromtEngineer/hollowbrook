"""llm.agent — a from-scratch agent harness (Part IV, Chapters 24–28).

The package is deliberately split into small files, one idea each:

    tools.py        Tool schemas, a registry, and sandboxed built-in tools   (Ch. 24, 26)
    backends.py     "Where does the next assistant message come from?"       (Ch. 24)
                    ScriptedBackend (tests), TinyLMBackend, AnthropicBackend
    harness.py      The agent loop: permissions, hooks, budget, sub-agents   (Ch. 24, 25, 27)
    context.py      Token budget, compaction, memory file                    (Ch. 25)
    mcp_mini.py     A minimal Model Context Protocol server and client       (Ch. 26)
    miniharness.py  A long-running coding-agent harness with PLAN/PROGRESS   (Ch. 27)
    multiagent.py   Orchestrator–workers, generator–evaluator, debate        (Ch. 28)

Quick start (fully deterministic, no model needed)::

    from llm.agent import Agent, AgentConfig, ScriptedBackend, ToolCall, make_builtin_tools
    tools = make_builtin_tools("/tmp/sandbox")
    backend = ScriptedBackend([
        {"text": "Let me compute that.", "tool_calls": [{"name": "calculator", "arguments": {"expression": "2+3"}}]},
        "2 + 3 = 5",
    ])
    transcript = Agent(backend, tools, AgentConfig()).run("What is 2+3?")
    print(transcript.pretty())
"""
from .tools import Tool, ToolRegistry, make_builtin_tools
from .backends import ToolCall, AssistantMessage, Backend, ScriptedBackend, TinyLMBackend, AnthropicBackend
from .context import estimate_tokens, ContextBudget, compact, truncate_tool_result, MemoryFile
from .harness import AgentConfig, Event, Hooks, Agent, Transcript, run_subagent, DEFAULT_SYSTEM
from .miniharness import MiniHarness
from .multiagent import orchestrate, generator_evaluator_loop, debate

# mcp_mini is imported lazily so `python3 -m llm.agent.mcp_mini` does not import the
# module twice (once via this package, once as __main__) and warn about it.
_LAZY_MCP = {"MCPServer", "MCPClient", "mcp_tools_to_registry"}


def __getattr__(name: str):
    if name in _LAZY_MCP:
        from . import mcp_mini
        return getattr(mcp_mini, name)
    raise AttributeError(f"module 'llm.agent' has no attribute '{name}'")


__all__ = [
    "Tool", "ToolRegistry", "make_builtin_tools",
    "ToolCall", "AssistantMessage", "Backend", "ScriptedBackend", "TinyLMBackend", "AnthropicBackend",
    "estimate_tokens", "ContextBudget", "compact", "truncate_tool_result", "MemoryFile",
    "AgentConfig", "Event", "Hooks", "Agent", "Transcript", "run_subagent", "DEFAULT_SYSTEM",
    "MCPServer", "MCPClient", "mcp_tools_to_registry",
    "MiniHarness",
    "orchestrate", "generator_evaluator_loop", "debate",
]
