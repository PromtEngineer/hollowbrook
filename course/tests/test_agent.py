"""Tests for llm.agent (Chapters 24–28). Everything uses ScriptedBackend: no model, no network."""
from __future__ import annotations

import os
import sys
import textwrap

import pytest

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

from llm.agent.tools import Tool, ToolRegistry, make_builtin_tools, safe_eval  # noqa: E402
from llm.agent.backends import ScriptedBackend, ToolCall, AssistantMessage, AnthropicBackend, TinyLMBackend  # noqa: E402
from llm.agent.context import ContextBudget, MemoryFile, compact, estimate_tokens, truncate_tool_result  # noqa: E402
from llm.agent.harness import Agent, AgentConfig, Hooks, run_subagent  # noqa: E402
from llm.agent.mcp_mini import MCPClient, MCPServer, mcp_tools_to_registry  # noqa: E402
from llm.agent.miniharness import MiniHarness  # noqa: E402
from llm.agent.multiagent import orchestrate, generator_evaluator_loop, debate  # noqa: E402


def call(name, **args):
    return {"text": "", "tool_calls": [{"name": name, "arguments": args}]}


# ------------------------------------------------------------------ tools
def test_registry_validation_and_errors_as_text():
    reg = ToolRegistry([Tool("echo", "Echo x", {"type": "object", "properties": {"x": {"type": "string"}},
                                                "required": ["x"]}, lambda x: x)])
    assert reg.schemas()[0]["input_schema"]["required"] == ["x"]
    assert reg.call("echo", {"x": "hi"}) == "hi"
    assert "missing required argument 'x'" in reg.call("echo", {})
    assert "unknown argument" in reg.call("echo", {"x": "a", "y": 1})
    assert "should be of type string" in reg.call("echo", {"x": 3})
    assert "unknown tool" in reg.call("nope", {})
    boom = Tool("boom", "raises", {"type": "object", "properties": {}}, lambda: 1 / 0)
    reg.register(boom)
    assert "ZeroDivisionError" in reg.call("boom", {})   # never raises


def test_sandbox_rejects_escape(tmp_path):
    reg = make_builtin_tools(str(tmp_path))
    (tmp_path / "a.txt").write_text("hello")
    assert reg.call("read_file", {"path": "a.txt"}) == "hello"
    out = reg.call("read_file", {"path": "../../etc/passwd"})
    assert "escapes the sandbox" in out
    assert "escapes the sandbox" in reg.call("write_file", {"path": "/tmp/x", "content": "x"})
    assert "a.txt" in reg.call("list_dir", {})
    assert "a.txt:1: hello" in reg.call("search", {"pattern": "hel+o"})


def test_calculator_safe_eval():
    reg = make_builtin_tools(os.getcwd())
    assert reg.call("calculator", {"expression": "(2 + 3) * 4"}) == "20"
    assert reg.call("calculator", {"expression": "2 ** 10"}) == "1024"
    assert "Error" in reg.call("calculator", {"expression": "__import__('os').system('ls')"})
    assert "Error" in reg.call("calculator", {"expression": "abs(-1)"})
    with pytest.raises(ValueError):
        safe_eval("9 ** 9 ** 9")


def test_run_python_tool(tmp_path):
    reg = make_builtin_tools(str(tmp_path))
    assert "42" in reg.call("run_python", {"code": "print(6*7)"})


# ---------------------------------------------------------------- backends
def test_scripted_backend_coercion():
    b = ScriptedBackend(["hi", call("calculator", expression="1+1"), AssistantMessage(text="x")])
    m = b.complete([], [], "")
    assert m.text == "hi" and m.tool_calls == []
    m = b.complete([], [], "")
    assert m.tool_calls[0].name == "calculator" and m.tool_calls[0].id.startswith("call_")
    assert b.complete([], [], "").text == "x"
    assert b.complete([], [], "").text == "(script exhausted)"
    assert len(b.calls) == 4


def test_anthropic_backend_needs_package_and_format():
    with pytest.raises((ImportError, RuntimeError)):
        AnthropicBackend()
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "t", "tool_calls": [{"id": "c1", "name": "f", "arguments": {"a": 1}}]},
            {"role": "tool_result", "tool_call_id": "c1", "content": "r"}]
    api = AnthropicBackend.to_api_messages(msgs)
    assert api[1]["content"][1] == {"type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1}}
    assert api[2] == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "r"}]}


def test_tinylm_backend_message_mapping():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "f", "arguments": {}}]},
            {"role": "tool_result", "tool_call_id": "c1", "content": "r"}]
    chat = TinyLMBackend.to_chat_messages(msgs, [{"name": "f"}], "sys")
    assert [m["role"] for m in chat] == ["system", "user", "tool_call", "tool_result"]
    from llm.chat import render
    assert "<|tool_result|>r<|end|>" in render(chat)


# ------------------------------------------------------------------ agent
def test_agent_loop_tool_call_then_answer(tmp_path):
    tools = make_builtin_tools(str(tmp_path))
    backend = ScriptedBackend([
        {"text": "Computing.", "tool_calls": [{"name": "calculator", "arguments": {"expression": "6*7"}, "id": "c1"}]},
        "The answer is 42.",
    ])
    t = Agent(backend, tools, AgentConfig(permission_policy="allow_read_only")).run("What is 6*7?")
    assert t.stop_reason == "done" and t.turns == 2 and t.tool_calls_made == 1
    assert t.final_text == "The answer is 42."
    assert [m["role"] for m in t.messages] == ["user", "assistant", "tool_result", "assistant"]
    assert t.messages[2] == {"role": "tool_result", "tool_call_id": "c1", "content": "42"}
    assert [e.kind for e in t.events] == ["assistant", "tool_call", "tool_result", "assistant", "done"]
    # the second model call saw the tool result
    assert backend.calls[1]["messages"][2]["content"] == "42"
    assert "42" in t.pretty()


def test_permission_policy_denies_write(tmp_path):
    tools = make_builtin_tools(str(tmp_path))
    backend = ScriptedBackend([call("write_file", path="x.txt", content="hi"), "ok"])
    t = Agent(backend, tools, AgentConfig(permission_policy="allow_read_only")).run("write")
    assert t.messages[2]["content"].startswith("Permission denied")
    assert "permission_denied" in [e.kind for e in t.events]
    assert not (tmp_path / "x.txt").exists()
    # "ask" policy defers to permission_fn
    backend = ScriptedBackend([call("write_file", path="x.txt", content="hi"), "ok"])
    t = Agent(backend, tools, AgentConfig(permission_policy="ask")).run("write", permission_fn=lambda c, tool: True)
    assert (tmp_path / "x.txt").read_text() == "hi"


def test_pre_tool_hook_blocks_and_post_hook_rewrites(tmp_path):
    tools = make_builtin_tools(str(tmp_path))
    hooks = Hooks()
    hooks.pre_tool.append(lambda c: "no calculators today" if c.name == "calculator" else None)
    hooks.post_tool.append(lambda c, r: r + " [logged]")
    seen = []
    hooks.on_event.append(lambda e: seen.append(e.kind))
    backend = ScriptedBackend([call("calculator", expression="1+1"), "done"])
    t = Agent(backend, tools, AgentConfig(permission_policy="allow_all"), hooks=hooks).run("x")
    assert t.messages[2]["content"] == "Blocked by hook: no calculators today [logged]"
    assert "hook" in seen


def test_max_turns_and_unknown_tool(tmp_path):
    tools = make_builtin_tools(str(tmp_path))
    backend = ScriptedBackend([call("nope")] * 5)
    t = Agent(backend, tools, AgentConfig(max_turns=2, permission_policy="allow_all")).run("x")
    assert t.stop_reason == "max_turns" and t.turns == 2
    assert "unknown tool" in t.messages[2]["content"]


def test_compaction_keeps_system_and_tail(tmp_path):
    tools = make_builtin_tools(str(tmp_path))
    (tmp_path / "big.txt").write_text("x" * 1500)
    script = [call("read_file", path="big.txt")] * 6 + ["done"]
    cfg = AgentConfig(permission_policy="allow_all", context_budget_tokens=1200, compaction_keep_last=2)
    t = Agent(ScriptedBackend(script), tools, cfg).run("read the big file many times")
    kinds = [e.kind for e in t.events]
    assert "compaction" in kinds
    assert t.messages[0] == {"role": "user", "content": "read the big file many times"}
    assert any(m["content"].startswith("[tool result truncated") for m in t.messages if m["role"] == "tool_result")
    ev = next(e for e in t.events if e.kind == "compaction")
    assert ev.data["tokens_after"] < ev.data["tokens_before"]


def test_compact_function_and_budget():
    msgs = [{"role": "user", "content": "task"}] + [
        {"role": "tool_result", "tool_call_id": str(i), "content": "r" * 100} for i in range(10)]
    out = compact(msgs, keep_last=3, summarizer=lambda older: f"{len(older)} older turns")
    assert out[0]["content"] == "task"
    assert out[1]["content"].startswith("[Summary of earlier work]") and "7 older turns" in out[1]["content"]
    assert out[-3:] == msgs[-3:] and len(out) == 5
    b = ContextBudget(100)
    assert not b.needs_compaction(msgs[:1]) and b.needs_compaction(msgs)
    assert estimate_tokens("a" * 40) == 10
    tr = truncate_tool_result("h" * 50 + "t" * 50, max_chars=20)
    assert tr.startswith("h" * 10) and tr.endswith("t" * 10) and "truncated" in tr


def test_memory_file_round_trip(tmp_path):
    mem = MemoryFile(str(tmp_path / "MEMORY.md"))
    assert mem.read() == "" and mem.render_for_prompt() == ""
    mem.append("first note")
    mem.append("second note")
    assert mem.read() == "first note\nsecond note\n"
    assert "second note" in mem.render_for_prompt()
    backend = ScriptedBackend(["ok"])
    Agent(backend, ToolRegistry(), AgentConfig(memory_path=mem.path)).run("x")
    assert "first note" in backend.calls[0]["system"]


def test_run_subagent_isolates_context(tmp_path):
    tools = make_builtin_tools(str(tmp_path))
    backend = ScriptedBackend([call("calculator", expression="2+2"), "child says 4"])
    parent = Agent(backend, tools, AgentConfig(permission_policy="allow_all"))
    out = run_subagent(parent, "compute", tools_subset=["calculator"], max_turns=3)
    assert out == "child says 4"
    assert [t["name"] for t in backend.calls[0]["tools"]] == ["calculator"]


# -------------------------------------------------------------------- MCP
def test_mcp_server_handle_in_process(tmp_path):
    srv = MCPServer(make_builtin_tools(str(tmp_path)))
    init = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init["result"]["protocolVersion"] == "2025-11-25"
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    r = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "calculator", "arguments": {}}})
    assert r["result"]["isError"] is True
    assert "error" in srv.handle({"jsonrpc": "2.0", "id": 3, "method": "bogus"})


def test_mcp_client_subprocess(tmp_path):
    env = dict(os.environ, PYTHONPATH=COURSE_DIR)
    client = MCPClient([sys.executable, "-m", "llm.agent.mcp_mini", "--serve", "--workdir", str(tmp_path)],
                       cwd=COURSE_DIR, env=env)
    try:
        info = client.initialize()
        assert info["serverInfo"]["name"] == "llm-agent-mini"
        assert client.ping()
        names = [t["name"] for t in client.list_tools()]
        assert "calculator" in names and "inputSchema" in client.list_tools()[0]
        assert client.call_tool("calculator", {"expression": "3*3"}) == "9"
        # an Agent using MCP tools transparently
        reg = mcp_tools_to_registry(client)
        backend = ScriptedBackend([call("calculator", expression="5*5"), "25"])
        t = Agent(backend, reg, AgentConfig(permission_policy="allow_all")).run("x")
        assert t.messages[2]["content"] == "25"
    finally:
        client.close()
    assert client.proc.returncode == 0


# ------------------------------------------------------------ MiniHarness
def _tiny_repo(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")


def test_miniharness_plan_session_verify(tmp_path):
    _tiny_repo(tmp_path)
    backend = ScriptedBackend([
        "1. Read tests. 2. Fix code.",                          # planner reply
        call("write_file", path="PLAN.md", content="hack"),    # blocked by hook
        call("write_file", path="hello.py", content="x = 1\n"),
        "I wrote hello.py and the tests pass.",
    ])
    h = MiniHarness(backend, str(tmp_path), AgentConfig(max_turns=5))
    t = h.run_session(task="add hello.py")
    assert (tmp_path / "PLAN.md").read_text().startswith("# Plan")
    assert "1. Read tests" in (tmp_path / "PLAN.md").read_text()
    assert "hack" not in (tmp_path / "PLAN.md").read_text()
    assert t.messages[2]["content"].startswith("Blocked by hook")
    assert (tmp_path / "hello.py").read_text() == "x = 1\n"
    progress = (tmp_path / "PROGRESS.md").read_text()
    assert "Session 1" in progress and "Verification: PASS" in progress
    assert "write_file" in (tmp_path / "harness.log").read_text()
    ok, report = h.verify()
    assert ok and report.startswith("PASS")
    # resume: a second session sees PROGRESS.md in its prompt
    t2 = h.resume(max_turns=2)
    assert "Session 1" in backend.calls[-1]["messages"][0]["content"]
    assert "Session 2" in (tmp_path / "PROGRESS.md").read_text()


# ------------------------------------------------------------- multiagent
def test_orchestrate_and_generator_evaluator(tmp_path):
    tools = make_builtin_tools(str(tmp_path))
    orch = ScriptedBackend(['["compute 1+1", "compute 2+2"]', "merged: 2 and 4"])
    worker = ScriptedBackend(["w1", "w2"])
    out = orchestrate(orch, worker, "do sums", tools, n_workers=2)
    assert out == "merged: 2 and 4"
    assert "Report: w" in orch.calls[1]["messages"][0]["content"]
    # planner_fn bypasses the orchestrator's split
    out = orchestrate(ScriptedBackend(["m"]), ScriptedBackend(["a", "b"]), "t", tools, planner_fn=lambda t: ["x", "y"])
    assert out == "m"

    gen = ScriptedBackend(["draft 1", "draft 2"])
    ev = ScriptedBackend(["Too short.", "ACCEPT"])
    result, rounds = generator_evaluator_loop(gen, ev, "write", tools, max_rounds=3)
    assert result == "draft 2" and rounds == 2
    assert "Too short." in gen.calls[1]["messages"][0]["content"]
    result, rounds = generator_evaluator_loop(ScriptedBackend(["ok"]), ScriptedBackend([]), "t", tools,
                                              accept_fn=lambda s: s == "ok")
    assert (result, rounds) == ("ok", 1)
    answers = debate([ScriptedBackend(["a", "a'"]), ScriptedBackend(["b", "b'"])], "q", rounds=2)
    assert answers == ["a'", "b'"]
