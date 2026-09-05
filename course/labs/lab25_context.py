"""Lab 25: context engineering — measure the window, compact it, remember outside it.

(a) A scripted 30-turn, tool-heavy episode fills a context: we plot the usage curve and
    show what happens when a window is exceeded with no compaction (the model goes silent).
(b) The same episode with compaction on: which turns fire a compaction, what survives,
    and what is lost — including a planted fact that a plain compaction throws away and a
    summariser keeps.
(c) A memory file: an agent writes "notes to its future self", and a *fresh* agent in a
    second run reads them from its system prompt.
(d) A sub-agent isolates a noisy search: the child's window grows, the parent's does not.
(e) Prompt caching: how much of each request is a byte-identical prefix of the previous one,
    with and without a timestamp in the system prompt, and across a compaction.

Run:  python3 labs/lab25_context.py            (quick, ~10 s)
      python3 labs/lab25_context.py --full     (adds a longer 90-turn episode)
"""
from _common import setup, check, banner, section, savefig, done, plt

import json
import os
import shutil
import tempfile
import time

from llm.agent import (Agent, AgentConfig, ContextBudget, Hooks, MemoryFile, ScriptedBackend, Tool,
                       ToolRegistry, compact, estimate_tokens, make_builtin_tools, run_subagent, DEFAULT_SYSTEM)
from llm.agent.context import message_text

args = setup("Lab 25: context engineering")
BOX = tempfile.mkdtemp(prefix="lab25_")
N_TURNS = 90 if args.full else 30
WINDOW = 8000                                        # AgentConfig's default context_budget_tokens


def call(name, **arguments):
    return {"text": "", "tool_calls": [{"name": name, "arguments": arguments}]}


# ------------------------------------------------------------------ a sandbox with files of different sizes
os.makedirs(os.path.join(BOX, "data"), exist_ok=True)
with open(os.path.join(BOX, "config.txt"), "w") as f:
    f.write("service=billing\ndb_port=4242\nretries=3\n")                  # the planted fact: 4242
with open(os.path.join(BOX, "log.txt"), "w") as f:
    for i in range(120):
        f.write(f"2026-09-05 10:{i % 60:02d}:00 {'ERROR' if i % 10 == 7 else 'INFO '} worker-{i % 4} handled request {1000 + i}\n")
for k in range(4):
    with open(os.path.join(BOX, "data", f"batch{k}.csv"), "w") as f:
        f.write("id,amount,status\n" + "\n".join(f"{k * 100 + i},{(i * 37) % 500},{'ok' if i % 5 else 'failed'}" for i in range(60)))
tools = make_builtin_tools(BOX)


def episode_script(n_turns: int) -> list:
    """Turn 1 reads the config (the fact), turns 2..n-1 are noisy reads, turn n answers."""
    script = [call("read_file", path="config.txt")]
    noisy = [call("read_file", path="log.txt"), call("search", pattern="ERROR"), call("list_dir", path="data"),
             call("read_file", path="data/batch0.csv"), call("search", pattern="failed", path="data")]
    for i in range(n_turns - 2):
        script.append(noisy[i % len(noisy)])
    script.append("The billing service listens on db_port 4242; the log has 12 ERROR lines.")
    return script


# =============================================================== (a) filling the window
section(f"(a) a {N_TURNS}-turn tool-heavy episode with compaction OFF")
no_compact = AgentConfig(permission_policy="allow_read_only", max_turns=N_TURNS + 5,
                         context_budget_tokens=10**9)          # a budget so large compaction never fires
t_off = Agent(ScriptedBackend(episode_script(N_TURNS)), tools, no_compact).run("Find the db port and count the errors.")
budget = ContextBudget(WINDOW, DEFAULT_SYSTEM, json.dumps(tools.schemas()))
print(f"   fixed cost per call (system prompt + 7 tool schemas): {budget.fixed} tokens")
usage_off = [budget.used(t_off.messages[:i]) for i in range(1, len(t_off.messages) + 1)]
by_role = {}
for m in t_off.messages:
    by_role[m["role"]] = by_role.get(m["role"], 0) + estimate_tokens(message_text(m))
print(f"   {t_off.turns} turns, {t_off.tool_calls_made} tool calls, {len(t_off.messages)} messages")
print(f"   tokens by role: {by_role}  (tool results dominate)")
print(f"   final usage {usage_off[-1]:,} tokens vs an {WINDOW:,}-token window: "
      f"{'OVER' if usage_off[-1] > WINDOW else 'under'} by {abs(usage_off[-1] - WINDOW):,}")
first_over = next((i for i, u in enumerate(usage_off) if u > WINDOW), None)
check(t_off.stop_reason == "done" and "compaction" not in [e.kind for e in t_off.events], "no compaction event fired with the huge budget")
check(first_over is not None, f"the {WINDOW}-token window is exceeded at message {first_over} of {len(t_off.messages)}")
largest = max(estimate_tokens(m["content"]) for m in t_off.messages if m["role"] == "tool_result")
check(largest <= 2000 // 4 + 20, f"every tool result was already capped at max_tool_result_chars=2000 (~{largest} tokens)")

print("\n   what an over-full window does to TinyLM (nano, 128-token window):")
from llm.pipeline import get_base_model              # noqa: E402  (torch import deferred until here)
from llm.agent import TinyLMBackend                   # noqa: E402
model, tok = get_base_model(quick=True, verbose=False)
tiny = TinyLMBackend(model, tok, max_new_tokens=16)
short = tiny.complete([{"role": "user", "content": "What is 2 + 3?"}], [], "Answer briefly.")
long_ = tiny.complete(t_off.messages[:8], [], "Answer briefly.")
print(f"   short prompt ({len(tok.encode('What is 2 + 3?'))} tokens): raw reply {short.raw!r}")
print(f"   8 messages ({budget.used(t_off.messages[:8]) - budget.fixed:,} est. tokens): raw reply {long_.raw!r}")
check(long_.raw == "" and short.raw != "", "past its window the model returns an EMPTY reply, which the loop would read as 'done'")


# =============================================================== (b) compaction on
section("(b) the same episode with compaction ON (budget 8000, threshold 0.8, keep_last 6)")
usage_on, comp_turns = [], []
hooks = Hooks()
hooks.on_event.append(lambda e: comp_turns.append(e.turn) if e.kind == "compaction" else None)
on = AgentConfig(permission_policy="allow_read_only", max_turns=N_TURNS + 5, context_budget_tokens=WINDOW)
backend_on = ScriptedBackend(episode_script(N_TURNS))
t_on = Agent(backend_on, tools, on, hooks=hooks).run("Find the db port and count the errors.")
usage_on = [budget.used(c["messages"]) for c in backend_on.calls]     # what each model call actually saw
comp_events = [e for e in t_on.events if e.kind == "compaction"]
for e in comp_events[:6]:
    print(f"   turn {e.turn:2d}: compaction {e.data['tokens_before']:,} -> {e.data['tokens_after']:,} tokens")
if len(comp_events) > 6:
    print(f"   ... {len(comp_events) - 6} more")
stubs = [m for m in t_on.messages if m["role"] == "tool_result" and m["content"].startswith("[tool result truncated")]
print(f"   {len(comp_events)} compactions; final window {usage_on[-1]:,} tokens; "
      f"{len(stubs)} of {t_on.tool_calls_made} tool results are now stubs; peak seen by the model {max(usage_on):,}")
check(len(comp_events) >= 1, f"compaction fired {len(comp_events)} time(s)")
check(max(usage_on) <= WINDOW, f"the model never saw more than {max(usage_on):,} tokens (window {WINDOW:,})")
check(t_on.messages[0]["content"] == "Find the db port and count the errors.", "the task (first message) survived every compaction")
survived = any("4242" in m["content"] for m in t_on.messages if m["role"] == "tool_result")
check(not survived, "the planted fact (db_port=4242, read at turn 1) did NOT survive: its tool result became a stub")
print(f"   the final answer still says 4242 only because it was scripted; a real model would have to guess")

# compaction with a summariser keeps facts, at the cost of a summarising call
def rule_summariser(older: list[dict]) -> str:
    """A stand-in for a model call: list what was done and keep any key=value facts."""
    calls = [f"{c['name']}({json.dumps(c['arguments'])})" for m in older if m["role"] == "assistant"
             for c in m.get("tool_calls", [])]
    facts = sorted({line.strip() for m in older if m["role"] == "tool_result"
                    for line in m["content"].splitlines() if "=" in line and len(line) < 40})
    return f"Ran {len(calls)} tool calls ({', '.join(sorted(set(c.split('(')[0] for c in calls)))}). Facts: {'; '.join(facts)}"


summarised = compact(t_off.messages, keep_last=6, summarizer=rule_summariser)
print(f"   with a summariser: {len(t_off.messages)} messages -> {len(summarised)}; {budget.used(t_off.messages):,} -> {budget.used(summarised):,} tokens")
print(f"   summary message: {summarised[1]['content'][:160]!r}")
check("4242" in summarised[1]["content"], "the summariser carried db_port=4242 across the compaction")


# =============================================================== (c) memory file
section("(c) a memory file: notes to my future self, across two separate runs")
mem_path = os.path.join(BOX, "MEMORY.md")
mem = MemoryFile(mem_path)
tools_mem = make_builtin_tools(BOX)
tools_mem.register(Tool("remember", "Append a short note to the memory file so future sessions can read it.",
                        {"type": "object", "properties": {"note": {"type": "string", "description": "one line"}},
                         "required": ["note"]}, lambda note: (mem.append(note), f"remembered: {note}")[1], read_only=False))
run1 = ScriptedBackend([call("read_file", path="config.txt"),
                        call("remember", note="billing service: db_port=4242 (from config.txt)"),
                        "Noted the port for next time."])
t1 = Agent(run1, tools_mem, AgentConfig(permission_policy="allow_all", memory_path=mem_path)).run("Look up the db port and remember it.")
print("   run 1 system prompt had memory block:", "Memory" in run1.calls[0]["system"])
print(f"   MEMORY.md now: {mem.read()!r}")
run2 = ScriptedBackend(["The db port is 4242 (from my notes)."])          # a fresh agent, empty context
t2 = Agent(run2, tools_mem, AgentConfig(permission_policy="allow_all", memory_path=mem_path)).run("What is the db port?")
print("   run 2 system prompt tail:\n      " + run2.calls[0]["system"].splitlines()[-2].strip() + "\n      " + run2.calls[0]["system"].splitlines()[-1].strip())
check("Memory" not in run1.calls[0]["system"], "run 1 started with no memory block (the file did not exist yet)")
check("db_port=4242" in run2.calls[0]["system"], "run 2 read the note from MEMORY.md in its system prompt without any tool call")
check(t2.tool_calls_made == 0, "run 2 answered with zero tool calls")


# =============================================================== (d) sub-agent isolation
section("(d) a sub-agent keeps a noisy search out of the parent's window")
shared = ScriptedBackend([
    call("delegate", task="Find every ERROR line in log.txt and tell me how many and from which workers."),   # parent, turn 1
    call("search", pattern="ERROR"),                                                                    # child, turn 1
    call("read_file", path="log.txt"),                                                                  # child, turn 2
    call("search", pattern="ERROR worker-1"),                                                           # child, turn 3
    "12 ERROR lines; all four workers appear (worker-0..3), three each.",                                # child, final
    "The log has 12 ERROR lines, spread evenly over the four workers.",                                  # parent, turn 2
])
parent_tools = make_builtin_tools(BOX)
parent = Agent(shared, parent_tools, AgentConfig(permission_policy="allow_read_only", max_turns=6))
parent_tools.register(Tool("delegate", "Hand a research sub-task to a sub-agent; returns only its final paragraph.",
                           {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
                           lambda task: run_subagent(parent, task, tools_subset=["search", "read_file"], max_turns=6)))
tp = parent.run("How many errors are in the log, and from which workers?")
per_call = [budget.used(c["messages"]) - budget.fixed for c in shared.calls]
who = ["parent", "child", "child", "child", "child", "parent"]
for i, (w, n, c) in enumerate(zip(who, per_call, shared.calls)):
    print(f"   call {i}: {w:6s} saw {len(c['messages']):2d} messages, ~{n:5,d} tokens, tools offered: {[t['name'] for t in c['tools']]}")
check(tp.final_text.startswith("The log has 12"), "the parent answered from the child's one-paragraph report")
check(max(per_call[1:5]) > 5 * per_call[5], f"the child's window grew to {max(per_call[1:5]):,} tokens; the parent's stayed at {per_call[5]:,}")
check(shared.calls[1]["tools"] and len(shared.calls[1]["tools"]) == 2, "the child was offered only the 2 tools in tools_subset")
check(len(tp.messages) == 4, f"the parent's transcript has {len(tp.messages)} messages: task, delegate call, one result, answer")


# =============================================================== (e) prompt-cache stable prefix
section("(e) prompt caching: how much of each request is a byte-identical prefix of the last one?")


def rendered(c: dict) -> str:
    """What a provider hashes: system + tools + messages, in order."""
    return c["system"] + "\n" + json.dumps(c["tools"]) + "\n" + json.dumps(c["messages"])


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def cache_report(calls: list[dict], label: str, show: int = 4) -> tuple[float, float]:
    """For consecutive requests: how much of the PREVIOUS request is reused verbatim (coverage),
    and what fraction of the NEW request that prefix is (the cacheable share of the bill)."""
    coverages, shares = [], []
    print(f"   {label}")
    for k in range(1, len(calls)):
        prev, cur = rendered(calls[k - 1]), rendered(calls[k])
        lcp = common_prefix(prev, cur)
        coverage, share = lcp / len(prev), lcp / len(cur)
        coverages.append(coverage)
        shares.append(share)
        if k <= show or coverage < 0.5:
            print(f"      call {k - 1}->{k}: prev {len(prev):6,d} chars, new {len(cur):6,d}, common prefix {lcp:6,d} "
                  f"= {100 * coverage:5.1f}% of prev reused, {100 * share:5.1f}% of the new request cacheable")
    print(f"      over {len(shares)} calls: min reuse of the previous request {100 * min(coverages):.1f}%, "
          f"mean cacheable share of each new request {100 * sum(shares) / len(shares):.1f}%")
    return min(coverages), sum(shares) / len(shares)


class Timestamped:
    """A backend wrapper that puts the current time in the system prompt on every call."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def complete(self, messages, tools, system):
        stamped = f"Current time: {time.time():.6f}\n" + system
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools, "system": stamped})
        return self.inner.complete(messages, tools, stamped)


static = ScriptedBackend(episode_script(12))
Agent(static, tools, no_compact).run("Find the db port and count the errors.")
min_static, share_static = cache_report(static.calls, "A. static system prompt, append-only history (compaction off)")
stamped = Timestamped(ScriptedBackend(episode_script(12)))
Agent(stamped, tools, no_compact).run("Find the db port and count the errors.")
min_stamp, share_stamp = cache_report(stamped.calls, "B. a timestamp at the top of the system prompt")
min_comp, share_comp = cache_report(backend_on.calls, "C. static prompt, compaction on (from part b)", show=2)
check(min_static > 0.99, f"A: every request reuses >99% of the previous one (all but the closing bracket); "
                          f"{100 * share_static:.0f}% of each new request is cacheable")
check(min_stamp < 0.02, f"B: the timestamp breaks the prefix at byte {common_prefix(rendered(stamped.calls[0]), rendered(stamped.calls[1]))}: "
                        f"{100 * share_stamp:.1f}% cacheable, every call pays full price")
check(min_comp < 0.5, f"C: the compaction turn rewrote the history: only {100 * min_comp:.1f}% of the previous request was reusable")

# ------------------------------------------------------------------ figure
fig, axes = plt().subplots(1, 2, figsize=(11, 3.6))
ax = axes[0]
ax.plot(range(1, len(usage_off) + 1), usage_off, color="#dc2626", label="compaction off (per message)")
ax.plot(range(1, len(usage_on) + 1), usage_on, color="#2563eb", marker=".", label="compaction on (per model call)")
ax.axhline(WINDOW, color="#0f172a", ls="--", lw=1, label=f"window {WINDOW}")
ax.axhline(0.8 * WINDOW, color="#f59e0b", ls=":", lw=1, label="compaction threshold 80%")
ax.set_xlabel("message / call index"); ax.set_ylabel("estimated tokens in context"); ax.legend(fontsize=8)
ax.set_title(f"{N_TURNS}-turn episode: context usage")
ax = axes[1]
ax.bar(range(len(per_call)), per_call, color=["#2563eb" if w == "parent" else "#f59e0b" for w in who])
ax.set_xticks(range(len(per_call))); ax.set_xticklabels([f"{i}\n{w}" for i, w in enumerate(who)], fontsize=8)
ax.set_ylabel("tokens seen (excluding fixed cost)"); ax.set_title("sub-agent isolation: parent (blue) vs child (orange)")
savefig(fig, "lab25_context.png")

shutil.rmtree(BOX, ignore_errors=True)
done()
