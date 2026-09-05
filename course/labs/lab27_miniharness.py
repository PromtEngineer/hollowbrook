"""Lab 27: a long-running coding-agent harness fixes a failing test in a sandbox repo.

    python3 labs/lab27_miniharness.py            # quick (default): scripted runs + nano TinyLM
    python3 labs/lab27_miniharness.py --full     # the same, with the small TinyLM base model

What you will see, in order:
  0. the permission gate on its own (policy vs. a human-in-the-loop function);
  1. an honest scripted agent: plan -> read the test -> edit the module -> the harness verifies;
  2. an over-claiming agent: it says "done" without fixing anything; the verifier catches it;
     a resume() session then reads PROGRESS.md and finishes the job;
  3. a hook that refuses edits to tests/ (the agent may not "fix" a test by deleting it);
  4. observability and cost: harness.log, the event stream, an estimate of tokens per call;
  5. the real TinyLM base model driving the same harness: it cannot use tools, and the
     harness contains that failure without crashing.

Every "model" here except section 5 is a ``ScriptedBackend``: the replies are fixed, so
the lab is deterministic and the harness mechanics are the only moving part.
"""
from _common import setup, check, banner, section, savefig, done, plt

import json
import os
import shutil
import textwrap
import time

from llm.agent import (Agent, AgentConfig, Hooks, MiniHarness, ScriptedBackend, TinyLMBackend,
                       estimate_tokens, make_builtin_tools)
from llm.agent.context import message_text
from llm.pipeline import get_base_model, run_path

args = setup("Lab 27: harness engineering — MiniHarness fixes a failing test")
SANDBOX = run_path("lab27_sandbox")

# ----------------------------------------------------------- the sandbox repo
MATHLIB_BROKEN = '''"""mathlib: a tiny statistics module (the agent's sandbox repo)."""


def mean(xs):
    return sum(xs) / (len(xs) - 1)      # bug: should divide by len(xs)
'''

TEST_FILE = '''from mathlib import mean, median


def test_mean():
    assert mean([2, 4, 6]) == 4


def test_median_odd():
    assert median([3, 1, 2]) == 2


def test_median_even():
    assert median([1, 2, 3, 4]) == 2.5
'''

MATHLIB_FIXED = '''"""mathlib: a tiny statistics module."""


def mean(xs):
    return sum(xs) / len(xs)


def median(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
'''

MATHLIB_HALF_FIXED = '''"""mathlib: a tiny statistics module."""


def mean(xs):
    return sum(xs) / len(xs)


def median(xs):
    return sorted(xs)[len(xs) // 2]      # wrong for even-length input
'''


def fresh_sandbox(path: str = SANDBOX) -> str:
    """A repo with one module and one failing test file; wiped on every call."""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(os.path.join(path, "tests"))
    with open(os.path.join(path, "mathlib.py"), "w") as f:
        f.write(MATHLIB_BROKEN)
    with open(os.path.join(path, "tests", "test_math.py"), "w") as f:
        f.write(TEST_FILE)
    return path


def call(name, **arguments):
    """A scripted assistant turn that makes one tool call."""
    return {"text": "", "tool_calls": [{"name": name, "arguments": arguments}]}


def show(path: str, name: str, max_lines: int = 40) -> None:
    p = os.path.join(path, name)
    text = open(p).read() if os.path.exists(p) else "(missing)"
    lines = text.rstrip().splitlines()
    print(f"\n┌─ {name} ({len(lines)} lines)")
    for line in lines[:max_lines]:
        print("│ " + line)
    if len(lines) > max_lines:
        print(f"│ ... ({len(lines) - max_lines} more lines)")
    print("└" + "─" * 40)


def cost_of(backend: ScriptedBackend) -> tuple[int, int]:
    """(model calls, estimated input tokens) — what an API bill would be made of."""
    total = 0
    for c in backend.calls:
        total += estimate_tokens(c["system"]) + estimate_tokens(json.dumps(c["tools"]))
        total += sum(estimate_tokens(message_text(m)) for m in c["messages"])
    return len(backend.calls), total


TASK = "Make the tests in tests/ pass: mean() is wrong and median() is missing."

# ================================================================= 0. the gate
section("0. the permission gate: policy first, then a human-in-the-loop function")
box = fresh_sandbox()
tools = make_builtin_tools(box)
script = [call("write_file", path="notes.txt", content="hello"), "ok"]
t = Agent(ScriptedBackend(script), tools, AgentConfig(permission_policy="allow_read_only")).run("write a note")
print("policy allow_read_only, write_file ->", t.messages[2]["content"][:70], "...")
check(t.messages[2]["content"].startswith("Permission denied"), "allow_read_only denies write_file")
asked = []


def human(call_, tool):                       # a stand-in for the prompt a real harness shows
    asked.append((call_.name, tool.read_only))
    return not call_.arguments.get("path", "").startswith("tests")


t = Agent(ScriptedBackend(script), tools, AgentConfig(permission_policy="ask")).run("write a note", permission_fn=human)
print(f"policy ask, permission_fn approved -> {t.messages[2]['content']!r}; the function was asked about {asked}")
check(os.path.exists(os.path.join(box, "notes.txt")), "'ask' policy defers to permission_fn, which approved the write")
print("(the gate answers *may this run?*; nothing here checks whether the result was any good — that is the verifier's job)")

# ================================================================= 1. honest run
section("1. an honest agent: plan, read, edit, verify")
box = fresh_sandbox()
honest = ScriptedBackend([
    # --- planner (read-only tools, max 6 turns): looks around, then writes the plan as text
    call("list_dir", path="."),
    call("read_file", path="tests/test_math.py"),
    "1. Read mathlib.py.\n2. Fix mean(): divide by len(xs).\n3. Add median() that averages the two "
    "middle values for even n.\n4. Run the tests and stop when they pass.",
    # --- coding session 1
    call("read_file", path="mathlib.py"),
    call("write_file", path="mathlib.py", content=MATHLIB_FIXED),
    call("run_tests"),
    "Fixed mean() (divide by len(xs)) and added median() with the even-length case; run_tests reports 3 passed.",
])
h = MiniHarness(honest, box, AgentConfig(max_turns=8))
t0 = time.perf_counter()
ok = h.loop(TASK, max_sessions=2)
secs = time.perf_counter() - t0
t = h.sessions[0]
print(f"loop() -> {ok} after {len(h.sessions)} session(s) in {secs:.1f}s; session 1: stop_reason={t.stop_reason}, "
      f"turns={t.turns}, tool_calls={t.tool_calls_made}")
print("events:", " -> ".join(e.kind for e in t.events))
check(ok and len(h.sessions) == 1, "the honest agent passes verification in one session")
show(box, "PLAN.md")
show(box, "PROGRESS.md")
show(box, "harness.log", max_lines=6)
calls, toks = cost_of(honest)
print(f"cost: {calls} model calls, ≈{toks:,} estimated input tokens (chars/4 over every call's context)")
check(honest.calls[3]["messages"][0]["content"].startswith("PLAN.md:"), "the coding session's first message is PLAN.md + PROGRESS.md")

# ====================================================== 2. over-claiming + resume
section("2. an agent that claims success without evidence — and a resume() that finishes the job")


class GuardedHarness(MiniHarness):
    """MiniHarness plus one more pre-tool hook: the agent may not edit the tests."""

    def _hooks(self) -> Hooks:
        hooks = super()._hooks()

        def protect_tests(call_):
            target = os.path.normpath(call_.arguments.get("path", ""))
            if call_.name == "write_file" and target.split(os.sep)[0] == "tests":
                return "tests/ is read-only for the agent: fix the code, not the tests"
            return None

        hooks.pre_tool.insert(0, protect_tests)
        return hooks


box = fresh_sandbox()
liar = ScriptedBackend([
    # --- planner: does not even look
    "1. Fix mean. 2. Add median. 3. Run tests.",
    # --- session 1: tries to delete the failing test, half-fixes the module, claims victory
    call("write_file", path="tests/test_math.py", content="def test_ok():\n    assert True\n"),
    call("write_file", path="mathlib.py", content=MATHLIB_HALF_FIXED),
    "Done. mean() is fixed and median() is implemented; all tests pass.",
    # --- session 2 (resume): reads what the verifier wrote, finishes properly
    call("read_file", path="mathlib.py"),
    call("write_file", path="mathlib.py", content=MATHLIB_FIXED),
    call("run_tests"),
    "median() now averages the two middle values for even n; run_tests reports 3 passed.",
])
g = GuardedHarness(liar, box, AgentConfig(max_turns=6))
s1 = g.run_session(task=TASK)
ok1, report1 = g.verify()
print(f"session 1: stop_reason={s1.stop_reason}, the model's last words: {s1.final_text!r}")
print(f"verifier : {'PASS' if ok1 else 'FAIL'} — first line of the report: {report1.splitlines()[0]!r}")
blocked = [m["content"] for m in s1.messages if m["role"] == "tool_result" and m["content"].startswith("Blocked by hook")]
print(f"hook     : {blocked[0] if blocked else '(nothing blocked)'}")
check(not ok1, "the verifier rejects the over-claiming session (tests still fail)")
check(blocked and "tests/" in blocked[0], "the pre-tool hook blocked the edit to tests/")
check(open(os.path.join(box, "tests", "test_math.py")).read() == TEST_FILE, "the test file is untouched")
n_before = len(liar.calls)
s2 = g.resume()
ok2, _ = g.verify()
first_prompt = liar.calls[n_before]["messages"][0]["content"]
print(f"\nresume(): session 2 started with {len(liar.calls[n_before]['messages'])} message(s) in context "
      f"(a fresh window) whose prompt contains 'Verification: FAIL': {'Verification: FAIL' in first_prompt}")
print(f"session 2: {s2.stop_reason}, {s2.tool_calls_made} tool calls; verifier: {'PASS' if ok2 else 'FAIL'}")
check("Verification: FAIL" in first_prompt and "test_median_even" in first_prompt,
      "the resumed session saw the failing test's name through PROGRESS.md, not through memory")
check(ok2, "after the resumed session the tests pass")
show(box, "PROGRESS.md", max_lines=60)

# --------------------------------------------- resuming from a *new process*
section("2b. resuming from a new process: the files are the checkpoint")
g2 = GuardedHarness(ScriptedBackend(["Nothing left to do: PROGRESS.md says verification passed."]), box,
                    AgentConfig(max_turns=3))
s3 = g2.resume()
print(f"a brand-new harness object on the same directory: has_plan={g2.has_plan()}, session stop={s3.stop_reason}, "
      f"verify={'PASS' if g2.verify()[0] else 'FAIL'}")
headings = [l for l in open(os.path.join(box, "PROGRESS.md")).read().splitlines() if l.startswith("## Session")]
print("PROGRESS.md session headings now:", headings)
print("(note: the new object numbered its session from its own counter, not from the file — see the chapter)")
check(s3.stop_reason == "done" and g2.verify()[0], "a fresh process resumes from the files alone")

# ============================================================ 3. observability
section("3. observability: the event stream and the cost of every session")
kinds = {}
for tr in g.sessions:
    for e in tr.events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
print("event kinds over the two guarded sessions:", kinds)
log_lines = [json.loads(l) for l in open(os.path.join(box, "harness.log"))]
print(f"harness.log: {len(log_lines)} tool calls; tools used: {sorted({l['tool'] for l in log_lines})}")
per_call = [sum(estimate_tokens(message_text(m)) for m in c["messages"]) for c in liar.calls]
n_msgs = [len(c["messages"]) for c in liar.calls]
print("estimated context tokens per model call (planner | session 1 | session 2):", per_call)
print("messages in context per model call                                     :", n_msgs)
calls, toks = cost_of(liar)
print(f"total: {calls} model calls, ≈{toks:,} estimated input tokens")
print("(session 2's first call is *bigger* than session 1's last: PROGRESS.md carries the whole pytest report."
      " The history is what resets — one message — and the files bound how much comes back.)")
check(n_msgs[4] == 1 and n_msgs[3] > 1, "a new session starts from one message: the history is reset, the files persist")

# ================================================= 4. the real model, contained
section("4. the real TinyLM base model in the same harness (it has never seen a tool call)")
model, tok = get_base_model(quick=args.quick, verbose=False)
box = fresh_sandbox(run_path("lab27_sandbox_tinylm"))
from llm.chat import render
from llm.agent.miniharness import SESSION_SYSTEM
schemas = make_builtin_tools(box).schemas()
chat_msgs = TinyLMBackend.to_chat_messages([{"role": "user", "content": TASK}], schemas, SESSION_SYSTEM)
n_prompt = len(tok.encode(render(chat_msgs)))
print(f"the session prompt (system + {len(schemas)} tool schemas + task) is {n_prompt} tokens; "
      f"the {'nano' if args.quick else 'small'} model's window is {model.cfg.max_seq_len}")
print(f"-> with no room to decode, TinyLMBackend returns {TinyLMBackend(model, tok).complete([{'role': 'user', 'content': TASK}], schemas, SESSION_SYSTEM).raw!r} (silently)")
model.extend_context(2048)                    # positions it was never trained on; babble is the point


class RecordingTinyLM(TinyLMBackend):
    """TinyLMBackend that keeps the raw generations so we can look at them."""
    raws: list = []

    def complete(self, messages, tools, system):
        r = super().complete(messages, tools, system)
        self.raws.append(r.raw)
        return r


real = RecordingTinyLM(model, tok, max_new_tokens=40)
hr = GuardedHarness(real, box, AgentConfig(max_turns=3))
t0 = time.perf_counter()
ok_real = hr.loop(TASK, max_sessions=2)
secs_real = time.perf_counter() - t0
for i, tr in enumerate(hr.sessions, 1):
    print(f"session {i}: stop={tr.stop_reason}, turns={tr.turns}, tool calls={tr.tool_calls_made}, "
          f"said: {tr.final_text[:70]!r}")
print(f"raw generation of session 1: {real.raws[1][:110]!r}")
print(f"loop() -> {ok_real} in {secs_real:.1f}s (window extended to 2048 so the model could answer at all)")
check(not ok_real and all(tr.tool_calls_made == 0 for tr in hr.sessions),
      "a base model produces no tool calls; the harness records two failed sessions and stops")
check("Verification: FAIL" in open(os.path.join(box, "PROGRESS.md")).read(), "PROGRESS.md says FAIL, not the model")

# ----------------------------------------------------------------- figure
fig, ax = plt().subplots(figsize=(7, 3))
colors = ["#7c3aed"] + ["#2563eb"] * 3 + ["#16a34a"] * 4
ax.bar(range(1, len(per_call) + 1), per_call, color=colors[:len(per_call)])
ax.set_xlabel("model call (planner, then session 1, then session 2)")
ax.set_ylabel("est. context tokens")
ax.set_title("Lab 27: each session starts from an empty context; the files carry the state")
ax.axvline(1.5, color="#64748b", ls=":"); ax.axvline(4.5, color="#64748b", ls=":")
savefig(fig, "lab27_context_per_call.png")
done()
