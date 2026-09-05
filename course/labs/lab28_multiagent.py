"""Lab 28: multi-agent patterns — orchestrator–workers, generator/evaluator, debate.

    python3 labs/lab28_multiagent.py            # quick (default): scripted agents only
    python3 labs/lab28_multiagent.py --full     # + a wider fan-out and TinyLM agents debating

Every agent here is an ordinary ``Agent`` from Chapter 24. What makes a system
"multi-agent" is only the program around them: who sees what, and how the outputs are
combined. The lab measures the two things that decide whether a pattern is worth it:
wall-clock (parallel fan-out overlaps latency) and cost (every extra agent re-reads a
context). It also shows the failure modes: a sycophantic evaluator, and a debate in
which agreement is not correctness.
"""
from _common import setup, check, banner, section, savefig, done, plt

import json
import threading
import time

from llm.agent import (Agent, AgentConfig, AssistantMessage, ScriptedBackend, TinyLMBackend, ToolRegistry,
                       debate, estimate_tokens, generator_evaluator_loop, make_builtin_tools, orchestrate)
from llm.agent.context import message_text
from llm.pipeline import get_base_model, run_path
from llm.tasks import TaskExample, verify

args = setup("Lab 28: multi-agent systems")
LATENCY = 0.25                      # seconds per simulated model call (a real API call is 1–10 s)
T_START = time.perf_counter()


def call(name, **arguments):
    return {"text": "", "tool_calls": [{"name": name, "arguments": arguments}]}


class RoutedBackend:
    """A *thread-safe* scripted backend: the reply script is chosen by which route key
    appears in the first user message, and each route keeps its own position.

    ``ScriptedBackend`` replays one list in call order, which is fine for one agent
    but a race when several worker threads share it: whichever thread asks first gets
    the next line, whoever it was meant for. Routing by task fixes that. Each call also
    sleeps ``latency`` seconds and records (route, start, end) so we can draw the overlap.
    """

    def __init__(self, routes: dict, latency: float = 0.0) -> None:
        self.routes = {k: [ScriptedBackend._coerce(x) for x in v] for k, v in routes.items()}
        self.latency = latency
        self.calls: list[dict] = []
        self.timeline: list[tuple[str, float, float]] = []
        self._pos: dict = {}
        self._lock = threading.Lock()

    def complete(self, messages, tools, system) -> AssistantMessage:
        t0 = time.perf_counter() - T_START
        first = messages[0]["content"]
        key = next((k for k in self.routes if k in first), "?")
        with self._lock:
            i = self._pos.get(key, 0)
            self._pos[key] = i + 1
            self.calls.append({"messages": [dict(m) for m in messages], "tools": tools, "system": system, "route": key})
        time.sleep(self.latency)
        script = self.routes.get(key, [])
        reply = script[i] if i < len(script) else AssistantMessage(text="(script exhausted)")
        with self._lock:
            self.timeline.append((key, t0, time.perf_counter() - T_START))
        return reply


def cost_of(*backends) -> tuple[int, int]:
    """(model calls, estimated input tokens) summed over backends."""
    n, total = 0, 0
    for b in backends:
        for c in b.calls:
            n += 1
            total += estimate_tokens(c["system"]) + estimate_tokens(json.dumps(c["tools"]))
            total += sum(estimate_tokens(message_text(m)) for m in c["messages"])
    return n, total


def gantt(timeline, width: int = 40) -> None:
    end = max(t1 for _, _, t1 in timeline)
    for key, t0, t1 in sorted(timeline, key=lambda x: (x[1], x[0])):
        a, b = int(width * t0 / end), max(int(width * t1 / end), int(width * t0 / end) + 1)
        print(f"  {key[:26]:<26} |{' ' * a}{'█' * (b - a)}{' ' * (width - b)}| {t0:5.2f}–{t1:5.2f} s")


tools = make_builtin_tools(run_path("lab28_sandbox"))
SUMS = {"12 + 7 + 9": "28", "(3 + 5 + 4) / 3": "4.0", "2 ** 10": "1024"}
TASK = "For the report, compute three numbers: 12 + 7 + 9, (3 + 5 + 4) / 3, and 2 ** 10."
summary: dict[str, tuple[int, int, float]] = {}      # pattern -> (calls, tokens, seconds)

# ======================================================== 1. single agent
section("1. baseline: one agent does all three sums in sequence")
single = RoutedBackend({"report": [call("calculator", expression=e) for e in SUMS] + ["28, 4.0 and 1024."]},
                       latency=LATENCY)
t0 = time.perf_counter()
t = Agent(single, tools, AgentConfig(permission_policy="allow_all")).run(TASK)
secs = time.perf_counter() - t0
n, toks = cost_of(single)
summary["single agent"] = (n, toks, secs)
print(f"answer: {t.final_text!r}  | {n} model calls, ≈{toks} input tokens, {secs:.2f}s")
print("context tokens per call:", [sum(estimate_tokens(message_text(m)) for m in c["messages"]) for c in single.calls])
check(t.final_text.startswith("28"), "the single agent answers")

# ================================================= 2. orchestrator–workers
section("2. orchestrator–workers: split, fan out in parallel threads, merge")
subtasks = [f"Use the calculator to compute {e}" for e in SUMS]
orch = ScriptedBackend([json.dumps(subtasks), "Report numbers: 28, 4.0 and 1024."])
workers = RoutedBackend({e: [call("calculator", expression=e), f"{e} = {v}"] for e, v in SUMS.items()},
                        latency=LATENCY)
T_START = time.perf_counter()
t0 = time.perf_counter()
out = orchestrate(orch, workers, TASK, tools, n_workers=3)
secs = time.perf_counter() - t0
n, toks = cost_of(orch, workers)
summary["orchestrator–workers"] = (n, toks, secs)
print(f"merged answer: {out!r}  | {n} model calls, ≈{toks} input tokens, {secs:.2f}s")
print("worker timeline (each worker = 2 model calls of 0.25 s):")
gantt(workers.timeline)
busy = sum(t1 - t0_ for _, t0_, t1 in workers.timeline)
span = max(t1 for _, _, t1 in workers.timeline) - min(t0_ for _, t0_, _ in workers.timeline)
print(f"workers were busy for {busy:.2f} s of model latency in total but spanned only {span:.2f} s of wall-clock: "
      f"{busy / span:.1f}× overlap")
print(f"what the orchestrator saw when merging: {orch.calls[1]['messages'][0]['content'][:120]!r}...")
check(out.startswith("Report numbers"), "the orchestrator merged the worker reports")
check(span < 0.7 * busy, "three workers overlap in time (fan-out saves wall-clock)")
check(all(len(c["messages"]) <= 3 for c in workers.calls), "each worker's context holds only its own subtask")

# ================================================= 3. generator / evaluator
section("3. generator / evaluator: a fresh pair of eyes rejects once, then accepts")
GEN_TASK = "Write a Python function median(xs) that returns the median of a list of numbers."
V1 = "def median(xs):\n    return sorted(xs)[len(xs) // 2]"
V2 = "def median(xs):\n    s = sorted(xs)\n    n = len(s)\n    m = n // 2\n    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2"
gen = ScriptedBackend([V1, V2])
ev = ScriptedBackend(["Not accepted: median([1, 2, 3, 4]) returns 3; for even n the two middle values must be averaged (2.5).",
                      "ACCEPT"])
best, rounds = generator_evaluator_loop(gen, ev, GEN_TASK, ToolRegistry(), max_rounds=3)
print(f"accepted after {rounds} rounds:\n" + "\n".join("    " + l for l in best.splitlines()))
print(f"what the generator saw in round 2: ...{gen.calls[1]['messages'][0]['content'][-95:]!r}")
print(f"what the evaluator saw in round 1: {ev.calls[0]['messages'][0]['content'][:60]!r}... (task + candidate only)")
n, toks = cost_of(gen, ev)
summary["generator/evaluator (model judge)"] = (n, toks, 0.0)
check(rounds == 2 and best == V2, "the loop returns the second draft after one rejection")
check("Reviewer feedback" in gen.calls[1]["messages"][0]["content"], "the critique is fed back to the generator")


def program_judge(candidate: str) -> bool:
    """A *program* as the judge: run the candidate and test it (the tests are the ground truth)."""
    ns: dict = {}
    try:
        exec(candidate, ns)                      # our own scripted strings; never exec untrusted model output
        return ns["median"]([1, 2, 3, 4]) == 2.5 and ns["median"]([3, 1, 2]) == 2
    except Exception:
        return False


gen2 = ScriptedBackend([V1, V2])
ev_unused = ScriptedBackend([])
best2, rounds2 = generator_evaluator_loop(gen2, ScriptedBackend(["Looks wrong, try again."]), GEN_TASK,
                                          ToolRegistry(), max_rounds=3, accept_fn=program_judge)
print(f"\nwith accept_fn = a program that runs the tests: accepted after {rounds2} rounds "
      f"(the model evaluator is only consulted when the program says no)")
check(rounds2 == 2 and best2 == V2, "a program judge accepts exactly the candidate that passes")

sycophant = ScriptedBackend(["ACCEPT"] * 3)
best3, rounds3 = generator_evaluator_loop(ScriptedBackend([V1, V2]), sycophant, GEN_TASK, ToolRegistry(), max_rounds=3)
print(f"\nwith a sycophantic evaluator (always ACCEPT): accepted after {rounds3} round(s); "
      f"program_judge(best) = {program_judge(best3)}  <- the wrong draft was accepted")
check(rounds3 == 1 and not program_judge(best3), "a sycophantic evaluator accepts the wrong draft")

# ============================================================= 4. debate
section("4. debate: agreement is not correctness")
Q = "What is 17 + 25? Reply with the number only."
agents = [ScriptedBackend(["41", "41"]), ScriptedBackend(["41", "41"]), ScriptedBackend(["42", "41"])]
r1 = debate([ScriptedBackend([b.script[0].text]) for b in agents], Q, rounds=1)
answers = debate(agents, Q, rounds=2)
print(f"round 1 answers: {r1}   (agent 2 alone is right)")
print(f"round 2 answers: {answers}   (agent 2 saw two peers say 41 and conformed)")
print(f"what agent 2 saw in round 2: {agents[2].calls[1]['messages'][0]['content'][-70:]!r}")
ex = TaskExample("add", "What is 17 + 25?", "17 + 25 = 42", {"answer": 42})
majority = max(set(answers), key=answers.count)
print(f"majority vote: {majority}; verify() says {verify(ex, majority):.0f}. "
      f"A verifier over round-1 answers would have picked {[a for a in r1 if verify(ex, a)]}")
check(majority == "41" and verify(ex, majority) == 0.0, "the unanimous answer is wrong; only the verifier knows")

# ============================================================== 5. cost
section("5. the bill: calls, tokens and seconds per pattern")
print(f"  {'pattern':<36}{'model calls':>12}{'est. input tokens':>19}{'wall-clock':>12}")
for name, (n, toks, secs) in summary.items():
    print(f"  {name:<36}{n:>12}{toks:>19,}{secs:>11.2f}s" if secs else f"  {name:<36}{n:>12}{toks:>19,}{'-':>12}")
s_n, s_tok, s_sec = summary["single agent"]
o_n, o_tok, o_sec = summary["orchestrator–workers"]
print(f"\n  orchestrator–workers vs single agent: {o_n / s_n:.2f}× the calls, {o_tok / s_tok:.2f}× the tokens, "
      f"{s_sec / o_sec:.2f}× faster")
check(o_n > s_n and o_sec < s_sec, "fan-out costs more calls and finishes sooner")

# ============================================================== --full extras
if args.full:
    section("6. --full: six workers, and three TinyLM agents debating")
    exprs = {f"{i} * {i}": str(i * i) for i in range(11, 17)}
    orch6 = ScriptedBackend([json.dumps([f"Use the calculator to compute {e}" for e in exprs]), "merged six squares"])
    w6 = RoutedBackend({e: [call("calculator", expression=e), f"{e} = {v}"] for e, v in exprs.items()}, latency=LATENCY)
    T_START = time.perf_counter()
    t0 = time.perf_counter()
    orchestrate(orch6, w6, "squares of 11..16", tools, n_workers=6)
    secs6 = time.perf_counter() - t0
    busy6 = sum(t1 - t0_ for _, t0_, t1 in w6.timeline)
    print(f"six workers: {busy6:.2f} s of latency in {secs6:.2f} s wall-clock ({busy6 / secs6:.1f}× overlap)")
    gantt(w6.timeline)
    check(secs6 < 0.5 * busy6, "six workers overlap at least 2×")

    model, tok = get_base_model(quick=False, verbose=False)
    model.extend_context(512)
    backends = [TinyLMBackend(model, tok, max_new_tokens=12, temperature=temp) for temp in (0.0, 0.7, 1.0)]
    t0 = time.perf_counter()
    out = debate(backends, Q, rounds=2)
    print(f"three TinyLM base models (T = 0, 0.7, 1.0) after 2 rounds: {[o[:30] for o in out]}  [{time.perf_counter() - t0:.1f}s]")
    print("   (a base model cannot follow the question; more copies of it do not help — capability is not a vote)")
    check(all(verify(ex, o) == 0.0 for o in out), "debating base models are all wrong: debate cannot create capability")

# ----------------------------------------------------------------- figure
fig, axes = plt().subplots(1, 2, figsize=(11, 3.4))
ax = axes[0]
keys = sorted({k for k, _, _ in workers.timeline}, key=lambda k: list(SUMS).index(k))
for row, key in enumerate(keys):
    for k, a, b in workers.timeline:
        if k == key:
            ax.barh(row, b - a, left=a, color="#2563eb", alpha=0.8)
for k, a, b in single.timeline:
    ax.barh(len(keys), b - a, left=a - single.timeline[0][1], color="#f59e0b", alpha=0.8)
ax.set_yticks(range(len(keys) + 1)); ax.set_yticklabels([f"worker {k}" for k in keys] + ["single agent"])
ax.set_xlabel("seconds"); ax.set_title("model-call latency: 3 workers overlap, 1 agent serialises")
ax = axes[1]
names = list(summary)
ax.bar(range(len(names)), [summary[n][1] for n in names], color=["#f59e0b", "#2563eb", "#7c3aed"])
ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
ax.set_ylabel("estimated input tokens"); ax.set_title("the bill")
for i, n_ in enumerate(names):
    ax.text(i, summary[n_][1], f"{summary[n_][0]} calls", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
savefig(fig, "lab28_patterns.png")
done()
