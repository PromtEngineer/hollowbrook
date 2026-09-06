"""Lab 24: the agent loop, with a scripted model, a tiny real model, and an API dialect.

(1) A two-tool task (calculator + write_file) driven by a ScriptedBackend: the transcript,
    the event list, and what the model saw on each call. Then every other way a run can end:
    an unknown tool, max_turns, a backend exception, and two tool calls in one turn.
(2) The permission gate: allow_read_only denies write_file and the (scripted) model adapts;
    the "ask" policy defers to a permission function.
(3) Hooks: a pre_tool hook that blocks a path, a post_tool hook that redacts a secret, and an
    on_event hook that prints a live trace.
(4) TinyLMBackend: the instruction-tuned model of Lab 15 never emits a tool call; Chapter 21's
    tool-trained model does, but only under the exact prompt prefix it was trained with; a short
    tool-SFT learns the syntax and not the semantics. Tool use is trained, not parsed into existence.
    --full trains that model here (~3-5 min on a laptop CPU) and saves runs/lab24_toolsft_nano.pt.
(5) AnthropicBackend.to_api_messages: the same conversation in the Messages API format, and
    how to enable the real backend (no network is used in this lab).

Run:  python3 labs/lab24_agent_loop.py            (quick, ~1-2 min: three tiny models decoding on CPU)
      python3 labs/lab24_agent_loop.py --full     (adds the tool-call SFT, ~5 min)
"""
from _common import setup, check, banner, section, savefig, done, plt

import glob
import json
import os
import shutil
import tempfile
import time
from types import SimpleNamespace

from llm.agent import (Agent, AgentConfig, AnthropicBackend, AssistantMessage, Hooks, ScriptedBackend, Tool,
                       ToolCall, ToolRegistry, make_builtin_tools)
from llm.agent.context import estimate_tokens
from llm.chat import parse_tool_call
from llm.pipeline import run_path

args = setup("Lab 24: the agent loop")
BOX = tempfile.mkdtemp(prefix="lab24_")
tools = make_builtin_tools(BOX)
TASK = "Compute (17 + 25) * 3 and save the result to result.txt."


def call(name, **arguments):
    """One scripted assistant turn that calls one tool."""
    return {"text": "", "tool_calls": [{"name": name, "arguments": arguments}]}


# =============================================================== (1) the loop
section("(1) a two-tool task: think -> act -> observe, twice, then a final answer")
print(f"   {len(tools)} tools: {tools.names()}; their schemas cost ~{estimate_tokens(json.dumps(tools.schemas()))} tokens per call")
script = [
    {"text": "I'll compute the expression first.",
     "tool_calls": [{"id": "call_1", "name": "calculator", "arguments": {"expression": "(17 + 25) * 3"}}]},
    {"text": "126. Now I'll save it.",
     "tool_calls": [{"id": "call_2", "name": "write_file", "arguments": {"path": "result.txt", "content": "126\n"}}]},
    "Done: (17 + 25) * 3 = 126, saved to result.txt.",
]
backend = ScriptedBackend(script)
t = Agent(backend, tools, AgentConfig(permission_policy="allow_all")).run(TASK)
print("\n" + t.pretty() + "\n")
print("   events:")
for e in t.events:
    print("   ", e)
print(f"\n   stop_reason={t.stop_reason!r} turns={t.turns} tool_calls_made={t.tool_calls_made}")
check(t.stop_reason == "done" and t.turns == 3 and t.tool_calls_made == 2, "3 turns, 2 tool calls, stopped because the last reply had no tool calls")
check(open(os.path.join(BOX, "result.txt")).read() == "126\n", "result.txt contains 126")
check([e.kind for e in t.events] == ["assistant", "tool_call", "tool_result"] * 2 + ["assistant", "done"], "event kinds: (assistant, tool_call, tool_result) x2, assistant, done")
check(t.messages[2] == {"role": "tool_result", "tool_call_id": "call_1", "content": "126"}, "the tool result is paired with call_1 and its content is the calculator's text")
print("   what the model saw on each call (roles):")
for i, c in enumerate(backend.calls):
    print(f"      call {i}: {[m['role'] for m in c['messages']]}  ~{estimate_tokens(c['system']) + estimate_tokens(json.dumps(c['tools'])) + sum(estimate_tokens(json.dumps(m)) for m in c['messages'])} tokens")
check(backend.calls[1]["messages"][2]["content"] == "126", "the second model call saw the first tool result in its messages")

section("(1b) the other ways a run ends, and errors as observations")
# an unknown tool name is an observation, not a crash
b = ScriptedBackend([call("calc", expression="2 ** 10"), call("calculator", expression="2 ** 10"), "1024."])
t2 = Agent(b, tools, AgentConfig(permission_policy="allow_all")).run("What is 2 ** 10?")
print("   unknown tool ->", t2.messages[2]["content"][:95])
check(t2.messages[2]["content"].startswith("Error: unknown tool 'calc'") and t2.final_text == "1024.", "unknown tool: an error text, then the (scripted) model corrected the name and finished")
# schema validation, no tool run
print("   bad arguments ->", tools.call("calculator", {"expr": "1"}))
print("   extra argument ->", tools.call("calculator", {"expression": "1", "expr": "1"}))
check("missing required argument 'expression'" in tools.call("calculator", {"expr": "1"})
      and "unknown argument 'expr'" in tools.call("calculator", {"expression": "1", "expr": "1"}),
      "validation errors name the argument and list the expected ones; no tool code ran")
# max_turns: a model that never stops calling tools
t3 = Agent(ScriptedBackend([call("list_dir")] * 10), tools, AgentConfig(permission_policy="allow_all", max_turns=3)).run("loop")
check(t3.stop_reason == "max_turns" and t3.turns == 3, f"a model that always calls a tool is cut off: stop_reason={t3.stop_reason!r} after {t3.turns} turns")


class FlakyBackend:                                  # any object with .complete() is a backend
    def complete(self, messages, tools, system):
        raise TimeoutError("upstream API timed out")


t4 = Agent(FlakyBackend(), tools, AgentConfig()).run("anything")
check(t4.stop_reason == "error" and t4.events[-1].kind == "error", f"a backend exception ends the run with stop_reason={t4.stop_reason!r}: {t4.events[-1].data['error']}")
# two tool calls in one assistant turn -> two tool_result messages, paired by id
b = ScriptedBackend([{"text": "", "tool_calls": [{"id": "a", "name": "calculator", "arguments": {"expression": "2+2"}},
                                                  {"id": "b", "name": "calculator", "arguments": {"expression": "3+3"}}]}, "4 and 6."])
t5 = Agent(b, tools, AgentConfig(permission_policy="allow_all")).run("2+2 and 3+3?")
check([m["tool_call_id"] for m in t5.messages if m["role"] == "tool_result"] == ["a", "b"] and t5.turns == 2,
      "two calls in one turn -> two tool_result messages (ids a, b) in one turn")
# parsing tool calls out of raw model text (the TinyLM dialect)
raw_ok = '<|tool_call|>{"name": "calculator", "arguments": {"expression": "6*7"}}<|end|>'
raw_bad = '<|tool_call|>{"name": "calculator", "arguments": {"expression": "6*7"}<|end|>'   # missing brace
print(f"   parse_tool_call(valid)     -> {parse_tool_call(raw_ok)}")
print(f"   parse_tool_call(malformed) -> {parse_tool_call(raw_bad)}")
check(parse_tool_call(raw_ok)["name"] == "calculator" and parse_tool_call(raw_bad) is None, "malformed JSON parses to None: the reply is then treated as a final answer, not a crash")


# =============================================================== (2) permissions
section("(2) the permission gate: allow_read_only denies write_file; the model adapts")
os.remove(os.path.join(BOX, "result.txt"))
adaptive = [
    call("calculator", expression="(17 + 25) * 3"),
    call("write_file", path="result.txt", content="126\n"),
    "I computed 126, but I am not allowed to write files under this policy. Please save it yourself or grant write access.",
]
b = ScriptedBackend(adaptive)
t = Agent(b, tools, AgentConfig(permission_policy="allow_read_only")).run(TASK)
print(t.pretty())
check(t.messages[4]["content"].startswith("Permission denied: 'write_file'"), "the denial arrived as a tool_result the model can read")
check(not os.path.exists(os.path.join(BOX, "result.txt")), "result.txt was NOT written")
check("permission_denied" in [e.kind for e in t.events] and t.stop_reason == "done", "a permission_denied event was emitted and the run still ended cleanly")

print("\n   policy 'ask': the harness asks a permission function (a real UI would prompt the user)")


def ask_user(c: ToolCall, tool: Tool) -> bool:
    decision = tool.read_only or c.arguments.get("path", "").endswith(".txt")
    print(f"      [permission prompt] {c.name}({json.dumps(c.arguments)[:60]}) read_only={tool.read_only} -> {'allow' if decision else 'deny'}")
    return decision


t = Agent(ScriptedBackend(adaptive), tools, AgentConfig(permission_policy="ask")).run(TASK, permission_fn=ask_user)
check(os.path.exists(os.path.join(BOX, "result.txt")), "under 'ask', the permission function allowed the .txt write and the file exists")
t = Agent(ScriptedBackend(adaptive), tools, AgentConfig(permission_policy="ask")).run(TASK)   # no permission_fn at all
check(t.messages[2]["content"].startswith("Permission denied: 'calculator'") and t.messages[4]["content"].startswith("Permission denied"),
      "under 'ask' with no permission function, every tool is denied, the read-only calculator included (safe default)")


# =============================================================== (3) hooks
section("(3) hooks: pre_tool blocks a path, post_tool redacts a secret, on_event traces")
os.makedirs(os.path.join(BOX, "secrets"), exist_ok=True)
with open(os.path.join(BOX, "secrets", ".env"), "w") as f:
    f.write("API_KEY=sk-live-8f2a9c\nDEBUG=false\n")
hooks = Hooks()
hooks.pre_tool.append(lambda c: "secrets/ is read-only for the agent" if c.name == "write_file" and c.arguments.get("path", "").startswith("secrets/") else None)
import re
hooks.post_tool.append(lambda c, r: re.sub(r"(API_KEY=)\S+", r"\1[REDACTED]", r) if c.name == "read_file" else None)
hooks.on_event.append(lambda e: print(f"      trace | turn {e.turn} | {e.kind:<17} | {json.dumps(e.data)[:70]}"))
b = ScriptedBackend([call("read_file", path="secrets/.env"),
                     call("write_file", path="secrets/.env", content="API_KEY=stolen\n"),
                     "I read the config (the key is redacted) and was not allowed to change it."])
t = Agent(b, tools, AgentConfig(permission_policy="allow_all"), hooks=hooks).run("Read secrets/.env and rotate the key.")
print("   read result the model saw:", t.messages[2]["content"].replace("\n", " | "))
print("   write result the model saw:", t.messages[4]["content"])
check("[REDACTED]" in t.messages[2]["content"] and "sk-live" not in t.messages[2]["content"], "post_tool hook redacted the API key before the model saw it")
check(t.messages[4]["content"] == "Blocked by hook: secrets/ is read-only for the agent", "pre_tool hook blocked the write even though the policy was allow_all")
check(open(os.path.join(BOX, "secrets", ".env")).read().startswith("API_KEY=sk-live"), "the file on disk is untouched")
check(any(e.kind == "hook" for e in t.events), "a 'hook' event recorded the block")


# =============================================================== (4) TinyLM backend
section("(4) a real (tiny) model on the other side: TinyLMBackend")
import torch                                              # noqa: E402  (deferred: ~8 s import)
from llm.pipeline import get_base_model                   # noqa: E402
from llm.agent import TinyLMBackend                       # noqa: E402
from llm.chat import render                               # noqa: E402

base, tok = get_base_model(quick=True, verbose=False)     # the nano base model (Part II)
from llm.model import TinyLM                              # noqa: E402
calc_only = tools.subset(["calculator"])
SYS = "Use the calculator tool, then answer."
question = "What is 17 + 25?"

# How long is the prompt with the tool schemas pasted in?  Longer than the window.
from llm.agent.backends import ContextTooLongError        # noqa: E402
from llm.chat import encode_chat                          # noqa: E402
msgs = [{"role": "user", "content": question}]
n_full = len(encode_chat(tok, TinyLMBackend.to_chat_messages(msgs, calc_only.schemas(), SYS, compact_tools=False), add_generation_prompt=True))
n_compact = len(encode_chat(tok, TinyLMBackend.to_chat_messages(msgs, calc_only.schemas(), SYS), add_generation_prompt=True))
print(f"   nano model: {base.num_params():,} params, window {base.cfg.max_seq_len} tokens")
print(f"   prompt with ONE full JSON schema (compact_tools=False): {n_full} tokens; with the compact listing (default): {n_compact} tokens")
try:
    TinyLMBackend(base, tok, max_new_tokens=8, compact_tools=False).complete(msgs, calc_only.schemas(), SYS)
    err = None
except ContextTooLongError as e:
    err = e
print(f"   -> {type(err).__name__}: {err}")
check(n_full > base.cfg.max_seq_len and err is not None, "an over-long prompt raises ContextTooLongError instead of silently generating nothing")
t_err = Agent(TinyLMBackend(base, tok, max_new_tokens=8, compact_tools=False), calc_only, AgentConfig(permission_policy="allow_all"), system_prompt=SYS).run(question)
check(t_err.stop_reason == "error" and "ContextTooLongError" in t_err.events[-1].data["error"], f"inside Agent.run the same error becomes stop_reason={t_err.stop_reason!r}")


class NoListingBackend(TinyLMBackend):
    """For the *nano* models below: name the tool in the system prompt and send NO tool listing.
    Even the compact listing costs ~50 of nano's 128 tokens, and the checkpoint --full trains is
    fine-tuned without it. Chapter 21's checkpoints are the opposite case: they were trained WITH
    the listing (the served prompt is stored in the checkpoint), so they get the plain backend."""

    @staticmethod
    def to_chat_messages(messages, tools, system, compact_tools=True):
        return TinyLMBackend.to_chat_messages(messages, [], system)


def calc_registry(tool_name: str) -> ToolRegistry:
    """The calculator under the name a given model was trained with ('calculator' or Ch. 21's 'calc')."""
    return ToolRegistry([Tool(tool_name, "Evaluate an arithmetic expression.",
                              {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
                              tools.get("calculator").fn)])


def try_model(model, label: str, system: str, tool_name: str, questions, backend_cls=NoListingBackend) -> tuple[int, int]:
    """Probe a model on fresh questions: how many replies parse as a tool call, and how many
    of those carry the RIGHT expression. Returns (well_formed, correct_args)."""
    be = backend_cls(model, tok, max_new_tokens=64)           # a full tool call is ~55 Storyland tokens
    schemas = calc_registry(tool_name).schemas()              # ignored by NoListingBackend, listed by TinyLMBackend
    well_formed = correct = 0
    for a, b_ in questions:
        r = be.complete([{"role": "user", "content": f"What is {a} + {b_}?"}], schemas, system)
        well_formed += bool(r.tool_calls)
        expr = r.tool_calls[0].arguments.get("expression", "") if r.tool_calls and r.tool_calls[0].name == tool_name else ""
        correct += expr.replace(" ", "") == f"{a}+{b_}"
    n = len(questions)
    print(f"   {label}: on {n} questions: {well_formed}/{n} well-formed tool calls, {correct}/{n} with the right expression")
    return well_formed, correct


def run_agent(model, system: str, tool_name: str, q: str, backend_cls=NoListingBackend):
    return Agent(backend_cls(model, tok, max_new_tokens=64), calc_registry(tool_name),
                 AgentConfig(permission_policy="allow_all", max_turns=3), system_prompt=system).run(q)


# (4a) a model that was never trained on tool calls: the instruction-tuned model of Lab 15 if it
#      exists (it follows instructions but has never seen <|tool_call|>), else the base model.
QS = [(3, 4), (60, 31), (88, 9)]
if os.path.exists(run_path("sft_nano.pt")):
    untrained, label_u = TinyLM.load(run_path("sft_nano.pt")), "instruction-SFT model (Lab 15, no tool data)"
else:
    untrained, label_u = base, "base model"
t0 = time.perf_counter()
ok_u, right_u = try_model(untrained, label_u, SYS, "calculator", QS)
t_u = run_agent(untrained, SYS, "calculator", question)
print(f"   {label_u} in the loop: raw {t_u.messages[1]['content'][:60]!r} -> stop_reason={t_u.stop_reason!r}, "
      f"tool calls {t_u.tool_calls_made}  ({time.perf_counter() - t0:.1f}s)")
check(t_u.stop_reason in ("done", "max_turns"), f"the loop terminated cleanly whatever the {label_u.split(' (')[0]} produced")
print(f"   -> a model with no tool training {'DID' if ok_u else 'did NOT'} emit a valid <|tool_call|>: tool use is a trained behaviour (Chapters 15, 21)")

# (4b) Chapter 21's tool-trained model (SFT on tool traces + GRPO) under the EXACT prefix it was
#      trained with: the `small` model (256-token window), tool 'calc', the short system prompt,
#      the compact tool listing that TinyLMBackend prepends by default, and questions in its 0-20
#      range. Lab 21 stores the served system prompt in the checkpoint's `extra` field so a harness
#      can check that it is serving the prompt the policy was trained on.
SYS21, Q21, QS21 = "Use the calc tool.", "What is 17 + 5?", [(3, 4), (8, 9), (6, 19)]
ch21 = [n for n in ("lab21_tool_grpo.pt", "lab21_tool_sft.pt") if os.path.exists(run_path(n))]
if ch21:
    m21 = TinyLM.load(run_path(ch21[0]))
    served = TinyLMBackend.to_chat_messages([], calc_registry("calc").schemas(), SYS21)[0]["content"]
    stored = torch.load(run_path(ch21[0]), map_location="cpu", weights_only=False).get("extra", {}).get("system_prompt")
    print(f"   served system prompt ({len(tok.encode(served))} tokens) == the prompt stored in the checkpoint: {served == stored}")
    ok21, right21 = try_model(m21, f"Chapter 21 model (runs/{ch21[0]}), trained prefix", SYS21, "calc", QS21, TinyLMBackend)
    ok21n, _ = try_model(m21, "   the same model with the tool listing stripped", SYS21, "calc", QS21, NoListingBackend)
    t21 = run_agent(m21, SYS21, "calc", Q21, TinyLMBackend)
    print(t21.pretty())
    check(served == stored, "the harness serves the system prompt the checkpoint was trained with (Chapter 21's rule)")
    check(t21.tool_calls_made >= 1 and t21.messages[2]["content"] == "22",
          f"Chapter 21's model called calc({(t21.messages[1].get('tool_calls') or [{}])[0].get('arguments')}) and the harness returned 22")
    check("22" in t21.final_text, f"its final answer uses the result: {t21.final_text!r}")
    check(right21 >= 2 and ok21 > ok_u, f"well-formed AND correct on fresh questions: {right21}/3 (vs {ok_u}/3 well-formed for the untrained model)")
    check(ok21n < ok21, f"with the listing stripped the SAME weights drop to {ok21n}/3 well-formed: a policy is conditioned on its trained prefix")
else:
    print("   (no runs/lab21_tool_*.pt: run labs/lab21_agentic_rl.py to see a tool-trained model succeed here)")

# (4c) our own short SFT: 150 steps on 600 traces. --full trains it (and saves it); later runs load it.
own = run_path("lab24_toolsft_nano.pt")
sft_model = None
if os.path.exists(own):
    sft_model = TinyLM.load(own)
    print(f"   loaded runs/lab24_toolsft_nano.pt from an earlier --full run")
elif args.full:
    from llm.sft import sft_train, SFTConfig             # noqa: E402
    import random

    def tool_traces(n, seed):
        """Raw conversations (lists of role dicts): sft_train accepts these directly."""
        rng, out = random.Random(seed), []
        for _ in range(n):
            a, b_ = rng.randint(0, 99), rng.randint(0, 99)
            out.append([{"role": "system", "content": SYS},
                        {"role": "user", "content": f"What is {a} + {b_}?"},
                        {"role": "tool_call", "content": json.dumps({"name": "calculator", "arguments": {"expression": f"{a} + {b_}"}})},
                        {"role": "tool_result", "content": str(a + b_)},
                        {"role": "assistant", "content": f"{a} + {b_} = {a + b_}"}])
        return out

    print("   --full: fine-tuning the nano base on 600 tool-call traces (150 steps)...")
    sft_model, _ = get_base_model(quick=True, verbose=False)         # a fresh copy of the base
    t0 = time.perf_counter()
    hist = sft_train(sft_model, tok, tool_traces(600, 1), SFTConfig(steps=150, batch_size=8, lr=1e-3, warmup_steps=10, max_len=128, log_every=50), verbose=True)
    print(f"   SFT done in {time.perf_counter() - t0:.0f}s: loss {hist.train_loss[0]:.2f} -> {hist.train_loss[-1]:.2f}")
    sft_model.save(own, extra={"stage": "toolsft"})
    print(f"   saved runs/lab24_toolsft_nano.pt")
else:
    print("   (run with --full to train a 150-step tool-SFT here: ~3.5 min)")

if sft_model is not None:
    ok_sft, right_sft = try_model(sft_model, "150-step tool-SFT model", SYS, "calculator", QS)
    t_sft = run_agent(sft_model, SYS, "calculator", question)
    print(t_sft.pretty())
    expr = (t_sft.messages[1].get("tool_calls") or [{}])[0].get("arguments", {}).get("expression")
    check(t_sft.tool_calls_made >= 1 and t_sft.messages[2]["role"] == "tool_result",
          f"the 150-step model emitted a well-formed call; the harness ran calculator({expr!r}) -> {t_sft.messages[2]['content']!r}")
    check(ok_sft > ok_u, f"well-formed tool calls on 3 fresh questions: {ok_sft}/3 vs {ok_u}/3 untrained")
    check(t_sft.stop_reason == "done" and t_sft.turns == 2, f"the loop fed the result back and the model answered in turn 2: {t_sft.final_text!r}")
    right = expr is not None and expr.replace(" ", "") == "17+25"
    print(f"   expression correct for {question!r}: {right} (expected '17 + 25'); on the other 3: {right_sft}/3")
    if not right:
        print("   -> SYNTAX learned, SEMANTICS not: 150 steps teach this 0.3M-param model the tool-call format but not")
        print("      to copy the numbers from the question. The harness cannot tell: the call is valid, the tool runs,")
        print("      and the model trusts the (wrong) result. Chapter 21's recipe (500 SFT steps on a 0-20 range, then")
        print("      GRPO with a verifiable reward) is what grounds the arguments; see (4b).")


# =============================================================== (5) the API dialect
section("(5) AnthropicBackend.to_api_messages: the same conversation, Messages-API shaped")
convo = [{"role": "user", "content": TASK},
         {"role": "assistant", "content": "I'll compute the expression first.",
          "tool_calls": [{"id": "call_1", "name": "calculator", "arguments": {"expression": "(17 + 25) * 3"}}]},
         {"role": "tool_result", "tool_call_id": "call_1", "content": "126"},
         {"role": "assistant", "content": "", "tool_calls": [{"id": "call_2", "name": "write_file", "arguments": {"path": "result.txt", "content": "126\n"}},
                                                             {"id": "call_3", "name": "list_dir", "arguments": {}}]},
         {"role": "tool_result", "tool_call_id": "call_2", "content": "Wrote 4 chars to result.txt"},
         {"role": "tool_result", "tool_call_id": "call_3", "content": "result.txt"}]
api = AnthropicBackend.to_api_messages(convo)
print(json.dumps(api, indent=1)[:1500])
check([m["role"] for m in api] == ["user", "assistant", "user", "assistant", "user"], "roles alternate user/assistant: tool results travel inside USER messages")
check(api[1]["content"][1]["type"] == "tool_use" and api[1]["content"][1]["input"] == {"expression": "(17 + 25) * 3"}, "a tool call is a tool_use content block with `input`")
check(len(api[4]["content"]) == 2 and all(b["type"] == "tool_result" for b in api[4]["content"]), "two consecutive tool results share one user message (two tool_result blocks)")

# the reverse direction, on a fake response object (what the SDK would return)
fake = SimpleNamespace(content=[SimpleNamespace(type="text", text="Let me check."),
                                SimpleNamespace(type="tool_use", id="toolu_01", name="calculator", input={"expression": "1+1"})],
                       usage=SimpleNamespace(input_tokens=812, output_tokens=37), stop_reason="tool_use")
msg = AnthropicBackend.from_api_response(fake)
check(msg.tool_calls[0].id == "toolu_01" and msg.usage == {"input_tokens": 812, "output_tokens": 37}, "from_api_response keeps the provider's tool id and the token usage")
try:
    AnthropicBackend()
except (ImportError, RuntimeError) as e:
    print(f"   AnthropicBackend() without the package/key -> {type(e).__name__}: {e}")
    check(True, "the real backend refuses to start without `pip install anthropic` and ANTHROPIC_API_KEY")
print("   to enable it:  pip install anthropic && export ANTHROPIC_API_KEY=...  then")
print("                  Agent(AnthropicBackend(model='claude-sonnet-5'), tools, AgentConfig(permission_policy='allow_read_only')).run(TASK)")

# ------------------------------------------------------------------ figure: the event timeline of run (1)
fig, ax = plt().subplots(figsize=(9, 2.6))
colors = {"assistant": "#2563eb", "tool_call": "#7c3aed", "tool_result": "#16a34a", "done": "#16a34a",
          "permission_denied": "#dc2626", "hook": "#f59e0b", "error": "#dc2626", "max_turns": "#dc2626", "compaction": "#64748b"}
evs = Agent(ScriptedBackend(script), make_builtin_tools(BOX), AgentConfig(permission_policy="allow_all")).run(TASK).events
for i, e in enumerate(evs):
    ax.scatter(i, e.turn, s=160, color=colors.get(e.kind, "#64748b"), zorder=3)
    ax.annotate(e.kind, (i, e.turn), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8, rotation=30)
ax.set_yticks([1, 2, 3]); ax.set_ylabel("turn"); ax.set_xlabel("event index"); ax.set_ylim(0.5, 4.2)
ax.set_title("Run (1): Transcript.events — think (blue), act (purple), observe (green)")
savefig(fig, "lab24_events.png")

shutil.rmtree(BOX, ignore_errors=True)
done()
