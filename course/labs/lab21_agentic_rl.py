"""Lab 21: agentic RL — TinyLM learns to call a calculator tool.

Environment: ``llm.rl.CalculatorEnv`` (one addition question, a ``calc`` tool, reward 1.0 for the
right answer, +0.2 if the tool was called with the right expression *and* the answer is right).

(a) a multi-turn rollout on the untrained (base) model: it has no idea what a tool is.
(b) synthetic tool-use conversations (user -> tool_call -> tool_result -> answer), the loss mask
    that keeps tool results out of the loss, and an SFT warm-start written in ~15 lines with
    ``chat.build_sft_example`` + ``chat.collate``. Cached as runs/lab21_tool_sft.pt.
(c) multi-turn GRPO for N steps; tool-use rate, correctness and reward before/after, and the
    trained tokens of one trajectory decoded from its mask.

Run:  python3 labs/lab21_agentic_rl.py            (quick)
      python3 labs/lab21_agentic_rl.py --full
"""
from _common import setup, check, banner, section, savefig, done, plt

import json
import os
import random
import time

import torch

from llm import chat, rl, tasks
from llm.model import TinyLM
from llm.pipeline import TOKENIZER_PATH, get_tokenizer, run_path
from llm.rl import CalculatorEnv, GRPOConfig, make_optimizer, multi_turn_grpo_step, multi_turn_rollout

args = setup("Lab 21: agentic RL — learning to use a calculator tool")

SFT_STEPS = 700                      # one-time warm-start (cached); ~4 min on a laptop CPU
GRPO_STEPS = 4 if args.quick else 30
N_TASKS, GROUP, MAX_NEW, MAX_TURNS = 2, 4, 70, 3
N_EVAL = 16 if args.quick else 40
TOOL_FRAC = 0.6                      # share of SFT conversations that use the tool
SYSTEM = "Use the calc tool."        # short: a nano TinyLM has 128 positions and a tool call costs ~60


class CalcEnv(CalculatorEnv):
    """CalculatorEnv with a system prompt short enough for the nano context."""
    system_prompt = SYSTEM


tok = get_tokenizer()
pad_id, end_id, _ = rl._special(tok)
roll_cfg = GRPOConfig(group_size=GROUP, prompts_per_step=N_TASKS, max_new_tokens=MAX_NEW, temperature=1.0)
eval_cfg = GRPOConfig(group_size=1, max_new_tokens=MAX_NEW, temperature=0.0)   # greedy for evaluation


def show(traj: rl.Trajectory) -> None:
    for t in traj.turns:
        print(f"      {t.role:12s} {t.text!r}")
    print(f"      reward {traj.reward:.1f} | tool calls {traj.n_tool_calls} | {len(traj.ids)} tokens | done={traj.done}")


def evaluate(model: TinyLM, n: int, seed0: int = 5000) -> dict:
    """Greedy episodes on n fixed tasks: tool-use rate, accuracy, mean reward."""
    trajs = [multi_turn_rollout(model, tok, CalcEnv.from_seed(seed0 + i), eval_cfg, MAX_TURNS) for i in range(n)]
    return {"tool_rate": sum(t.n_tool_calls > 0 for t in trajs) / n,
            "accuracy": sum(t.reward >= 1.0 for t in trajs) / n,
            "tool_and_correct": sum(t.reward >= 1.2 for t in trajs) / n,
            "reward": sum(t.reward for t in trajs) / n,
            "turns": sum(sum(tr.role == "assistant" for tr in t.turns) for t in trajs) / n}


# ------------------------------------------------------------------ (a) untrained
section("(a) a rollout on the untrained base model")
base = TinyLM.load(run_path("base_nano.pt"))
torch.manual_seed(args.seed)
traj = multi_turn_rollout(base, tok, CalcEnv.from_seed(1), roll_cfg, MAX_TURNS)
show(traj)
check(traj.n_tool_calls == 0, "the base model never produces a parseable tool call")

# ------------------------------------------------------------------ (b) SFT on synthetic tool use
section("(b) synthetic tool-use conversations and the loss mask")


def make_conversation(a: int, b: int, use_tool: bool) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"What is {a} + {b}?"}]
    if use_tool:
        call = {"name": "calc", "arguments": {"expression": f"{a} + {b}"}}
        msgs.append({"role": "tool_call", "content": json.dumps(call)})
        msgs.append({"role": "tool_result", "content": str(a + b)})
    msgs.append({"role": "assistant", "content": f"{a} + {b} = {a + b}"})
    return msgs


rng = random.Random(args.seed)
convs = [make_conversation(rng.randint(0, 20), rng.randint(0, 20), rng.random() < TOOL_FRAC) for _ in range(500)]
n_tool = sum(len(c) == 5 for c in convs)
print(f"   {len(convs)} conversations: {n_tool} with a tool call, {len(convs) - n_tool} direct answers")
ids, mask = chat.build_sft_example(tok, convs[0])
print(f"   example 0: {len(ids)} tokens, {sum(mask)} trainable")
print("   trained tokens in [brackets]:")
print("   " + "".join(f"[{tok.token_str(i)}]" if m else tok.token_str(i) for i, m in zip(ids, mask)))
trained_text = tok.decode([i for i, m in zip(ids, mask) if m])
given_text = tok.decode([i for i, m in zip(ids, mask) if not m])
check("<|tool_result|>" in given_text and "<|tool_call|>" in trained_text,
      "the tool_call marker is trained, the tool_result turn is not")
check(str(convs[0][3]["content"]) not in trained_text.split("<|end|>")[0] if len(convs[0]) == 5 else True,
      "the tool's output is not among the tokens of the tool-call turn the model must produce")

sft_path = run_path("lab21_tool_sft.pt")
if os.path.exists(sft_path):
    policy = TinyLM.load(sft_path)
    print(f"   loaded cached warm-start {sft_path}")
else:
    print(f"   [one-time] SFT warm-start: {SFT_STEPS} steps, batch 16, lr 1e-3 (masked CE on assistant tokens)")
    policy = TinyLM.load(run_path("base_nano.pt"))
    data = [chat.build_sft_example(tok, c, max_len=policy.cfg.max_seq_len) for c in convs]
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(args.seed)
    policy.train()
    t0 = time.perf_counter()
    for step in range(SFT_STEPS):
        pick = torch.randint(0, len(data), (16,), generator=g).tolist()
        x, y, m = chat.collate([data[i] for i in pick], pad_id)      # (16, T-1) each; m aligned to y
        _, loss = policy(x, y, loss_mask=m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == SFT_STEPS - 1:
            print(f"      sft step {step:4d} | masked loss {loss.item():.3f} | {time.perf_counter() - t0:.0f}s")
    policy.eval()
    policy.save(sft_path, TOKENIZER_PATH, extra={"stage": "sft-tool-use", "steps": SFT_STEPS})
    print(f"   saved {sft_path}")

print("   a rollout from the SFT model (temperature 1):")
torch.manual_seed(args.seed)
traj_sft = multi_turn_rollout(policy, tok, CalcEnv.from_seed(1), roll_cfg, MAX_TURNS)
show(traj_sft)
t0 = time.perf_counter()
before = evaluate(policy, N_EVAL)
print(f"   before GRPO ({N_EVAL} greedy episodes, {time.perf_counter() - t0:.0f}s): " +
      " | ".join(f"{k} {v:.2f}" for k, v in before.items()))
check(before["tool_rate"] > 0.3, "after SFT the policy calls the tool in a good share of episodes")

# ------------------------------------------------------------------ (c) multi-turn GRPO
section(f"(c) multi-turn GRPO: {GRPO_STEPS} steps x {N_TASKS} tasks x G={GROUP} trajectories")
cfg = GRPOConfig(group_size=GROUP, prompts_per_step=N_TASKS, max_new_tokens=MAX_NEW, temperature=1.0,
                 lr=1e-4, clip_eps_low=0.2, clip_eps_high=0.28, dynamic_sampling=True, seed=args.seed)
opt = make_optimizer(policy, cfg)
torch.manual_seed(args.seed)
history = []
t0 = time.perf_counter()
last_groups = None
for step in range(GRPO_STEPS):
    st = multi_turn_grpo_step(policy, tok, CalcEnv.from_seed, cfg, opt, n_tasks=N_TASKS, max_turns=MAX_TURNS,
                              task_offset=100 + step * N_TASKS)
    last_groups = st.pop("groups", last_groups)
    history.append(st)
    if step % max(1, GRPO_STEPS // 6) == 0 or step == GRPO_STEPS - 1:
        print(f"   step {step:3d} | reward {st['reward']:.2f} | acc {st['accuracy']:.2f} | tool rate {st['tool_call_rate']:.2f} | "
              f"turns {st['turns']:.1f} | skipped {st['skipped_frac']:.2f} | loss {st.get('loss', 0):+.3f} | {time.perf_counter() - t0:.0f}s")
t_grpo = time.perf_counter() - t0
after = evaluate(policy, N_EVAL)
print(f"\n   GRPO wall-clock {t_grpo:.0f}s ({t_grpo / GRPO_STEPS:.1f}s per step)")
print(f"   {'metric':18s} {'before':>8s} {'after':>8s}")
for k in before:
    print(f"   {k:18s} {before[k]:8.2f} {after[k]:8.2f}")
if args.full:
    check(after["reward"] >= before["reward"], "mean reward did not fall after GRPO")
    check(after["tool_rate"] >= before["tool_rate"], "the tool-use rate did not fall: tool trajectories earn the bonus")

# the loss mask on a trained trajectory
section("the loss mask on one trajectory from the last GRPO step")
traj = next((t for g in last_groups for t in g if t.n_tool_calls > 0), last_groups[0][0])
show(traj)
tr_ids = traj.ids[1:].tolist()                                      # mask[t] scores ids[t + 1]
trained = tok.decode([i for i, m in zip(tr_ids, traj.mask.tolist()) if m > 0])
untrained = tok.decode([i for i, m in zip(tr_ids, traj.mask.tolist()) if m == 0])
print(f"   trained  ({int(traj.mask.sum())} tokens): {trained!r}")
print(f"   given    ({int((traj.mask == 0).sum())} tokens): {untrained!r}")
check("<|tool_result|>" not in trained and "<|user|>" not in trained, "no environment or prompt token is in the loss")
check(all(tr.text in trained for tr in traj.turns if tr.role == "assistant"), "every assistant turn is in the loss")

policy.save(run_path("lab21_tool_grpo.pt"), TOKENIZER_PATH, extra={"stage": "agentic-grpo", "steps": GRPO_STEPS})
print(f"   saved runs/lab21_tool_grpo.pt")

# ------------------------------------------------------------------ figure
fig, axes = plt().subplots(1, 2, figsize=(11, 4))
ax = axes[0]
for key, color in (("reward", "#2563eb"), ("accuracy", "#16a34a"), ("tool_call_rate", "#f59e0b")):
    ax.plot([h[key] for h in history], label=key, color=color, lw=1.5)
ax.set_xlabel("GRPO step"); ax.set_ylim(0, 1.3); ax.legend(); ax.set_title("rollout statistics during training")
ax = axes[1]
keys = ["tool_rate", "accuracy", "tool_and_correct", "reward"]
xs = range(len(keys))
ax.bar([x - 0.2 for x in xs], [before[k] for k in keys], width=0.4, color="#94a3b8", label="before GRPO")
ax.bar([x + 0.2 for x in xs], [after[k] for k in keys], width=0.4, color="#2563eb", label="after GRPO")
ax.set_xticks(list(xs)); ax.set_xticklabels(keys, fontsize=8); ax.legend(); ax.set_title(f"greedy evaluation on {N_EVAL} tasks")
fig.tight_layout()
savefig(fig, "lab21_agentic_rl.png")
done()
