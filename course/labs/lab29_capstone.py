"""Lab 29: the capstone — the whole TinyLM pipeline, end to end, in one script.

    python3 labs/lab29_capstone.py            # --quick: nano model, ~5 min on a laptop CPU
    python3 labs/lab29_capstone.py --full     # small model, ~25 min
    python3 labs/lab29_capstone.py --fresh    # ignore capstone_* checkpoints and retrain them

Stages (each one is a chapter you have already done):

    corpus -> curation report (8) -> tokenizer (2) -> pretrain / load base (10)
    -> mid-training anneal (13) -> SFT (15) -> preference pairs + DPO (17)
    -> GRPO on addition (19) -> on-policy distillation into a student (20)
    -> evals of every checkpoint (23) -> a TinyLM agent using a calculator tool (21, 24)

Checkpoints that already exist in runs/ (from the earlier labs, or from a previous run of
this script) are reused; only what is missing is trained. The stage table and the eval
table are written to runs/capstone_report.md.
"""
from _common import setup, check, banner, section, savefig, done, plt

import copy
import json
import math
import os
import random
import time

import torch

from llm import chat, distill, tasks
from llm.agent import Agent, AgentConfig, ScriptedBackend, TinyLMBackend, Tool, ToolRegistry
from llm.agent.tools import safe_eval
from llm.config import preset
from llm.data import (CurationReport, QualityClassifier, add_noise, corpus_text, curate, make_corpus,
                      mix_sources, tokenize_and_pack)
from llm.dpo import DPOConfig, dpo_eval, dpo_train, make_reference
from llm.evals import contamination_check, eval_tasks, perplexity
from llm.model import TinyLM
from llm.optim import build_optimizer, lr_at, set_lr
from llm.pipeline import (BASE_FULL, BASE_QUICK, TOKENIZER_PATH, device_summary, get_base_model, get_corpus,
                          get_tokenizer, get_tokens, run_path)
from llm.reward import make_preference_pairs, make_preference_pairs_from_model
from llm.rl import GRPOConfig, grpo_train
from llm.sft import SFTConfig, make_sft_examples, sft_train
from llm.train import estimate_loss, get_batch

args = setup("Lab 29: capstone — TinyLM end to end", extra=lambda p: p.add_argument(
    "--fresh", action="store_true", help="retrain every capstone_* checkpoint"))
TAG = "nano" if args.quick else "small"
T_LAB = time.perf_counter()

# On a heavily oversubscribed machine several PyTorch threads spin-wait on each other and a
# 300k-parameter model runs *slower* with 2 threads than with 1 (tests/test_rl.py makes the
# same choice); fall back to one thread only when the load is well past the core count.
load = os.getloadavg()[0]
if load > 3 * os.cpu_count():
    torch.set_num_threads(1)
print(f"{device_summary()} | load average {load:.1f} on {os.cpu_count()} cores -> using {torch.get_num_threads()} thread(s)")

# ------------------------------------------------------------------ helpers
stages: list[tuple[str, str, float]] = []           # (stage, note, seconds)


class Stage:
    """Time one stage and record a one-line note for the report."""

    def __init__(self, name: str) -> None:
        self.name, self.note = name, ""
        section(name)

    def __enter__(self) -> "Stage":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        secs = time.perf_counter() - self.t0
        stages.append((self.name, self.note, secs))
        print(f"   [{self.name}: {secs:.1f}s]")


def _parent_of(path: str) -> str | None:
    """Which checkpoint a capstone_* file was trained from (recorded in its ``extra`` dict)."""
    try:
        return torch.load(path, map_location="cpu").get("extra", {}).get("parent")
    except Exception:
        return None


def existing(*names: str, d_model: int, parent: str | None = None) -> str | None:
    """The first checkpoint in runs/ that exists and has the right width (nano vs small).

    Checkpoints from the earlier labs (sft_nano.pt, grpo_small.pt, ...) are always welcome.
    This script's own capstone_* files are reused only if they were trained from the
    checkpoint this run is actually using upstream (``parent``), so the chain stays a
    lineage; --fresh ignores them altogether."""
    for n in names:
        p = run_path(n)
        if not os.path.exists(p):
            continue
        if n.startswith("capstone_"):
            if args.fresh or (parent is not None and _parent_of(p) != os.path.basename(parent)):
                continue
        try:
            if TinyLM.load(p).cfg.d_model == d_model:
                return p
        except Exception:                           # a checkpoint from another format
            continue
    return None


def save_stage(model: TinyLM, name: str, stage: str, parent: str | None = None) -> str:
    p = run_path(name)
    model.save(p, TOKENIZER_PATH, extra={"stage": stage, "parent": os.path.basename(parent) if parent else None})
    return p


D_MODEL = preset(TAG).d_model
paths: dict[str, str] = {}

# ================================================================ 0. corpus
with Stage("0. corpus and curation report (Chapter 8)") as st:
    N_DOCS = 1500 if args.quick else 6000
    EVAL_QUESTIONS = [
        "What color was the kite that Mia carried up the tallest hill in Storyland?",
        "Which animal returned the lost silver bell to Leo at the old stone bridge?",
        "How many pears did Zoe count in the purple box beside the frozen lake?",
    ]
    clean = make_corpus(N_DOCS, seed=0)
    dirty = add_noise(clean, seed=0, eval_questions=EVAL_QUESTIONS)
    junk = [d["text"] for d in dirty if d.get("planted") in ("spam", "non_english", "too_short")]
    n_lab = min(300, int(len(junk) * 0.6))
    clf = QualityClassifier().fit([d["text"] for d in clean[:n_lab]] + junk[:n_lab], [1] * n_lab + [0] * n_lab)
    report = CurationReport()
    curated = curate(dirty, eval_texts=EVAL_QUESTIONS, clf=clf, report=report)
    print(report.table())
    print(f"   {len(dirty):,} raw docs -> {len(curated):,} curated")
    st.note = f"{len(dirty):,} raw -> {len(curated):,} docs"
    check(len(curated) < len(dirty) and all(d.get("planted") != "contaminated" for d in curated),
          "curation drops planted junk and every contaminated document")

# ============================================================= 1. tokenizer
with Stage("1. tokenizer (Chapter 2)") as st:
    tok = get_tokenizer()
    sample = corpus_text(curated[:300])
    ratio = tok.compression_ratio(sample)
    print(f"   {os.path.relpath(TOKENIZER_PATH)}: vocab {tok.vocab_size} ({len(tok.special_tokens)} specials), "
          f"{ratio:.2f} bytes/token on the curated sample")
    st.note = f"vocab {tok.vocab_size}, {ratio:.2f} bytes/token"
    check(tok.vocab_size > 256 and ratio > 2.5, "the course tokenizer compresses Storyland > 2.5×")

# ============================================================ 2. pretrain
with Stage("2. base model: pretrain or load (Chapter 10)") as st:
    base_path = BASE_QUICK if args.quick else BASE_FULL
    had_base = os.path.exists(base_path)
    base, _ = get_base_model(quick=args.quick, verbose=True)
    train_tokens, val_tokens = get_tokens(tok)
    T_EVAL = min(128, base.cfg.max_seq_len)
    ppl_base = perplexity(base, val_tokens, batch_size=16, seq_len=T_EVAL, n_batches=5)
    print(f"   {'loaded' if had_base else 'trained'} {os.path.basename(base_path)}: {base.num_params():,} non-embedding params, "
          f"{base.cfg.n_layers} layers × d {base.cfg.d_model}; val perplexity {ppl_base:.2f}")
    st.note = f"{'reused' if had_base else 'trained'} {os.path.basename(base_path)}, ppl {ppl_base:.2f}"
    paths["base"] = base_path
    check(ppl_base < 10, "the base model's Storyland perplexity is far below uniform (~870)")

# ======================================================== 3. mid-training
with Stage("3. mid-training anneal on a math-heavy mix (Chapter 13)") as st:
    p = existing(f"capstone_mid_{TAG}.pt", "lab13_annealed.pt", d_model=D_MODEL, parent=paths["base"])
    eval_docs = make_corpus(600, seed=12345)
    math_eval = tokenize_and_pack([d for d in eval_docs if d["source"] == "math"], tok)
    if p is None:
        model = copy.deepcopy(base)
        steps, B = (40, 8) if args.quick else (120, 16)
        mix = tokenize_and_pack(mix_sources(get_corpus(), {"stories": 1, "math": 4}, n_out=3000, seed=0), tok)
        before = estimate_loss(model, math_eval, 16, T_EVAL, 4)
        opts = build_optimizer(model, "adamw", lr=3e-4, weight_decay=0.1)
        g = torch.Generator().manual_seed(0)
        model.train()
        for step in range(steps):
            set_lr(opts, lr_at(step, steps, 1.0, 0, "wsd", decay_frac=1.0))     # linear decay to 0
            x, y = get_batch(mix, B, T_EVAL, g)
            _, loss = model(x, y)
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for o in opts:
                o.step()
        model.eval()
        after = estimate_loss(model, math_eval, 16, T_EVAL, 4)
        print(f"   {steps} anneal steps on stories:math = 1:4; held-out MATH loss {before:.3f} -> {after:.3f}")
        p = save_stage(model, f"capstone_mid_{TAG}.pt", "mid-training", parent=paths["base"])
        st.note = f"{steps} steps, math loss {before:.3f} -> {after:.3f}"
        check(after < before, "annealing on a math-heavy mix lowers the held-out math loss")
    else:
        model = TinyLM.load(p)
        print(f"   reusing {os.path.basename(p)}; held-out MATH loss {estimate_loss(model, math_eval, 16, T_EVAL, 4):.3f}")
        st.note = f"reused {os.path.basename(p)}"
    paths["mid"] = p

# =================================================================== 4. SFT
with Stage("4. supervised fine-tuning (Chapter 15)") as st:
    p = existing(f"sft_{TAG}.pt", f"lab15_sft_{TAG}.pt", "lab20_teacher_sft_small.pt", f"lab17_sft_{TAG}.pt",
                 f"capstone_sft_{TAG}.pt", d_model=D_MODEL, parent=paths["mid"])
    sft_examples = make_sft_examples(800, seed=1)
    val_examples = make_sft_examples(24, seed=2)
    if p is None:
        model = TinyLM.load(paths["mid"])
        steps = 120 if args.quick else 200
        cfg = SFTConfig(steps=steps, batch_size=16, lr=3e-4, warmup_steps=10, log_every=max(1, steps // 4),
                        eval_every=steps)
        hist = sft_train(model, tok, sft_examples, cfg, val_examples=val_examples, verbose=True)
        print(f"   SFT loss {hist.train_loss[0]:.3f} -> {hist.train_loss[-1]:.3f}; val accuracy {hist.val_acc[-1]:.2f}")
        p = save_stage(model, f"capstone_sft_{TAG}.pt", "sft", parent=paths["mid"])
        st.note = f"{steps} steps, loss {hist.train_loss[0]:.2f} -> {hist.train_loss[-1]:.2f}"
        check(hist.train_loss[-1] < hist.train_loss[0] * 0.8, "SFT lowers the masked assistant-token loss by > 20%")
    else:
        print(f"   reusing {os.path.basename(p)}")
        st.note = f"reused {os.path.basename(p)}"
    paths["sft"] = p

# ================================================================ 5. DPO
with Stage("5. preference pairs and DPO (Chapter 17)") as st:
    p = existing(f"dpo_{TAG}.pt", f"lab17_dpo_{TAG}.pt", f"capstone_dpo_{TAG}.pt", d_model=D_MODEL, parent=paths["sft"])
    if p is None:
        model = TinyLM.load(paths["sft"])
        synthetic = make_preference_pairs(tasks.make_examples(160 if args.quick else 320, seed=5), n_wrong_styles=1)
        on_policy, stats = make_preference_pairs_from_model(model, tok, tasks.make_examples(24, seed=6),
                                                            n_samples=4, max_new_tokens=16)
        pairs = synthetic + on_policy
        held = make_preference_pairs(tasks.make_examples(40, seed=7), n_wrong_styles=1, seed=1)
        print(f"   {len(synthetic)} synthetic pairs + {len(on_policy)} on-policy pairs "
              f"(sampled {stats['n_prompts']} prompts × 4: accuracy {stats['sample_accuracy']:.2f}, "
              f"{stats['n_all_wrong']} all-wrong, {stats['n_all_correct']} all-right)")
        cfg = DPOConfig(steps=40 if args.quick else 60, batch_size=8, lr=1e-4 if args.quick else 5e-5, beta=0.1,
                        warmup=4, log_every=10)
        ref = make_reference(model)
        before = dpo_eval(model, ref, tok, held, cfg)
        dpo_train(model, ref, tok, pairs, cfg, verbose=True)
        after = dpo_eval(model, ref, tok, held, cfg)
        print(f"   held-out pairs: margin {before['margin']:+.3f} -> {after['margin']:+.3f}, "
              f"pair accuracy {before['accuracy']:.2f} -> {after['accuracy']:.2f}")
        p = save_stage(model, f"capstone_dpo_{TAG}.pt", "dpo", parent=paths["sft"])
        st.note = f"{len(pairs)} pairs, {cfg.steps} steps, margin {before['margin']:+.2f} -> {after['margin']:+.2f}"
        check(after["margin"] > before["margin"] and after["accuracy"] > 0.5, "DPO raises the implicit-reward margin on held-out pairs")
    else:
        print(f"   reusing {os.path.basename(p)}")
        st.note = f"reused {os.path.basename(p)}"
    paths["dpo"] = p

# ================================================================ 6. GRPO
MAX_VALUE = 9 if args.quick else 20
with Stage("6. GRPO with a verifiable reward on addition (Chapter 19)") as st:
    p = existing(f"grpo_{TAG}.pt", f"lab19_grpo_{TAG}.pt", f"capstone_grpo_{TAG}.pt", d_model=D_MODEL, parent=paths["dpo"])
    if p is None:
        model = TinyLM.load(paths["dpo"])
        cfg = GRPOConfig(group_size=8, prompts_per_step=4, max_new_tokens=16, steps=12 if args.quick else 20,
                         lr=2e-4, log_every=3)
        add_train = tasks.make_examples(64, seed=9, tasks=["add"], max_value=MAX_VALUE)
        hist = grpo_train(model, tok, add_train, cfg, verbose=True)
        r0 = sum(h["reward"] for h in hist[:3]) / 3
        r1 = sum(h["reward"] for h in hist[-3:]) / 3
        print(f"   mean reward, first 3 steps {r0:.3f} -> last 3 steps {r1:.3f}; "
              f"entropy {hist[0].get('entropy', float('nan')):.2f} -> {hist[-1].get('entropy', float('nan')):.2f}")
        p = save_stage(model, f"capstone_grpo_{TAG}.pt", "grpo", parent=paths["dpo"])
        st.note = f"{cfg.steps} steps × {cfg.prompts_per_step}×G{cfg.group_size}, reward {r0:.2f} -> {r1:.2f}"
        check(all(math.isfinite(h["loss"]) for h in hist), "every GRPO step produced a finite loss")
    else:
        print(f"   reusing {os.path.basename(p)}")
        st.note = f"reused {os.path.basename(p)}"
    paths["grpo"] = p

# ================================================================= 7. OPD
with Stage("7. on-policy distillation into a student (Chapter 20)") as st:
    teacher = TinyLM.load(paths["grpo"])
    if args.quick:
        # quick: same-size "self-distillation" — the RL'd model teaches the pre-RL SFT model
        student, student_name = TinyLM.load(paths["sft"]), "the SFT checkpoint (same size)"
    else:
        sp = existing("sft_nano.pt", "lab20_student_sft_nano.pt", "capstone_sft_nano.pt", d_model=preset("nano").d_model)
        if sp is None:                                    # a nano student that at least knows the chat format
            student, _ = get_base_model(quick=True, verbose=False)
            sft_train(student, tok, sft_examples, SFTConfig(steps=60, batch_size=16, lr=3e-4, warmup_steps=5,
                                                             log_every=100), verbose=False)
            student_name = "base_nano + 60 SFT steps"
        else:
            student, student_name = TinyLM.load(sp), os.path.basename(sp)
    p = existing(f"capstone_opd_{TAG}.pt", d_model=student.cfg.d_model, parent=paths["grpo"])
    if p is None:
        cfg = distill.OPDConfig(steps=6 if args.quick else 16, group_size=4, prompts_per_step=4, max_new_tokens=16,
                                lr=2e-4, log_every=max(1, (6 if args.quick else 16) // 4))
        add_train = tasks.make_examples(64, seed=9, tasks=["add"], max_value=MAX_VALUE)
        hist = distill.opd_train(student, teacher, tok, add_train, cfg, verbose=True)
        print(f"   student = {student_name}, teacher = {os.path.basename(paths['grpo'])}: "
              f"reverse KL {hist[0]['reverse_kl']:.3f} -> {hist[-1]['reverse_kl']:.3f}")
        p = save_stage(student, f"capstone_opd_{TAG}.pt", "opd", parent=paths["grpo"])
        st.note = f"{student_name} <- teacher, {cfg.steps} steps, rKL {hist[0]['reverse_kl']:.2f} -> {hist[-1]['reverse_kl']:.2f}"
        check(all(math.isfinite(h["loss"]) for h in hist), "every OPD step produced a finite loss")
    else:
        print(f"   reusing {os.path.basename(p)}")
        st.note = f"reused {os.path.basename(p)}"
    paths["opd"] = p

# =============================================================== 8. evals
with Stage("8. evaluate every checkpoint (Chapter 23)") as st:
    n_all, n_add = (40, 24) if args.quick else (60, 30)
    held_all = tasks.make_examples(n_all, seed=2024)
    held_add = tasks.make_examples(n_add, seed=2025, tasks=["add"], max_value=MAX_VALUE)
    contam = contamination_check(clean, held_all)
    print(f"   {n_all} mixed-task + {n_add} addition questions; contamination vs the corpus: {contam:.2f} "
          f"(8-gram overlap; story_qa prompts quote Storyland by design)")
    rows = []
    for name in ("base", "mid", "sft", "dpo", "grpo", "opd"):
        m = TinyLM.load(paths[name])
        t0 = time.perf_counter()
        r_all = eval_tasks(m, tok, held_all, max_new_tokens=16)
        r_add = eval_tasks(m, tok, held_add, max_new_tokens=16)
        ppl = perplexity(m, val_tokens, batch_size=16, seq_len=min(128, m.cfg.max_seq_len), n_batches=5)
        rows.append((name, os.path.basename(paths[name]), r_all.accuracy, r_add.accuracy, ppl, time.perf_counter() - t0))
        sample = next((c for _, c, ok in r_add.samples if ok), r_add.samples[0][1])
        print(f"   {name:<5} {os.path.basename(paths[name]):<26} acc(all) {r_all.accuracy:.2f}  acc(add) {r_add.accuracy:.2f}  "
              f"ppl {ppl:6.2f}   e.g. {held_add[0].prompt!r} -> {r_add.samples[0][1][:24]!r}")
    st.note = f"{len(rows)} checkpoints × {n_all + n_add} questions"
    check(rows[2][2] > rows[0][2], "SFT beats the base model on the mixed tasks (a base model cannot answer questions)")

# ============================================================ 9. the agent
with Stage("9. a TinyLM agent answers with a calculator tool (Chapters 21, 24)") as st:
    SYS = "You are TinyLM. Use the calc tool for arithmetic, then answer."
    calc = Tool("calc", "Evaluate an arithmetic expression.",
                {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
                lambda expression: str(safe_eval(expression)))
    reg = ToolRegistry([calc])
    system_text = TinyLMBackend.to_chat_messages([], reg.schemas(), SYS)[0]["content"]   # exactly what the backend renders
    Q = "What is 17 + 25?"

    # (a) the scripted reference: what a tool-using turn looks like in this harness
    scripted = ScriptedBackend([{"text": "", "tool_calls": [{"name": "calc", "arguments": {"expression": "17 + 25"}}]},
                                "17 + 25 = 42"])
    t_ref = Agent(scripted, reg, AgentConfig(max_turns=3, permission_policy="allow_all"), system_prompt=SYS).run(Q)
    print("   scripted reference transcript:\n" + "\n".join("      " + l for l in t_ref.pretty().splitlines()))

    # (b) a tool-SFT'd TinyLM, reused if a checkpoint exists, else trained now from the GRPO model
    p = existing(f"tool_sft_{TAG}.pt", f"agent_sft_{TAG}.pt", f"capstone_tool_{TAG}.pt", d_model=D_MODEL, parent=paths["grpo"])
    if p is None:
        model = TinyLM.load(paths["grpo"])
        model.extend_context(512)                          # the tool schema alone is ~130 tokens
        rng = random.Random(3)
        convs = []
        for _ in range(300):
            a, b = rng.randint(0, 40), rng.randint(0, 40)
            convs.append([{"role": "system", "content": system_text},
                          {"role": "user", "content": f"What is {a} + {b}?"},
                          {"role": "tool_call", "content": json.dumps({"name": "calc", "arguments": {"expression": f"{a} + {b}"}})},
                          {"role": "tool_result", "content": str(a + b)},
                          {"role": "assistant", "content": f"{a} + {b} = {a + b}"}])
        n_tok = len(tok.encode(chat.render(convs[0], add_generation_prompt=False)))
        steps = 60 if args.quick else 200
        losses = distill.sft_steps(model, tok, convs, steps=steps, lr=1e-3, batch_size=4)
        print(f"   tool-SFT: {len(convs)} traces of {n_tok} tokens, {steps} steps, loss {losses[0]:.3f} -> {losses[-1]:.3f}")
        p = save_stage(model, f"capstone_tool_{TAG}.pt", "tool-sft", parent=paths["grpo"])
        st.note = f"tool-SFT {steps} steps, loss {losses[0]:.2f} -> {losses[-1]:.2f}"
    else:
        model = TinyLM.load(p)
        print(f"   reusing {os.path.basename(p)}")
        st.note = f"reused {os.path.basename(p)}"
    paths["tool"] = p
    backend = TinyLMBackend(model, tok, max_new_tokens=80)      # a tool-call turn is ~51 tokens
    probe = backend.complete([{"role": "user", "content": Q}], reg.schemas(), SYS)
    print(f"   raw generation for {Q!r}: {probe.raw[:100]!r}")
    t_real = Agent(backend, reg, AgentConfig(max_turns=3, permission_policy="allow_all"), system_prompt=SYS).run(Q)
    print("   TinyLM transcript:\n" + "\n".join("      " + l for l in t_real.pretty().splitlines()))
    ex = tasks.TaskExample("add", Q, "17 + 25 = 42", {"answer": 42})
    called = t_real.tool_calls_made > 0
    right = tasks.verify(ex, t_real.final_text) == 1.0
    agent_line = (f"TinyLM {'called the tool' if called else 'did not call the tool'} and its final answer was "
                  f"{'correct' if right else 'wrong'} ({t_real.final_text[:30]!r})")
    print("   " + agent_line)
    st.note += f"; {agent_line}"
    check(t_real.stop_reason in ("done", "max_turns"), "the agent loop terminates with a real model behind it")
    check(t_ref.final_text == "17 + 25 = 42" and t_ref.tool_calls_made == 1, "the scripted reference agent uses the tool and answers")

# ============================================================== report
total = time.perf_counter() - T_LAB
lines = [f"# Capstone report ({TAG} model, {'quick' if args.quick else 'full'} mode)", "",
         f"{device_summary()} | load average at start {load:.1f} | total wall-clock {total:.0f}s", "",
         "## Stages", "", "| stage | what happened | seconds |", "|---|---|---:|"]
lines += [f"| {n} | {note} | {s:.1f} |" for n, note, s in stages]
lines += ["", "## Every checkpoint on the same questions", "",
          f"{n_all} mixed-task questions (add, sub, reverse, upper, count, first, story_qa), {n_add} addition questions "
          f"(operands ≤ {MAX_VALUE}), perplexity on held-out Storyland at T={T_EVAL}. Contamination (8-gram) {contam:.2f}.", "",
          "| stage | checkpoint | acc (all) | acc (add) | perplexity | eval s |", "|---|---|---:|---:|---:|---:|"]
lines += [f"| {n} | {ck} | {a:.2f} | {b:.2f} | {ppl:.2f} | {s:.0f} |" for n, ck, a, b, ppl, s in rows]
lines += ["", "## Agent", "", agent_line, ""]
with open(run_path("capstone_report.md"), "w") as f:
    f.write("\n".join(lines))
banner("capstone summary")
print("\n".join(lines[6:6 + len(stages) + 2]))
print()
print("\n".join(lines[-len(rows) - 2 - 5:-5]))
print(f"\nreport written to {os.path.relpath(run_path('capstone_report.md'))}; total {total:.0f}s")

fig, axes = plt().subplots(1, 2, figsize=(11, 3.6))
ax = axes[0]
x = range(len(rows))
ax.bar([i - 0.2 for i in x], [r[2] for r in rows], width=0.4, color="#2563eb", label="accuracy (all tasks)")
ax.bar([i + 0.2 for i in x], [r[3] for r in rows], width=0.4, color="#f59e0b", label="accuracy (addition)")
ax.set_xticks(list(x)); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylim(0, 1); ax.legend(fontsize=8)
ax.set_title(f"Lab 29: every stage on the same questions ({TAG})")
ax = axes[1]
ax.barh(range(len(stages)), [s for _, _, s in stages], color="#16a34a")
ax.set_yticks(range(len(stages))); ax.set_yticklabels([n.split(" (")[0][:34] for n, _, _ in stages], fontsize=7)
ax.invert_yaxis(); ax.set_xlabel("seconds"); ax.set_title("where the time went")
fig.tight_layout()
savefig(fig, "lab29_capstone.png")
done()
