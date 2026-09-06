# Chapter 29: Capstone — TinyLM end to end, then the scale-up map

**Part IV · ~3 hours · Prerequisites: every chapter; at minimum 2, 8, 10, 13, 15, 17, 19, 20, 23, 24, 27**

> 🎯 Goal: Run the whole pipeline in one sitting and know exactly what changes at scale.
> 🧪 Lab: `labs/lab29_capstone.py` · 🎛️ Interactive: none for this chapter (open `interactive/00_pipeline_map.html` again and tick the stages off)

## Why this matters

The outline promised that you would take a language model through every stage a frontier model goes through in 2026, on a laptop, with code you can read in full. This chapter cashes that promise in one script: `lab29_capstone.py` starts from raw documents and ends with a model calling a calculator tool inside the harness of Chapter 27, running curation, tokenization, pretraining, mid-training, SFT, DPO, GRPO, on-policy distillation and an evaluation of every checkpoint on the same questions in between. The **capstone** is not a new technique; it is the discipline of running the old ones in one line and reading one table at the end. The second half of the chapter is the map you need next: what a $100 run does differently, what changes at 1B, 10B and 100B+, how to keep up, and what you can now do.

## The idea in pictures 📐

The pipeline is the one from Chapter 0, now with a filename at every arrow:

```mermaid
flowchart LR
    R["raw docs<br/>make_corpus + add_noise"] --> C["curate()<br/>CurationReport"]
    C --> T["BPETokenizer<br/>runs/tokenizer.json"]
    T --> B["pretrain / load<br/>base_{nano,small}.pt"]
    B --> M["anneal on math mix<br/>capstone_mid_*.pt"]
    M --> S["sft_train<br/>sft_*.pt"]
    S --> D["pairs + dpo_train<br/>capstone_dpo_*.pt"]
    D --> G["grpo_train on add<br/>grpo_*.pt"]
    G --> O["opd_train → student<br/>capstone_opd_*.pt"]
    G --> A["tool-SFT → TinyLMBackend agent<br/>capstone_tool_*.pt"]
    B & M & S & D & G & O --> E["eval_tasks + perplexity<br/>runs/capstone_report.md"]
```

Read it as: every stage consumes the previous checkpoint and writes its own; the evaluation reads all of them. Two rules make the script re-runnable. **Checkpoint reuse**: each stage first looks for a checkpoint in `runs/` (its own `capstone_*` file or the one the chapter's lab produced, such as `sft_nano.pt` or `grpo_small.pt`) and trains only if none exists, so a second run takes seconds, and a crash in stage 7 does not cost you stages 0–6. A lineage rule: every `capstone_*` file records which checkpoint it was trained from (`extra["parent"]`), and it is reused only if that parent is the checkpoint the current run is actually using upstream; otherwise it is retrained, so the chain stays a chain even when an earlier lab's checkpoint appears in `runs/` between two runs.

The second figure is the **scale-up map**: a table with one column per scale (TinyLM, nanochat's $100 run, 1B, 10B, 100B+) and one row per pipeline stage, showing what stops being simple in that stage as you move right. It is discussed in its own section below.

![The scale-up map: what changes from TinyLM to a frontier run](../figures/29_scale_map.svg)

## The idea in code

There is no new library code in this chapter. The script is about 430 lines of calls into functions you have already read, and the pattern for every stage is the same. Here it is with the lab's `existing()` helper reduced to its core (the lab's version also checks the recorded parent):

```python
import os
from llm.model import TinyLM
from llm.pipeline import TOKENIZER_PATH, get_tokenizer, run_path
from llm.sft import SFTConfig, make_sft_examples, sft_train

def existing(*names, d_model):                       # the first checkpoint in runs/ that exists and has the right width
    for n in names:
        p = run_path(n)
        if os.path.exists(p) and TinyLM.load(p).cfg.d_model == d_model:
            return p
    return None

tok = get_tokenizer()
paths = {"mid": run_path("capstone_mid_nano.pt")}                                   # the previous stage's checkpoint
p = existing("sft_nano.pt", "capstone_sft_nano.pt", d_model=96)                    # reuse if one exists ...
if p is None:                                                                       # ... else train from the previous stage
    model = TinyLM.load(paths["mid"])
    sft_train(model, tok, make_sft_examples(800, seed=1), SFTConfig(steps=120, lr=3e-4), verbose=False)
    p = run_path("capstone_sft_nano.pt")
    model.save(p, TOKENIZER_PATH, extra={"stage": "sft", "parent": paths["mid"]})      # records its parent
paths["sft"] = p
print(os.path.basename(p))                          # sft_nano.pt (Lab 15's checkpoint, reused) or capstone_sft_nano.pt
```

The lab wraps each such block in a `Stage` context manager that times it and records a one-line note for the report. `existing()` checks the candidate's width (`cfg.d_model`), so a nano checkpoint is never reused in a `--full` run. **The evaluation** (Chapter 23) is the same call for every checkpoint: greedy decoding, `tasks.verify` as grader, plus perplexity on held-out Storyland:

```python
from llm.evals import eval_tasks, perplexity
from llm.pipeline import get_tokens
from llm import tasks
_, val_tokens = get_tokens(tok)                                            # held-out Storyland
held_all = tasks.make_examples(40, seed=2024)                              # seven task types
held_add = tasks.make_examples(24, seed=2025, tasks=["add"], max_value=9)  # what GRPO trained on
paths["base"] = run_path("base_nano.pt")                                   # the lab has all six stages here
for name in ("base", "sft"):
    m = TinyLM.load(paths[name])
    r_all, r_add = eval_tasks(m, tok, held_all, max_new_tokens=16), eval_tasks(m, tok, held_add, max_new_tokens=16)
    ppl = perplexity(m, val_tokens, batch_size=16, seq_len=128, n_batches=5)
    print(f"{name}: acc(all) {r_all.accuracy:.2f}  acc(add) {r_add.accuracy:.2f}  ppl {ppl:.2f}")
# base: acc(all) 0.00  acc(add) 0.00  ppl 2.90
# sft:  acc(all) 0.40  acc(add) 0.00  ppl 840.26
```

**Tool-SFT** is supervised fine-tuning on conversations that contain a tool call and a tool result, so that the model learns to emit the harness's format. The training conversations use *exactly* the system prompt that `TinyLMBackend` renders at inference (the prompt plus the tool schemas as JSON), because a format learned under one system prompt does not transfer to another:

```python
from llm.agent import TinyLMBackend, Tool, ToolRegistry
from llm.agent.tools import safe_eval
SYS = "You are TinyLM. Use the calc tool for arithmetic, then answer."
calc = Tool("calc", "Evaluate an arithmetic expression.",
            {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
            lambda expression: str(safe_eval(expression)))
reg = ToolRegistry([calc])
system_text = TinyLMBackend.to_chat_messages([], reg.schemas(), SYS)[0]["content"]     # compact tool listing, not JSON schemas
conv = [{"role": "system", "content": system_text},
        {"role": "user", "content": "What is 17 + 25?"},
        {"role": "tool_call", "content": '{"name": "calc", "arguments": {"expression": "17 + 25"}}'},
        {"role": "tool_result", "content": "42"},
        {"role": "assistant", "content": "17 + 25 = 42"}]
```

`sft_train` accepts such conversations directly (`sft_train(model, tok, convs, SFTConfig(steps=200, batch_size=4, lr=1e-3, warmup_steps=5, max_len=512))`), and `build_sft_example` masks everything except the `tool_call` and `assistant` turns, so the model is trained to produce the call and the final answer and never the tool's output. `TinyLMBackend` lists tools compactly by default (name, description, argument names) because a full JSON schema costs well over a hundred extra Storyland tokens per tool (123 for `calc`; Lab 27 measured about 170 per tool for the seven built-ins); a whole trace is 197 tokens, of which the tool-call turn is 51 (the JSON is spelled out nearly byte by byte), which is why the agent's `max_new_tokens` is 80 and the model's context is extended to 512 for this stage.

## Worked example 🧪

```bash
python3 labs/lab29_capstone.py            # --quick: the nano model; about 5 minutes the first time, 30 s when everything is reused
python3 labs/lab29_capstone.py --full     # the small model; about 25 minutes the first time
python3 labs/lab29_capstone.py --fresh    # retrain every capstone_* checkpoint
```

The timings below come from a 4-core machine with two PyTorch threads (`OMP_NUM_THREADS=2`) while other labs were running (load average 4–8); the script drops to one thread only when the load exceeds three times the core count. Because the earlier labs' checkpoints in `runs/` are what the capstone reuses, your table will differ from this one whenever those labs have been re-run. The lab prints two tables: the **stage table** (what each stage did, what it reused, how long it took) and the **checkpoint comparison table** (the same evaluation applied to every saved checkpoint, so stages can be compared rather than admired).

### Quick mode (nano model)

Quick mode is what you get after working through the course in order, because `runs/` then already holds Lab 15's `sft_nano.pt`, Lab 17's `lab17_dpo_nano.pt` and Lab 19's `grpo_nano.pt`. The curation report is Chapter 8's table on 1,500 clean documents plus planted noise:

```
stage                           kept  dropped  planted problems caught
language_filter                 1824      114  non_english:45, spam:69
heuristic_filter                1773       51  spam:6, too_short:45
pii_scrub (rewrites)            1773        0  pii:45
exact_dedup                     1615      158  exact_dup:79, pii:1
minhash_dedup                   1518       97  exact_dup:1, near_dup:32, pii:19
quality_classifier              1514        4  pii:4
decontaminate                   1511        3  contaminated:3
   1,938 raw docs -> 1,511 curated
```

Then the base model is loaded, mid-training runs for 40 steps the first time (`held-out MATH loss 1.770 -> 1.666`; reused afterwards), three stages report `reusing ...`, and the two stages no earlier lab produced in this form (OPD and tool-SFT) run for real the first time and are reused after that:

```
--- 7. on-policy distillation into a student (Chapter 20) ---
opd step    0 | reverse KL 5.2856 | acc 0.00 | len   7.8 | 0.2s
opd step    5 | reverse KL 5.5846 | acc 0.00 | len   8.4 | 1.2s
   student = the SFT checkpoint (same size), teacher = grpo_nano.pt: reverse KL 5.286 -> 5.585
```

A reverse KL of 5 nats per token says the SFT and GRPO models disagree violently about what to say next, and six OPD steps are too few to move it; the stage exists here to show the mechanics. On an empty `runs/` the script trains every stage itself (309 s the first time on a busy machine): SFT's masked loss fell from 9.61 to 3.38 in 120 steps at a validation accuracy of 0.04; DPO's held-out margin rose from 0.00 to +1.86 on synthetic pairs alone (the SFT model was wrong on all 24 sampled prompts, so no on-policy pairs); GRPO's reward stayed at 0.06–0.07 for 12 steps with `acc 0.00` throughout, because a group with no correct answer has nothing to learn from. A 300k-parameter model learns the *format* of an answer long before the answers.

The table that the whole course was building towards, with each checkpoint's answer to `'What is 7 + 2?'`:

```
| stage | checkpoint            | acc (all) | acc (add) | perplexity | eval s |   'What is 7 + 2?' ->
|---|---|---:|---:|---:|---:|
| base  | base_nano.pt          | 0.00 | 0.00 |    2.90 | 2 |   '+   78.'
| mid   | capstone_mid_nano.pt  | 0.00 | 0.00 |    2.89 | 2 |   '+    + 18.'
| sft   | sft_nano.pt           | 0.40 | 0.00 |  840.26 | 1 |   '3 + 63 = 87'
| dpo   | lab17_dpo_nano.pt     | 0.00 | 0.21 | 2526.88 | 2 |   '7 + 2 = 8'
| grpo  | grpo_nano.pt          | 0.00 | 0.17 |  982.33 | 2 |   '7 + 2 = 8'
| opd   | capstone_opd_nano.pt  | 0.35 | 0.00 | 1705.34 | 1 |   '3 + 63 = 97'
```

Base and mid continue the prompt as a Storyland document and answer nothing. SFT (Lab 15 trained it on `upper`, `reverse`, `add` and `count`) produces the shape of an answer and scores 0.40 on the mixed tasks, at a Storyland perplexity of 840 instead of 2.9. Lab 17's DPO and Lab 19's GRPO checkpoints were trained on addition alone: they copy the operands (`7 + 2 = ...`), get a fifth of the single-digit sums right, and score zero on everything else, at perplexities in the thousands on the text they were pretrained on. That is the alignment tax of Chapter 14 in its broadest sense, capability lost to post-training (here fluency lost to task-tuning), taken to its limit by a 300k-parameter model pushed hard on one task; a bigger model pays it in a much smaller coin. OPD moved the SFT model toward the GRPO teacher without changing either score. With 40 and 24 questions the 95 % intervals are about ±0.15 and ±0.2, so `sft` versus `opd` is a tie; the base-versus-rest gap and the addition scores of `dpo` and `grpo` are the real differences (Chapter 23).

The last stage is the agent:

```
[sft] 300 examples | 120 steps | batch 4 | 379,200 trainable params
step     0 | loss 10.0813 | lr 2.00e-04 | grad_norm 33.61 | 1,854 tok/s
step    60 | loss 0.9257 | lr 5.81e-04 | grad_norm 1.69 | 2,570 tok/s
step   119 | loss 0.4089 | lr 1.00e-04 | grad_norm 0.85 | 2,641 tok/s
   tool-SFT: 300 traces of 197 tokens, 120 steps, loss 10.081 -> 0.409
   raw generation for 'What is 17 + 25?': '<|tool_call|>{"name": "calc", "arguments": {"expression": "26 + 30"}}<|end|>'
   TinyLM transcript:
      USER: What is 17 + 25?
      ASSISTANT:
        -> call calc({"expression": "26 + 30"})
        <- result: 56
      ASSISTANT: 37 + 30 = "
      [done after 2 turns, 1 tool calls]
   TinyLM called the tool and its final answer was wrong ('37 + 30 = "')
✅ the agent loop terminates with a real model behind it
✅ the scripted reference agent uses the tool and answers
6/6 checks passed in 121.7s
```

120 steps of the ordinary SFT loop (warmup, cosine decay, a masked loss on the two assistant turns) are enough for the nano model to emit a well-formed tool call that the harness parses and executes. The operands are invented (`26 + 30` for `17 + 25`), the final answer ignores the `56` it was given and trails off into a quote mark. Sixty steps were not enough under the cosine schedule (loss 1.68, near-JSON with misspelled keys); the same sixty steps at a constant `1e-3` were, which is a reminder that a schedule halves the learning rate you thought you had. The whole quick run took two minutes with the machine shared the first time; with every checkpoint reused it takes about 30 s, whatever the load.

### Full mode (small model)

```
| stage | what happened | seconds |
|---|---|---:|
| 0. corpus and curation report (Chapter 8) | 7,743 raw -> 6,045 docs | 6.0 |
| 1. tokenizer (Chapter 2) | vocab 871, 4.04 bytes/token | 0.0 |
| 2. base model: pretrain or load (Chapter 10) | reused base_small.pt, ppl 2.07 | 11.3 |
| 3. mid-training anneal on a math-heavy mix (Chapter 13) | reused lab13_annealed.pt | 4.2 |
| 4. supervised fine-tuning (Chapter 15) | reused sft_small.pt | 0.2 |
| 5. preference pairs and DPO (Chapter 17) | reused lab17_dpo_small.pt | 0.2 |
| 6. GRPO with a verifiable reward on addition (Chapter 19) | reused grpo_small.pt | 0.3 |
| 7. on-policy distillation into a student (Chapter 20) | sft_nano.pt <- teacher, 16 steps, rKL 3.25 -> 2.73 | 26.7 |
| 8. evaluate every checkpoint (Chapter 23) | 6 checkpoints × 90 questions | 108.0 |
| 9. a TinyLM agent answers with a calculator tool (Chapters 21, 24) | tool-SFT 200 steps, loss 10.37 -> 0.06; TinyLM called the tool and its final answer was wrong ('17 + 25 = 41') | 239.6 |

| stage | checkpoint | acc (all) | acc (add) | perplexity | eval s |   'What is 20 + 15?' ->
|---|---|---:|---:|---:|---:|
| base | base_small.pt         | 0.00 | 0.00 |   2.07 | 24 |   '.'
| mid  | lab13_annealed.pt     | 0.00 | 0.00 |   2.10 | 18 |   '.'
| sft  | sft_small.pt          | 0.48 | 0.20 | 183.38 | 24 |   '20 + 15 = 15'
| dpo  | lab17_dpo_small.pt    | 0.00 | 0.67 |   8.74 | 27 |   '20 + 15 = 37'
| grpo | grpo_small.pt         | 0.00 | 0.67 |   8.82 | 11 |   '20 + 15 = 37'
| opd  | capstone_opd_small.pt | 0.32 | 0.00 | 669.21 |  4 |   '3 + 3 = 97'

7/7 checks passed in 425.1s
```

Seven minutes on a shared machine, because every stage the earlier labs had already produced in `--full` mode was reused; when the capstone trained SFT, DPO and GRPO itself (an earlier run) it took 837 s. The reused checkpoints tell a cleaner story than the ones this script trains in a few minutes. Lab 15's `sft_small.pt` scores 0.48 on the mixed tasks and 0.20 on addition at a Storyland perplexity of 183. Lab 17's DPO checkpoint, trained on addition alone, reaches 0.67 on addition and nothing on the other tasks at a perplexity of 8.7, and Lab 19's GRPO checkpoint keeps both numbers; on 30 questions 0.20 and 0.67 are distinguishable (intervals of about ±0.15 that do not overlap), and 0.67 and 0.67 are not. The OPD student is the nano SFT model of Lab 15 distilled from the small GRPO teacher; 16 steps lower the reverse KL from 3.25 to 2.73 without changing its scores, and its 0.32 on the mixed tasks comes from Lab 15, not from distillation.

Two experiments behind this table are worth knowing, because you will meet both on your own checkpoints. GRPO from a *weak* policy learns nothing: most groups are all-wrong, `skipped` is 0.75–1.00. GRPO from a *good* policy (Lab 17's DPO model, 0.67 on addition) at `lr = 2e-4` with no reference model *destroys* it: reward 0.61 → 0.31 in 20 steps, held-out accuracy 0.67 → 0.07, because with `ppo_epochs = 1` nothing is clipped and Adam turns a few hundred noisy answer tokens into full-size updates. A KL anchor to a frozen copy (`kl_coef = 0.05`) at `lr = 2e-5` keeps the accuracy (0.67 → 0.70), which is the script's configuration for the small model; Chapter 19's KL term is the difference between improving a policy and wrecking one.

The agent is the finale, and it deserves to be read line by line:

```
   tool-SFT: 300 traces of 197 tokens, 200 steps, loss 10.365 -> 0.062
   raw generation for 'What is 17 + 25?': '<|tool_call|>{"name": "calc", "arguments": {"expression": "17 + 25"}}<|end|>l: "17 + 25 = 34<|end|><'
   TinyLM transcript:
      USER: What is 17 + 25?
      ASSISTANT:
        -> call calc({"expression": "17 + 25"})
        <- result: 42
      ASSISTANT: 17 + 25 = 41
      [done after 2 turns, 1 tool calls]
   TinyLM called the tool and its final answer was wrong ('17 + 25 = 41')
```

After 200 steps of tool-SFT, starting from Lab 19's GRPO model, the 2.5M-parameter model emits a syntactically perfect tool call with the right operands, the harness runs `safe_eval("17 + 25")`, and `42` comes back as a `tool_result` turn. Then the model's second turn answers `17 + 25 = 41`: it copied the operands but did not copy the result it was handed. That is nearly the whole course in four lines: a pretrained model (Chapter 10), taught a chat format (15), sharpened on a task (17, 19), taught a tool protocol, driven by the loop of Chapter 24, and caught by a verifier rather than by itself (27). The raw generation shows what the harness hides: after `<|end|>` the model keeps going (`l: "17 + 25 = 34`) and `TinyLMBackend` keeps only the part before the first foreign role tag. The outcome is a coin flip at this scale: two earlier runs of the same recipe produced `17 + 25 = 42` (correct, tool result copied) and `calc("6 + 13")` followed by `6 + 13 = 21` (wrong operands, tool result ignored), and the nano model above never copies the operands at all. Copying two numbers out of a 200-token prompt into a template is a harder skill than the template, and it is the skill Chapter 21's reward pays for; tool-SFT teaches the *protocol* of tool use, agentic RL its *purpose*. The harness did its Chapter 27 job in every case: a bounded loop, a validated call, a sandboxed execution, and a transcript that records the call and the answer, so that a verifier, not the model, has the last word.

The report is saved to `runs/capstone_report.md` and the figure to `figures/generated/lab29_capstone.png` (accuracy per stage on the left, seconds per stage on the right).

## The scale-up map

Everything in the lab has a counterpart in a frontier run. The figure `29_scale_map.svg` is the table; this section walks its columns.

### What nanochat's $100 run does differently

**nanochat** (Karpathy, October 2025) is the closest public relative of this course: one repository that trains a tokenizer, pretrains a Transformer, mid-trains, SFTs, optionally runs a little RL, evaluates, and serves a chat UI, in about four hours on 8×H100 for roughly $100. Set against Lab 29, the differences are instructive because they are *not* differences of kind:

- **Data volume.** TinyLM's whole corpus is about 0.4 million Storyland tokens, which the small base model passes over roughly eight times (700 steps × 32 × 128 tokens); nanochat streams about eleven billion FineWeb-Edu tokens (Chapter 9) from disk shards through the same kind of packed-window loader, with the tokens-per-parameter ratio chosen from a scaling rule (Chapter 9).
- **Tokenizer.** Its own byte-level BPE with 2¹⁶ = 65,536 entries; ours saturates at 871 because Storyland has 401 distinct chunks.
- **Model and optimizer.** ~560M parameters in a decoder block of the same family as the one you built (pre-norm, RoPE; the attention and MLP details differ, and the repository's `gpt.py` is the place to see how), Muon for the matrices and AdamW for the rest, in bfloat16; its speedrun lineage (modded-nanogpt) reached a GPT-2-grade result in a reported ~1.35 minutes on 8×H100 by April 2026 with Muon, FlashAttention-3, an FP8 head and multi-token prediction.
- **Parallelism.** Data parallelism across eight GPUs with an all-reduce per step (Lab 11 did this across two CPU processes).
- **Evals.** A fixed suite (a CORE-style aggregate, ARC, GSM8K, HumanEval, MMLU) on every checkpoint: the capstone's table with better questions.
- **Post-training.** SFT and a little RL on GSM8K; no preference stage, no distillation, no safety training. A $100 model is a research artefact, not a product.

nanochat is Lab 29 with some 10⁴ times the data, about 200 times the parameters, real benchmarks and a GPU. Every function name maps.

### What changes at 1B, 10B and 100B+

**Data volume and curation infrastructure.** At 1B parameters a compute-optimal run wants tens of billions of tokens and an over-trained one (Chapter 9) hundreds of billions; deduplication and quality classification become a cluster job over a shard store, with MinHash (Chapter 8) at petabyte scale. At 10B+ the mix is engineered: tuned source weights, a curriculum, decontamination against every eval you will report, synthetic rephrasing (Nemotron-CC, FineInstructions). At the 10T+-token horizon the 2026 evidence is that diversity beats aggressive filtering and that curation pipelines are themselves being evolved by models.

**Tokenizer size.** 32k–128k vocabularies at 1–10B, 128k–256k at the frontier, trained on a sample of the actual pretraining mix, with digit chunking and code-whitespace merges deliberately designed. The rule from Chapter 2 holds: the tokenizer and the mix are designed together.

**Precision, optimizer and architecture.** FP8 training is the 2026 default at scale; NVFP4 recipes are reported validated on multi-trillion-token runs to ~120B, MXFP4 pretraining is under study, and Hadamard rotations keep FP4 stable. Muon is mainstream (Kimi K2, GLM-4.5/5 and DeepSeek-V4 report using it), with the commonly cited ≈2× compute-efficiency over AdamW and a July 2026 study reporting it matches or beats AdamW on hybrid Mamba-attention MoE models. At 100B+ the model is a mixture of experts: DeepSeek-V4 (~1.6T parameters) keeps DeepSeekMoE and MTP, adds compressed sparse attention for million-token context and replaces plain residuals with manifold-constrained hyper-connections. Chapter 12 built the MoE block; what changes is that expert placement becomes a networking problem.

**4-D parallelism.** Chapter 11's taxonomy becomes the daily job: tensor parallelism inside a node, pipeline across nodes, expert parallelism for the MoE layers, data parallelism over everything, context parallelism for long sequences; 40 % MFU is a good large-run number and 50 % exceptional (Chapter 9), and fault tolerance is a feature. The loop of Chapter 10 is unchanged in shape.

**Evaluation infrastructure.** From a table on every checkpoint to a service: held-out internal evals, contamination checks on every source, agentic benchmarks (SWE-bench Verified, Terminal-Bench) in containers, and, after the 2026 audits (flawed tests in a majority of hard SWE-bench Verified tasks; BenchJack's benchmark exploits), a standing effort to audit the benchmarks themselves.

**RL infrastructure with async rollouts.** Chapter 19's `grpo_step` samples, scores and updates in one process. At scale the sampler is a separate inference fleet with paged KV caches running **async rollouts** (generation runs continuously on its own workers and the trainer consumes finished trajectories as they arrive), with a small policy lag (corrected by the clipped ratio you implemented), and agentic environments (Chapter 21) are containers started per episode; AgentRL, SkyRL-Agent, verl and rollout-as-a-service are the 2026 frameworks. The algorithmic fixes (DAPO, Dr. GRPO, GSPO, adaptive rollouts) address the entropy collapse you saw when every group had zero variance. On-policy distillation from a large teacher (Chapter 20) is reported 10–30× cheaper than RL for comparable gains.

**Safety.** Absent from the lab and from nanochat, mandatory in a product: a model spec (Chapter 22), RLAIF, refusal training, red-teaming, agentic-autonomy evals, interpretability as audit. The harness of Chapter 27 is the last line, not the first.

### What does not change

The next-token loss, byte-level BPE, the pre-norm block with RoPE, GQA and SwiGLU, the KV cache, loss-masked SFT, a KL-anchored preference loss, group-relative advantages, a verifier as the reward. Every function you called in Lab 29 has a name in a frontier codebase; what scales is the data, the infrastructure around each function, and the number of people whose whole job is one row of the map.

## A reading plan for staying current

The field moves monthly and most of what moves is engineering, so a **reading plan** is a habit, not a list. The one that works for practitioners in 2026:

1. **Primary sources first.** Read each major open-weight tech report in full (DeepSeek-V4, Kimi K3, GLM-5.x, Qwen3.x, gpt-oss, Gemma 4, Llama 4) and map its sections onto this course's chapters; a section with no chapter is what to learn next.
2. **One reference per stage**, replaced when a better one appears: FineWeb/DCLM/Nemotron-CC for data; Muon for optimisation; DeepSeekMoE, NSA/DSA and the hybrid SSM papers for architecture; DPO and rubric rewards for preferences; GRPO, DAPO, Dr. GRPO, GSPO for RL; the Thinking Machines blog for OPD; the SWE-bench Verified audit for evals; the Anthropic harness posts and the context-rot papers for agents.
3. **Aggregators as pointers** (Turing Post, Cameron Wolfe's surveys, the open-weight comparison pages, InfoQ): use them to find primary sources, and say "reported" when you repeat something you only read there.
4. **Reproduce one claim a month at toy scale.** Does clip-higher raise entropy? Does OPD beat rejection-sampling SFT at equal budget? Does a longer context hurt the agent? The library is built for this; it is the only reading method that produces understanding rather than familiarity.
5. **Read the code.** nanochat and modded-nanogpt for training; verl and SkyRL for RL infrastructure; the MCP and A2A specifications for agents; compare each with the file in `llm/`.

## What you can now do

A checklist to tick honestly.

- [ ] Compute a perplexity by hand from a bigram table (Ch. 1); train a byte-level BPE tokenizer and measure bytes/token (Ch. 2).
- [ ] Draw a Transformer block from memory, count its parameters, implement GQA attention with RoPE that matches PyTorch (Ch. 5–6); generate with a KV cache and explain every sampling knob (Ch. 7).
- [ ] Run a curation pipeline with dedup, a quality classifier and decontamination, and read its report (Ch. 8); size a model and its data for a budget and read a loss curve (Ch. 9).
- [ ] Pretrain with AdamW or Muon, with warmup, a schedule, clipping, checkpointing and resume (Ch. 10); explain the parallelisms (Ch. 11); build an MoE layer (Ch. 12); anneal and extend context (Ch. 13).
- [ ] Turn a base model into an assistant with loss-masked SFT or LoRA (Ch. 15); design a labeling task and measure agreement (Ch. 16).
- [ ] Train a reward model and run DPO (Ch. 17); derive the policy gradient and PPO's clip (Ch. 18); implement GRPO with a verifiable reward and its 2026 fixes (Ch. 19).
- [ ] Distil on-policy and say when it fails (Ch. 20); train a policy over multi-turn tool use (Ch. 21); run RLAIF against a written constitution (Ch. 22).
- [ ] Build an eval harness with confidence intervals, a contamination check and a position-bias check (Ch. 23).
- [ ] Write an agent loop with tools, a permission gate and hooks; manage a context budget; write an MCP server and client (Ch. 24–26).
- [ ] Build a harness whose sessions resume from files and whose "done" is decided by tests (Ch. 27); choose, measure or decline a multi-agent pattern (Ch. 28).
- [ ] Run the whole pipeline in one script, read the table of every checkpoint, and read a 2026 model card knowing which chapter built each term and what changes at scale.

## Try it yourself ✍️

1. **Fix a stage.** Force the capstone to train GRPO (`--fresh` with Lab 19's checkpoint renamed) with `MAX_VALUE = 5`, `steps = 40` and `temperature = 1.2`. Does any group get a non-zero variance, and does the reward move?
2. **Reuse across labs.** Run Labs 15, 17 and 19 in `--full` mode, then the capstone in `--full`. Which stages say "reused", and does the table change?
3. **A fairer table.** Add bootstrap confidence intervals (`llm.evals.bootstrap_ci`) to the checkpoint table and mark which differences are real at 95 %.
4. **The alignment tax, measured.** Add a column with perplexity on the *chat-formatted* validation examples (as `compare_checkpoints` does) and check whether it falls while the Storyland perplexity rises.
5. **Tool-SFT data.** Double the number of tool traces and include subtraction. Does the model copy the operands correctly more often? Count, over 20 questions, how many tool calls parse and how many have the right expression.
6. **Your own scale-up row.** Pick a 2026 open-weight tech report and fill in one column of the scale-up map with the numbers it states; mark every number you could not find.

## Check yourself ✅

<details><summary>1. Why does the capstone reuse checkpoints, and what could go wrong if it reused them blindly?</summary>

Reuse makes the pipeline cheap to re-run and resumable after a failure: a stage loads its checkpoint from `runs/` and trains only if none exists. Blind reuse breaks the lineage: a DPO checkpoint trained from one SFT model would be evaluated as if it followed a different one, and a nano checkpoint could be loaded into a small run. The script checks `d_model` and reuses its own files only when their recorded parent matches.
</details>

<details><summary>2. In the quick-mode table, Storyland perplexity rose from 2.90 (base) to 840 (SFT) and into the thousands (DPO, GRPO) while task accuracy rose from 0 to 0.40. Is that a bug?</summary>

No. Storyland perplexity measures how well the model continues stories; SFT, DPO and GRPO train it to answer questions instead, and every step away from the pretraining distribution costs fluency (the small model pays less: 2.07 to 8.7 for DPO in full mode). This is Chapter 14's alignment tax at toy scale, and why post-training is evaluated with task accuracy and preference metrics, not perplexity.
</details>

<details><summary>3. GRPO's reward stayed flat in quick mode. What diagnostic in the log explains it, and what would you change first?</summary>

`acc 0.00` on every step: every group of eight answers was wrong, so within each group the rewards were equal, advantages were zero and the update had nothing to push on. Change the starting point or the task difficulty (smaller operands, a better SFT model) before the RL hyperparameters; GRPO needs some correct samples to learn from.
</details>

<details><summary>4. Name three things nanochat does differently from Lab 29 and one thing it does not do at all.</summary>

Differently: ~10⁴× more data from FineWeb-Edu shards, a 65,536-entry tokenizer, a ~560M-parameter model trained with Muon in bfloat16 across eight GPUs, real benchmarks. Not at all: safety training (nor a preference stage or distillation).
</details>

<details><summary>5. At 100B+ parameters, which stage changes most in <em>kind</em> rather than in size, and why?</summary>

RL infrastructure. At toy scale sampling, scoring and updating happen in one process; at scale the sampler is a separate inference fleet with async rollouts, the trainer tolerates a policy lag corrected by the clipped ratio, and environments are containers started per episode. Curation and parallelism grow enormously but keep their shape; async RL changes the shape of the loop.
</details>

## Key takeaways

- The whole pipeline fits in one script because every stage is a function that loads a checkpoint and saves one; reuse and a lineage rule make it re-runnable.
- The table of every checkpoint on the same questions is the deliverable; single-stage logs are diagnostics.
- At toy scale stages show the right diagnostics rather than large gains: perplexity rises through post-training, GRPO needs correct samples, tool-SFT teaches a format before copying.
- nanochat is the same pipeline with 10⁴× the data and a GPU; every function maps.
- What changes at 1B → 100B+ is data volume and curation infrastructure, tokenizer size, FP8/FP4 and Muon, MoE, 4-D parallelism, eval infrastructure, async RL, and safety; what does not change is the loss, the block, the mask, the KL anchor, the group baseline and the verifier.
- Staying current is a habit: primary sources, one reference per stage, aggregators as pointers, and one toy reproduction a month.

## Going deeper

- 🆕 Karpathy, A. *nanochat* (October 2025), https://github.com/karpathy/nanochat ; Jordan, K. *modded-nanogpt* (2024–2026), https://github.com/KellerJordan/modded-nanogpt . The $100 pipeline and the speedrun lineage; read them file by file against `llm/`.
- 🆕 "SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales" (July 2026), https://arxiv.org/abs/2607.20548 ; PyTorch blog on Muon with DeepSpeed, https://pytorch.org/blog/using-muon-optimizer-with-deepspeed/
- 🆕 DeepSeek-AI, *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence* (2026), https://arxiv.org/abs/2606.19348 ; discussion at https://www.lmsys.org/blog/2026-04-25-deepseek-v4/
- 🆕 MXFP4 pretraining on native FP4 hardware (2026), https://arxiv.org/abs/2605.09825
- 🆕 Data at the 10T+ horizon: FineInstructions (2026), https://arxiv.org/abs/2601.22146 ; DataEvolve (2026), https://arxiv.org/abs/2603.14420
- 🆕 RL infrastructure: AgentRL (2025), https://arxiv.org/abs/2510.04206 ; verl agentic RL docs, https://verl.readthedocs.io/en/latest/start/agentic_rl.html ; Turing Post, "Reasoning RL in 2026", https://www.turingpost.com/p/reasoning-rl-in-2026
- 🆕 Thinking Machines, "On-policy distillation" (October 2025), https://thinkingmachines.ai/blog/on-policy-distillation/ ; "Rethinking On-Policy Distillation" (2026), https://arxiv.org/abs/2604.13016

---

← [Chapter 28](28-multi-agent-systems.md) · [Course home](../README.md) · [Appendix A](A-math-refresher.md) →
