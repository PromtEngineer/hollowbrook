# LLMs From Scratch to Agents — Course Outline

**Who this is for.** Someone who can write basic Python and wants to understand large language
models (LLMs) deeply enough to build one, train it through every stage, evaluate it, and wrap it in
an agent harness. No machine-learning background is assumed. Every idea is introduced with a plain
explanation first, then a picture, then code you run.

**What you build.** A small but complete language model called **TinyLM** that runs on a laptop CPU.
You take it through the exact stages a frontier model goes through in 2026:

```
raw text ──▶ curation ──▶ tokenizer ──▶ pretraining ──▶ mid-training ──▶ SFT ──▶ reward model / DPO
        ──▶ RL with verifiable rewards (GRPO) ──▶ on-policy distillation ──▶ evals ──▶ agent harness
```

Each stage is a chapter with a runnable **lab**, an **interactive explainer** you can play with in a
browser, worked **code snippets**, **exercises**, and a **self-check quiz**.

**Time.** About 60–80 hours if you do every lab and exercise. Each chapter is 1–3 hours.

**Currency.** Techniques are covered up to September 2026: Muon, sparse and hybrid attention,
MoE, MTP, FP8/FP4 training, GRPO and its 2026 variants (DAPO, Dr. GRPO, GSPO), rubric rewards,
on-policy distillation, agentic RL, MCP/A2A, long-running harness patterns. Where something is
still an open research question, the chapter says so.

---

## How to read the outline

Each chapter lists:

- **Goal** — the one sentence you should be able to say afterwards.
- **You will build** — the code artifact.
- **Interactive** — the browser playground in `course/interactive/`.
- **Lab** — the script in `course/labs/` that runs the stage.

---

## Part 0 — Orientation

### Chapter 0: The whole pipeline on one page
- Goal: Name every stage from raw text to a deployed agent and say what goes in and comes out of each.
- Covers: what an LLM is (a next-token predictor), why "training" has several stages, what
  base / instruct / reasoning / agentic models are, a map of the course, setting up your machine.
- Interactive: `00_pipeline_map.html` — click any stage to see inputs, outputs, and the chapter that covers it.
- Lab: `lab00_setup.py` — checks your Python, PyTorch and CPU, runs a 30-second smoke test of TinyLM.

## Part I — Foundations: what a language model is

### Chapter 1: Language models are next-token predictors
- Goal: Explain "predict the next token" and build a model that does it with counting alone.
- Covers: probability of a sentence, the chain rule, n-gram models, why counting stops working,
  perplexity as "how surprised is the model", sampling text from a distribution.
- You will build: a bigram/trigram model in 40 lines of Python.
- Interactive: `01_ngram_playground.html` — type text, watch the count table fill, generate.
- Lab: `lab01_ngram.py`.

### Chapter 2: Tokenization
- Goal: Turn text into integers and back, and explain why subword tokens won.
- Covers: characters vs words vs subwords, byte-pair encoding (BPE) step by step, pre-tokenization
  regexes, special tokens, vocabulary size trade-offs, the famous failure modes (numbers,
  whitespace, non-English, "strawberry"), 2026 directions (byte-level models, BLT, tokenizer-free).
- You will build: `llm/tokenizer.py` — a byte-level BPE tokenizer you train on your own corpus.
- Interactive: `02_bpe_stepper.html` — watch merges happen one at a time on your own text.
- Lab: `lab02_tokenizer.py` — train a 4k-vocab tokenizer, measure compression, inspect merges.

### Chapter 3: Embeddings — words as vectors
- Goal: Explain what an embedding is and why similar tokens end up near each other.
- Covers: one-hot vectors, lookup tables, dot products and cosine similarity, what "dimension" means,
  how embeddings are learned (they are just parameters), tied input/output embeddings.
- Interactive: `03_vector_playground.html` — drag 2-D vectors, see dot products and similarity.
- Lab: `lab03_embeddings.py` — inspect embeddings of a trained TinyLM, nearest neighbours.

### Chapter 4: How neural networks learn (the minimum you need)
- Goal: Explain loss, gradient, backpropagation and an optimizer step in your own words.
- Covers: a neuron, a layer, a loss function, the gradient as "which way is downhill", autograd
  (a 100-line scalar autograd engine), SGD, momentum, Adam/AdamW, learning rate, batch size,
  overfitting, train/validation split.
- You will build: `llm/microautograd.py` — a tiny autograd engine, then the same in PyTorch.
- Interactive: `04_gradient_descent.html` — a loss landscape you can roll a ball down with different optimizers.
- Lab: `lab04_autograd.py`.

### Chapter 5: Attention
- Goal: Explain attention as a soft, learned lookup and implement it.
- Covers: the problem attention solves, queries/keys/values, scaled dot-product attention, causal
  masking, multi-head attention, positional information and RoPE, efficient variants (MQA, GQA,
  MLA), FlashAttention as an implementation trick not a new model, sliding windows.
- You will build: `llm/model.py::Attention` with GQA and RoPE.
- Interactive: `05_attention_explorer.html` — type a sentence, see the attention matrix per head, toggle the causal mask.
- Lab: `lab05_attention.py` — verify your attention against PyTorch's reference numerically.

### Chapter 6: The Transformer block and the residual stream
- Goal: Draw a Transformer block from memory and count its parameters.
- Covers: the residual stream as shared memory that layers read from and write to, pre-norm vs
  post-norm, RMSNorm, the MLP (and SwiGLU), why depth and width matter, parameter and FLOP counting,
  initialization, what each part of the block "does" (intuitions from interpretability).
- You will build: `llm/model.py::Block` and `TinyLM`.
- Interactive: `06_block_dataflow.html` — animated data flow through a block, live parameter counter.
- Lab: `lab06_build_tinylm.py` — assemble TinyLM, check shapes, generate untrained text.

### Chapter 7: Inference — turning probabilities into text
- Goal: Explain every sampling knob and why generation is memory-bound.
- Covers: greedy, temperature, top-k, top-p, min-p, repetition penalties, the KV cache, why
  prefill and decode are different, batching, speculative decoding, quantization (INT8/INT4/FP8) at
  a glance, the cost model of serving.
- You will build: `llm/generate.py` with a KV cache.
- Interactive: `07_sampling_playground.html` — move temperature/top-p sliders and see the distribution reshape.
- Lab: `lab07_generate.py` — measure tokens/sec with and without a KV cache.

## Part II — Pretraining

### Chapter 8: Pretraining data — where trillions of tokens come from
- Goal: Describe a modern web-data pipeline and run one on a small corpus.
- Covers: Common Crawl, text extraction, language identification, heuristic filters (Gopher/C4
  rules), exact and fuzzy deduplication (MinHash), model-based quality classifiers (FineWeb-Edu,
  DCLM), PII handling, decontamination against evals, synthetic rephrasing (Nemotron-CC), code and
  math data, data mixing and curriculum, the 2026 quality-vs-diversity debate, licensing and ethics.
- You will build: `llm/data.py` — a curation pipeline (filter → dedup → classify → decontaminate → mix).
- Interactive: `08_data_pipeline.html` — feed documents through each filter and see what is dropped and why.
- Lab: `lab08_curate.py` — curate a planted-noise corpus; measure what each stage removes.

### Chapter 9: Scaling laws and compute budgets
- Goal: Estimate how big a model and how much data a compute budget buys.
- Covers: parameters, tokens, FLOPs (the 6·N·D rule), Kaplan vs Chinchilla, tokens-per-parameter
  in 2026 (overtraining small models), inference-aware scaling, muP and hyperparameter transfer,
  what a $100 / $10k / $10M run buys, reading a loss curve.
- Interactive: `09_scaling_calculator.html` — slide a compute budget, get model size, tokens, wall-clock.
- Lab: `lab09_scaling.py` — train three TinyLM sizes and fit a tiny scaling law.

### Chapter 10: The pretraining loop
- Goal: Pretrain TinyLM end-to-end and explain every line of the training loop.
- Covers: batching and packing, learning-rate warmup and schedules (cosine, WSD), weight decay,
  gradient clipping, mixed precision (bf16, FP8, FP4/NVFP4 in 2026), AdamW vs Muon (implemented),
  checkpointing and resuming, logging, loss spikes and how to handle them, validation loss.
- You will build: `llm/train.py`, `llm/optim.py::Muon`.
- Interactive: `10_training_dynamics.html` — a simulated run; change LR/schedule and watch stability.
- Lab: `lab10_pretrain.py` — a 5–10 minute CPU pretraining run; AdamW vs Muon comparison.

### Chapter 11: Distributed training
- Goal: Explain data, tensor, pipeline, expert and sequence parallelism and when each is used.
- Covers: why one GPU is not enough, data parallel and all-reduce, ZeRO/FSDP sharding, tensor
  parallel, pipeline parallel and bubbles, expert parallel for MoE, context/sequence parallel,
  4-D parallelism in 2026 trillion-parameter runs, MFU, communication vs compute, fault tolerance.
- Interactive: `11_parallelism_visualizer.html` — split a model across N devices and watch the traffic.
- Lab: `lab11_data_parallel.py` — 2-process data-parallel training on CPU with `torch.distributed`.

### Chapter 12: Modern architectures (2026)
- Goal: Read a 2026 model card and recognise every architectural term.
- Covers: Mixture-of-Experts (routing, load balancing, shared experts, DeepSeekMoE), sparse
  attention (NSA, DeepSeek Sparse Attention, compressed KV), hybrid linear/state-space layers
  (Mamba, gated DeltaNet, Nemotron-H, Kimi Linear), multi-token prediction, hyper-connections,
  long context to 1M tokens, diffusion language models, what stayed the same since 2017.
- You will build: `llm/model.py::MoE` — a mixture-of-experts MLP you can drop into TinyLM.
- Interactive: `12_moe_router.html` — tokens flow to experts; watch load balancing succeed or collapse.
- Lab: `lab12_moe.py` — train dense vs MoE TinyLM at equal active parameters.

### Chapter 13: Mid-training and continued pretraining
- Goal: Explain what happens between pretraining and post-training and why it matters in 2026.
- Covers: the annealing phase on high-quality data, long-context extension (RoPE scaling, YaRN),
  domain adaptation, adding instruction and reasoning data before SFT, curriculum, avoiding
  catastrophic forgetting, the base-model checkpoint as a product.
- Lab: `lab13_midtrain.py` — anneal TinyLM on a curated high-quality slice; extend context length.

## Part III — Post-training

### Chapter 14: The post-training pipeline
- Goal: Draw the 2026 post-training pipeline and say what each stage changes in the model.
- Covers: base → instruct → reasoning → agentic, chat templates and special tokens, the
  stages (SFT, preference learning, RLVR, on-policy distillation, safety), on-policy vs off-policy
  data, what "alignment" means operationally, how labs actually sequence it in 2026.
- Interactive: `14_posttraining_map.html`.

### Chapter 15: Supervised fine-tuning (SFT)
- Goal: Turn a base model into an instruction follower and explain loss masking.
- Covers: what instruction data looks like, where it comes from (human-written, synthetic,
  distilled, rejection-sampled), chat formats, masking the prompt from the loss, packing,
  hyperparameters (small LR, few epochs), full fine-tuning vs LoRA, what SFT can and cannot teach.
- You will build: `llm/sft.py`, `llm/chat.py`.
- Lab: `lab15_sft.py` — SFT TinyLM on a tiny instruction set; watch it start following formats.

### Chapter 16: Data labeling and curation for post-training
- Goal: Design a labeling task, measure annotator agreement, and curate a preference dataset.
- Covers: writing annotation guidelines, rubrics, pairwise vs Likert vs ranking, inter-annotator
  agreement (Cohen's kappa), quality control, using LLMs as labelers and their biases, synthetic
  preference data, filtering and deduplicating instruction data, data mixing for post-training.
- You will build: `interactive/16_labeling_tool.html` — a working preference-labeling UI that exports JSONL.
- Lab: `lab16_labeling.py` — compute agreement on your labels, build a preference set.

### Chapter 17: Reward models and preference optimization
- Goal: Train a reward model and run DPO, and explain when to use which.
- Covers: Bradley–Terry preferences, training a scalar reward model, reward hacking and
  Goodhart, DPO derived in plain language, variants (IPO, KTO, SimPO, ORPO), rubric-based and
  generative reward models (2026), LLM-as-judge as a reward.
- You will build: `llm/reward.py`, `llm/dpo.py`.
- Interactive: `17_dpo_explorer.html` — move the chosen/rejected log-probs, watch the DPO loss.
- Lab: `lab17_reward_dpo.py`.

### Chapter 18: Reinforcement learning for language models
- Goal: Derive the policy-gradient update and explain PPO's clipping and the KL penalty.
- Covers: RL vocabulary translated to LLMs (state = prompt + tokens so far, action = next token,
  reward at the end), REINFORCE, why variance is the enemy, baselines, advantages, PPO clipping,
  the KL-to-reference penalty, why LLM RL is mostly a bandit problem, RLHF end to end.
- You will build: `llm/rl.py::reinforce`, `ppo_loss`.
- Interactive: `18_policy_gradient.html` — a bandit whose policy you can watch update.
- Lab: `lab18_reinforce.py`.

### Chapter 19: GRPO, RLVR and reasoning models
- Goal: Implement GRPO with verifiable rewards and reproduce the "reasoning emerges" effect at toy scale.
- Covers: verifiable rewards, group-relative advantages, the GRPO objective, the DeepSeek-R1
  recipe, the 2026 fixes (DAPO clip-higher and dynamic sampling, Dr. GRPO, token-level loss,
  GSPO, CISPO), entropy collapse, length control, test-time compute and parallel thinking,
  what RLVR does and does not teach.
- You will build: `llm/rl.py::grpo_step` with DAPO options.
- Interactive: `19_grpo_simulator.html` — a group of rollouts, rewards, advantages and clipping, live.
- Lab: `lab19_grpo.py` — teach TinyLM multi-digit addition with a verifiable reward.

### Chapter 20: Distillation and on-policy distillation
- Goal: Explain teacher–student training and implement on-policy distillation.
- Covers: logit distillation, sequence-level distillation, why off-policy SFT on teacher outputs
  is brittle, on-policy distillation (student samples, teacher grades every token), the 2026
  recipes and when OPD fails, distillation as the cheapest post-training stage.
- You will build: `llm/distill.py`.
- Lab: `lab20_opd.py` — distill a larger TinyLM into a smaller one on-policy.

### Chapter 21: Agentic RL — training models to use tools
- Goal: Train a policy over multi-turn tool interactions and explain the infrastructure it needs.
- Covers: trajectories vs single responses, tool-call formats, environments and sandboxes,
  turn-level vs trajectory-level rewards, credit assignment, async rollouts, the 2026 frameworks
  (verl, SkyRL, AgentRL, ProRL-Agent), failure-driven data, reward hacking in agents.
- You will build: `llm/rl.py::CalculatorEnv` — a toy tool environment; `llm/rl.py::multi_turn_grpo_step`.
- Lab: `lab21_agentic_rl.py` — TinyLM learns to call a calculator tool.

### Chapter 22: Safety, alignment and model specs
- Goal: Explain how 2026 models are made helpful, honest and harmless, and where that fails.
- Covers: constitutions and model specs, Constitutional AI and RLAIF, refusal training,
  sycophancy, red-teaming, jailbreaks, safety evals, behaviour under agentic autonomy,
  interpretability as an audit tool.
- Lab: `lab22_constitution.py` — RLAIF with a tiny written constitution on TinyLM.

### Chapter 23: Evaluation
- Goal: Build an eval harness and explain why benchmark numbers mislead.
- Covers: perplexity vs capability, exact-match and unit-test evals, LLM-as-judge, contamination,
  the 2026 benchmark landscape (SWE-bench Verified, Terminal-Bench, HLE, ARC-AGI-3), agentic evals,
  statistical significance, building evals for your own use case, eval-driven development.
- You will build: `llm/evals.py`.
- Lab: `lab23_evals.py` — evaluate every TinyLM checkpoint you have trained so far.

## Part IV — Agents and harnesses

### Chapter 24: From model to agent — the loop
- Goal: Write an agent loop from scratch and explain tool calling.
- Covers: the observe–think–act loop, tool schemas and function calling, parsing tool calls,
  stop conditions, errors as observations, single-turn vs multi-turn, backends (your TinyLM, an API model, a scripted mock for tests).
- You will build: `llm/agent/harness.py`, `llm/agent/tools.py`, `llm/agent/backends.py`.
- Interactive: `24_agent_loop_tracer.html` — step through an agent transcript one event at a time.
- Lab: `lab24_agent_loop.py`.

### Chapter 25: Context engineering
- Goal: Manage a context window deliberately: what goes in, when, and what gets removed.
- Covers: the context budget, system prompts, tool results and their size, compaction and
  summarisation, memory files and scratchpads, retrieval and just-in-time context, sub-agents as
  context isolation, prompt caching, "context rot" and 2026 findings on long-horizon agents.
- Interactive: `25_context_budget.html` — watch a context fill up, compact, and recover.
- Lab: `lab25_context.py` — add compaction and memory to your harness.

### Chapter 26: Tools, MCP and agent protocols
- Goal: Write an MCP server and client and explain what MCP and A2A standardise.
- Covers: designing good tools (names, descriptions, error messages), the Model Context Protocol
  (resources, tools, prompts, transports), a minimal MCP server in Python, tool search for large
  tool sets, A2A for agent-to-agent, sandboxing and permissions, security threats (prompt
  injection through tool results).
- You will build: `llm/agent/mcp_mini.py`.
- Lab: `lab26_mcp.py`.

### Chapter 27: Harness engineering for long-running agents
- Goal: Build a coding agent harness with permissions, hooks, verification, and resumability.
- Covers: the harness as the product, permission gates, hooks (pre/post tool), verification loops
  (tests as ground truth), plan files and progress files, checkpointing and resuming across context
  windows, the initializer/coder and planner/generator/evaluator patterns, skills, sub-agents,
  observability and cost, 2026 lessons from production harnesses.
- You will build: `llm/agent/miniharness.py` — a coding agent that edits files and runs tests.
- Interactive: `27_harness_anatomy.html`.
- Lab: `lab27_miniharness.py` — the agent fixes a failing test in a sandbox repo.

### Chapter 28: Multi-agent systems
- Goal: Choose a multi-agent pattern (or decline one) for a given task.
- Covers: orchestrator–workers, pipelines, parallel fan-out, debate and verification, shared
  state, A2A, failure modes (duplicated work, lost context, runaway cost), when a single agent is better.
- Lab: `lab28_multiagent.py` — a two-agent generator/evaluator loop.

### Chapter 29: Capstone — TinyLM end to end, then the scale-up map
- Goal: Run the whole pipeline in one sitting and know exactly what changes at scale.
- Covers: the capstone script, a comparison table of every checkpoint, what nanochat's $100
  run does differently, what a 1B / 10B / 100B+ run adds (data, infra, parallelism, FP8,
  Muon, MoE, evals), a reading plan for staying current.
- Lab: `lab29_capstone.py`.

## Appendices
- A. Math refresher (vectors, matrices, probability, logs, derivatives) — with pictures.
- B. PyTorch in one page.
- C. Glossary (every bold term in the course).
- D. Reading list, 2017 → September 2026, grouped by chapter.
- E. Hardware and cost guide (laptop → 8×H100 → cluster).
- F. Troubleshooting (loss not going down, NaNs, slow training, reward hacking symptoms).

---

## Learning path variants

- **Fast path (20 h):** 0, 1, 2, 5, 6, 7, 10, 14, 15, 19, 24, 27, 29.
- **Training focus:** Parts I–III in full, skim Part IV.
- **Agents focus:** 0, 1, 2, 7, 14, 21, 23, Part IV in full.

## Conventions used everywhere

- **Bold** on a term means it is defined right there and appears in the glossary.
- 📐 marks a visual explainer, 🧪 a lab, 🎛️ an interactive, ✍️ an exercise, ✅ a self-check.
- Code snippets are taken from the library in `course/llm/` and are tested. If a snippet in a
  chapter differs from the library, the library is right — open an issue.
