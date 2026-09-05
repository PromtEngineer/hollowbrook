# LLMs From Scratch to Agents

A hands-on course that takes you from "what is a language model?" to training your own small
LLM through every modern stage — data curation, tokenizer, pretraining, mid-training, SFT,
reward models, DPO, GRPO with verifiable rewards, on-policy distillation, agentic RL, evals —
and then wrapping it in an agent harness with tools, MCP, context management and verification.

Everything runs on a laptop CPU. The model you train is small (**TinyLM**, ~2.5 M parameters), but
the *code path* is the same one frontier labs use in 2026, and every chapter ends with the
"what changes at scale" story.

> **Start here:** [OUTLINE.md](OUTLINE.md) for the map, then [Chapter 0](chapters/00-the-whole-pipeline.md).

## Setup (5 minutes)

```bash
cd course
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install torch numpy matplotlib pytest regex tqdm     # CPU PyTorch is fine
python3 labs/lab00_setup.py                              # smoke test: trains a tiny model for 30 s
```

Python 3.10+ and any CPU. A GPU is never required; where one helps, the chapter says so.

## How the course is organised

| Folder | What is in it |
|---|---|
| `chapters/` | 30 chapters + appendices, in reading order (`00-...` to `29-...`) |
| `labs/` | One runnable script per chapter. `--quick` (default, < 1 min) or `--full` (minutes) |
| `interactive/` | Browser playgrounds, one per chapter. Open `interactive/index.html` |
| `llm/` | The library you build: tokenizer, model, training, RL, agents (≈ 4 k lines, all tested) |
| `figures/` | Diagrams used by the chapters; `figures/generated/` is written by the labs |
| `tests/` | `python3 -m pytest tests -q` — every library function is covered |
| `runs/` | Checkpoints and tokenizers the labs produce (git-ignored) |

## The learning loop for each chapter

1. Read the chapter (intuition → picture → code).
2. Open the interactive and do its **Challenge**.
3. Run the lab: `python3 labs/labNN_*.py` then `--full`.
4. Do the ✍️ exercises and the ✅ self-check.
5. Tick it off in [PROGRESS.md](PROGRESS.md).

## Parts

- **Part 0 — Orientation**: the whole pipeline on one page.
- **Part I — Foundations**: next-token prediction, tokenization, embeddings, how networks learn,
  attention, the Transformer block, inference.
- **Part II — Pretraining**: data curation, scaling laws, the training loop (AdamW and Muon),
  distributed training, 2026 architectures (MoE, sparse and hybrid attention, MTP), mid-training.
- **Part III — Post-training**: SFT, labeling and curation, reward models and DPO, RL fundamentals,
  GRPO/RLVR and reasoning, distillation, agentic RL, safety, evaluation.
- **Part IV — Agents and harnesses**: the agent loop, context engineering, tools and MCP,
  harness engineering for long-running agents, multi-agent systems, capstone.

## Verifying everything

```bash
python3 -m pytest tests -q          # library tests (~3 min on CPU)
python3 labs/run_all_labs.py        # every lab in --quick mode (~15 min)
```

## Currency

Covers techniques through **September 2026**. Each chapter's *Going deeper* section marks 2025–2026
work with 🆕 and links to the source. Appendix D is the full reading list.

## License

Course text: CC BY 4.0. Code: MIT.
