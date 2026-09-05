# Appendix F: Troubleshooting

Symptoms → likely causes → what to try. Ordered by how often each bites in practice.

## F.1 Loss is not going down

1. **Learning rate wrong by 10×.** Too low: loss creeps. Too high: loss plateaus above the untrained
   value or spikes. Sweep `lr` over `{1e-4, 3e-4, 1e-3, 3e-3}` for 100 steps each (Lab 10 does this
   for you). Small models tolerate high LRs; Muon uses `0.02`-scale rates for matrices.
2. **Targets not shifted.** `y` must be `x` shifted one token left (`llm/train.py::get_batch`).
   If `y == x` the model learns to copy and loss drops to ~0 instantly — a different bug with an
   opposite symptom.
3. **Loss mask wrong** (SFT). If you mask *everything*, the loss is 0/0 → NaN or constant.
   Print `mask.sum()` per batch; decode the trainable tokens to check they are the assistant's.
4. **No warmup with a high LR.** The first steps blow up the optimizer state. Use 50–200 warmup steps.
5. **Data is garbage or too repetitive.** Decode a batch and read it. Every time.
6. **Tokenizer/model vocabulary mismatch.** `cfg.vocab_size` must be `tok.vocab_size`; ids ≥ V index
   out of the embedding (error) or an unused row (silent).

## F.2 Loss is NaN or Inf

- LR too high (most common). Halve it; add warmup; clip gradients to 1.0.
- `log(0)`: a probability of exactly zero after masking; use `log_softmax`, never `log(softmax)`.
- Overflow in fp16 (not bf16): switch to bf16 or fp32.
- A division by a zero standard deviation (GRPO advantages with identical rewards): add `eps`,
  or skip the group (dynamic sampling, Chapter 19).
- In DPO/GRPO: sequence log-probabilities summed over very long responses are large negative numbers;
  ratios `exp(new − old)` overflow. Clamp the log-ratio (e.g. to ±20) before `exp`.

## F.3 Training is slow

- Measure first: tokens/s printed by `llm/train.py`. TinyLM small ≈ 8–10 k tokens/s on a 4-core CPU.
- Batch too small: matrix multiplies are inefficient below a few thousand tokens per step.
- Python overhead: the MoE loop over experts and the KV-cache concatenation are written for
  clarity; production code fuses them.
- Threads: `torch.get_num_threads()` should equal your physical cores. Two trainings in parallel
  halve each other's speed (the labs share the CPU).
- On GPU: use bf16 autocast, `torch.compile`, FlashAttention; check that data loading is not the bottleneck.

## F.4 The model generates nonsense / repeats itself

- **Untrained or under-trained**: check val loss. Below ~2.5 on Storyland the text is fluent.
- **Temperature 1.0 on a small model** samples the long tail: try `temperature=0.7, top_p=0.9`, or
  `min_p=0.05`.
- **Repetition loops** at temperature 0 are normal for small models; use a repetition penalty or sampling.
- **Wrong prompt format** after SFT: the model expects the exact chat template (`llm/chat.py::render`).
  Missing `<|assistant|>` at the end of the prompt is the classic error.
- **Special tokens injected from user text**: encode user content with `allowed_special=False`.

## F.5 SFT made the model worse

- LR too high (use 1e-4 – 5e-4 for full fine-tuning of a small model; lower at scale) or too many
  epochs (1–3 is normal; more overfits and forgets).
- The instruction data is too small or too narrow; the model learns the *format* but forgets facts.
  Mix in some pretraining data (replay) or use LoRA to limit drift.
- Loss is computed on the prompt too (no mask): the model learns to predict questions, not answers.

## F.6 Reward goes up but the model is worse (reward hacking)

- Verify by *reading samples*. Classic hacks: answering every arithmetic question with "= 0" if the
  format reward is too generous; extremely short or extremely long outputs; repeating the
  question; exploiting a lenient answer extractor (Chapter 17's `extract_answer` is deliberately
  lenient — tighten it and the hack disappears).
- Fixes: make the reward stricter/verifiable, add a length penalty, add a KL penalty to the reference,
  lower the LR, use rubric rewards with several independent criteria.
- Entropy collapse (Chapter 19): the policy becomes deterministic and stops exploring. Raise
  `clip_eps_high` (DAPO clip-higher), lower the LR, or add an entropy bonus.

## F.7 GRPO/RL "does nothing"

- All rewards in a group equal → advantages zero → no gradient. The task is too easy or too hard for
  the current model. Fix the curriculum: start with 1-digit addition, then 2-digit. Use dynamic sampling.
- Group too small (use 8–16 samples per prompt at toy scale).
- Response cut off before the answer: raise `max_new_tokens` or add a length penalty so the model
  learns to finish.
- The reference model is the *same object* as the policy (`ref = model` instead of a `deepcopy`):
  KL is always zero and nothing constrains drift.

## F.8 The agent loops forever / never finishes

- No stop condition: `max_turns` must be set; the agent must be told to answer without a tool call
  when done.
- Tool errors are being swallowed silently; return the error text as the tool result so the model
  can react (Chapter 24: errors are observations).
- Context overflowed: enable compaction (Chapter 25) and truncate tool outputs.
- Permission denials return nothing useful; return "Permission denied: <reason>" as a result so the
  model can choose another route.

## F.9 Numbers do not match the chapter

Small differences (a few percent) are expected: CPU thread counts, PyTorch versions and random seeds
change results. Large differences (loss stuck above 4, accuracy near 0 after SFT) indicate a bug —
run `python3 -m pytest tests -q` first; if the tests pass, compare your lab command line with the
chapter's (`--quick` vs `--full`).
