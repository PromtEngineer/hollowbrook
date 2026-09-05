# Chapter 10: The pretraining loop

**Part II · ~3 hours · Prerequisites: Chapters 4, 6, 8, 9**
> 🎯 Goal: Pretrain TinyLM end-to-end and explain every line of the training loop.
> 🧪 Lab: `labs/lab10_pretrain.py` · 🎛️ Interactive: `interactive/10_training_dynamics.html`

## Why this matters

Everything before this chapter was preparation: a tokenizer, a model that outputs logits, a curated token stream, a budget. This chapter spends the budget. The **pretraining loop** is the program that repeatedly takes a batch of tokens, runs the model forward, measures how wrong its next-token predictions were, pushes the error back through the network to get a gradient, and moves every parameter a small step downhill — a few hundred thousand times for a frontier model, 700 times for TinyLM. `llm/train.py` is 170 lines and contains every mechanism a 2026 frontier run has: learning-rate warmup and decay, gradient accumulation, gradient clipping, decoupled weight decay, mixed precision, periodic validation, checkpointing and resuming, throughput logging, and a choice of two optimizers, AdamW and Muon.

Why read it line by line rather than call `train()` and move on? Because when a run goes wrong — and at scale runs go wrong weekly — the symptom is a number on a dashboard (the loss spiked, the gradient norm is climbing, tokens/second halved) and the fix is one of these lines. The lab trains the same model twice, once with AdamW and once with Muon, kills a run at step 100 and resumes it, and measures the throughput, so that you have seen each mechanism work before you need it.

## The idea in pictures 📐

```mermaid
flowchart LR
    T[packed tokens<br/>1-D tensor] --> B[get_batch<br/>random windows B×T]
    B --> S[one step:<br/>fwd → loss → bwd → clip → update]
    S --> S
    S -->|every log_every| H[History<br/>loss, lr, tok/s]
    S -->|every eval_every| V[estimate_loss<br/>on val tokens]
    V --> K[save_checkpoint<br/>model + optimizer + step + history]
    K -.->|resume_from| S
    style T fill:#e2e8f0,stroke:#64748b
    style K fill:#fef3c7,stroke:#f59e0b
```

Read the flow left to right: the packed stream from Chapter 8 is sliced into random windows, each window drives one step, and two side channels record what happened — a `History` of losses every few steps, and a validation loss plus a checkpoint every hundred. The dotted arrow is resuming: a checkpoint holds everything needed to continue as if the process had never stopped.

![The training step as a cycle](../figures/10_training_loop.svg)

The cycle diagram is one step of `train()`, with the line numbers in `llm/train.py` at the bottom. Start at station 0, the learning rate for this step, and go clockwise: fetch a batch, run the model forward to logits, compute the cross-entropy loss, run backward to fill every parameter's `.grad`, clip the gradient's global norm, let the optimizer update the weights, log. The dashed red arrow inside the circle is **gradient accumulation**: stations 1–4 can repeat several times, adding gradients up, before one clip-and-step, which gives the effect of a larger batch without the memory for one. The panel on the right lists the knobs in `TrainConfig` that control each station.

### The loop, line by line

Open `llm/train.py` alongside this section.

**`TrainConfig` (lines 22–45).** A dataclass of every knob, with defaults that work for TinyLM: 600 steps, batches of 32 windows × 128 tokens, AdamW at a peak LR of 10<sup>−3</sup> (small models like large learning rates; a 70B model uses ~3 × 10<sup>−4</sup>), Muon at 0.02, weight decay 0.1, 50 warmup steps, cosine decay to 10% of peak, gradient clipping at 1.0, validation every 100 steps, logging every 25. The number of tokens the run will see is `steps × batch_size × seq_len × grad_accum` — that is *D* from Chapter 9.

**`get_batch` (48–54).** Batching by **random windows**: pick `batch_size` random start offsets into the packed stream, take `seq_len` tokens from each as the input `x`, and the same window shifted one token to the right as the target `y`. Every position in `x` is a training example: position *t* is trained to predict `y[t] = x[t+1]`. Because the stream is packed with EOS separators, a window may start mid-document and cross into the next one; the model learns to treat EOS as a reset. Real runs do the same thing with a shuffled, sharded stream and an *epoch* (one pass) rather than random windows, but the tensor shapes are identical: `x, y: (B, T)`.

**`estimate_loss` (57–71).** Validation loss on a *fixed* set of windows — the generator is re-seeded with the same seed every call, so two evaluations differ only because the model changed. It switches the model to `eval()` and back so that dropout (off in TinyLM anyway) behaves.

**`History` (74–84)** and **checkpoints (87–101).** A checkpoint holds the model weights, *every optimizer's state* (Adam's running moments, Muon's momentum buffers — without them a resumed run takes a visible hit), the step count, the training config and the history so far. `load_checkpoint` returns the step to continue from.

**`train` (104–170).** The setup: seed PyTorch (108), build the optimizer or the Muon+AdamW pair (111), and if `resume_from` points at a checkpoint, load it and jump `start_step` forward (115–118). The batch generator is seeded with `seed + start_step` (119), so a resumed run draws different batches than the uninterrupted one would have — good enough for training, but it means you cannot expect bit-identical curves after a resume. Then the loop:

- **Learning rate (126–127).** `lr_at` returns a multiplier in [0, 1] for this step, and `set_lr` applies it to every optimizer's base LR. **Warmup** — a linear ramp from 0 over the first `warmup_steps` — exists because Adam's running estimates of gradient scale are unreliable for the first few dozen steps, and a full-size step in a random direction can put the model somewhere it never recovers from. After warmup, **cosine decay** follows half a cosine wave down to `min_lr_ratio × peak` at the last step, and **WSD** (warmup–stable–decay) stays flat at peak until the last 20% of training, then decays linearly to zero. WSD is the 2025–2026 favourite for a practical reason: with cosine, the schedule is tied to the planned length, so you cannot extend a run or branch a "finished" model from the middle of it without retraining; with WSD, any checkpoint from the stable phase can be given its own short decay and becomes a finished model. The lab plots all three from `lr_at`.

- **Zero the gradients (130–131).** PyTorch *accumulates* into `.grad`, so it must be cleared before each step — except that accumulation is exactly what the next block wants.

- **Forward and backward with accumulation (133–140).** For each of `grad_accum` micro-batches: fetch, forward under `torch.autocast` (bf16 on hardware that supports it — off by default on CPU), divide the loss by `grad_accum` so the accumulated gradient is a *mean* over the whole effective batch, and call `backward()`. Loss is reported as the mean over micro-batches too. `tokens_seen` counts every token for the throughput number.

- **Clip (143).** `clip_grad_norm_` computes the global L2 norm of all gradients together and, if it exceeds `grad_clip` (1.0), scales every gradient down so the norm equals 1.0. It returns the *pre-clipping* norm, which is the single most useful health signal in the log: a slowly rising gradient norm precedes most loss spikes by hundreds of steps.

- **Step (144–145).** Each optimizer updates its parameters. For AdamW this is where **decoupled weight decay** happens: the parameter is first shrunk toward zero by `lr × weight_decay × w`, *then* the Adam update is added. The "decoupled" is the whole point (Loshchilov & Hutter 2019): the older way added `weight_decay × w` to the gradient, which Adam then rescaled by its per-parameter normaliser, so the effective decay varied wildly across parameters. `build_optimizer` (`optim.py:107–113`) applies decay only to matrices (`ndim ≥ 2`): decaying a norm gain or a bias toward zero has no regularising meaning.

- **Log (148–158).** Every `log_every` steps: loss, LR multiplier, gradient norm and tokens per second, both printed and appended to `History`.

- **Validate and checkpoint (161–168).** Every `eval_every` steps and at the end: estimate the validation loss, print its perplexity (*e*<sup>loss</sup>, "the effective number of equally likely next tokens"), and save a checkpoint tagged with `step + 1` — the number of *completed* steps, so that resuming continues at the right one.

### AdamW vs Muon

**Adam** keeps two running averages per parameter: the mean of recent gradients *m* (momentum) and the mean of recent *squared* gradients *v*, and updates each parameter by *m*/(√*v* + ε). Read this as: every single weight moves by roughly the same amount per step, whatever the size of its gradient — Adam normalises *per element*. AdamW is Adam with decoupled weight decay. It has been the default LLM optimizer since GPT-2, and it works; the question since 2024 is whether something reaches the same loss in fewer tokens.

![Muon: momentum matrix → Newton–Schulz → orthogonal update](../figures/10_muon.svg)

**Muon** (Keller Jordan, 2024: MomentUm Orthogonalised by Newton–Schulz) normalises *per matrix* instead. Follow the figure left to right. A weight matrix's momentum-averaged gradient *M* is a linear map; picture what it does to the unit circle (panel 1): it squashes it into an ellipse whose axes are the matrix's **singular values** — the stretch factors along its principal directions. A gradient matrix typically has a few large singular values and many tiny ones, so an Adam or SGD step moves the weights mostly along a handful of dominant directions and barely at all along the rest. Muon replaces *M* by the nearest **orthogonal matrix** *O* = *UV*<sup>T</sup> (panel 3: every singular value set to 1, so the map only rotates), then steps along *O*. Read this as: push equally hard in *every* direction the gradient identified, not just the loud ones.

Computing *UV*<sup>T</sup> exactly needs an SVD, which is slow on GPUs. Panel 2 is the trick: the **Newton–Schulz iteration** repeatedly applies a fixed quintic polynomial *X* ← *aX* + *b*(*XX*<sup>T</sup>)*X* + *c*(*XX*<sup>T</sup>)<sup>2</sup>*X* with coefficients (3.4445, −4.7750, 2.0315) chosen so that every singular value in (0, 1] is pushed toward 1; five iterations of nothing but matrix multiplies get them within a factor of ~1.5 of 1, which is all an optimizer needs (`optim.py:21–38`). The step then scales by √(rows/cols) so that the update's RMS matches Adam's, letting one learning rate (0.02) serve every matrix shape, and applies decoupled weight decay (`optim.py:75–80`; the scaling is from Moonshot's "Muon is Scalable" recipe).

**Why matrices only.** "Orthogonal" is a property of a linear map from one vector space to another, which is what a `Linear` weight is. The embedding table is *not* a map — its rows are lookup entries, each one an independent vector — and orthogonalising it would force unrelated tokens' embeddings to stay mutually independent. RMSNorm gains and biases are 1-D. So `split_params` (`optim.py:85–101`) sends the 21 projection matrices of TinyLM-nano to Muon and the embedding (tied with the output head) plus the 7 norm gains to AdamW.

🆕 **Adoption in 2026.** Muon went from a speedrun trick (it set the modded-nanoGPT records through 2025) to the pretraining optimizer of Kimi K2 (1T parameters, 15.5T tokens, Moonshot 2025 — the "Muon is Scalable" paper, which reports roughly 2× the compute efficiency of AdamW at matched loss), GLM-4.5 through GLM-5, and DeepSeek-V4 (https://arxiv.org/abs/2606.19348). A July 2026 study, "SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales" (https://arxiv.org/abs/2607.20548), reports that Muon matches or beats AdamW on a 30B/3B-active hybrid Mamba–attention MoE trained for 3T tokens and on a 72B/8B LatentMoE with multi-token prediction, with the largest gains on coding and commonsense tasks and gains that *grow with batch size* — consistent with Muon's better use of each batch. GLM-5 (744B) is reported to use a "Muon Split" that orthogonalises MLA up-projections per head rather than as one matrix. The 2× figure is the number people cite; treat it as "reported at these scales by these labs" rather than a law, and expect the lab's tiny comparison to show a smaller, noisier gap.

### Mixed precision

A 32-bit float (**fp32**) has 8 exponent bits (range) and 23 mantissa bits (precision). **bf16** keeps the 8 exponent bits and cuts the mantissa to 7: the same range as fp32, ~3 significant digits, half the memory and — on tensor cores — 2–4× the matmul throughput. Since ~2021 every large run computes forward and backward in bf16 while keeping a **master copy** of the weights in fp32 for the optimizer update (small updates would otherwise round to zero); `torch.autocast` does the casting per operation and `train.py:135` turns it on with `dtype="bfloat16"`. On CPU the labs run fp32 because CPU bf16 matmuls are not faster.

🆕 **FP8 is the 2026 default at scale.** DeepSeek-V3 (Dec 2024) was the first published trillion-token run with FP8 matmuls (8 bits: e4m3 for activations/weights, e5m2 for gradients, with per-block scaling factors to recover range), and by 2026 the major open-weight runs train the linear layers in FP8 with bf16 accumulation. **FP4** is the research frontier: NVIDIA's NVFP4 recipe (4-bit with per-16-element fp8 scales) is reported validated on multi-trillion-token runs up to ~120B parameters, MXFP4 pretraining on native FP4 hardware is studied in https://arxiv.org/abs/2605.09825 (Hadamard rotations of the activations are what keeps it stable), and gpt-oss (2025) shipped its weights in MXFP4. Each halving of precision roughly doubles matmul throughput and halves memory; each also needs a new stabilisation trick, which is what the papers are about.

### Loss spikes and what to do about them

A **loss spike** is a sudden jump in training loss — sometimes a few tenths of a nat that recovers in a hundred steps, sometimes a divergence to ln *V* (the model has forgotten everything) that never recovers. Every large run sees them; the PaLM paper counted ~20 in one 540B run. Known causes: a learning rate that is too high for the current phase (most common; the spike often arrives right after warmup ends or when the batch composition shifts), attention logits growing without bound so the softmax saturates, fp16 overflow (a reason bf16 replaced it), and occasionally a single pathological batch. The remedies, roughly in the order people try them:

1. **Lower the peak LR or lengthen warmup** — costs a little final loss, fixes most spikes.
2. **Gradient clipping** (already in the loop) — bounds the damage of any one step.
3. **Skip the batch** — PaLM's recipe: restart from a checkpoint ~100 steps before the spike and skip the next 200–500 batches; the same data in a different order rarely re-spikes.
4. **z-loss** (PaLM) — add 10<sup>−4</sup> · (log *Z*)<sup>2</sup> to the loss, where *Z* is the softmax normaliser; it keeps the logits from drifting to huge values. Read this as: a tiny penalty for the output distribution being unnormalised.
5. **QK-norm** — RMSNorm the queries and keys before the dot product so attention logits stay bounded (standard in 2025–2026 architectures).
6. A smaller LR for embeddings, or a tighter initialisation, when the spike is traced to the embedding table.

### What to watch

The log line `step 250 | loss 2.3151 | lr×0.876 | grad_norm 0.61 | 8,200 tok/s` is a dashboard. **Loss** should fall fast, then slowly; compare train and val every eval. **Gradient norm** should settle to a roughly constant value after warmup; a steady climb is the early warning for a spike. **LR** should follow the schedule you intended (a wrong `steps` makes cosine decay at the wrong time). **Tokens per second** should be flat; a drop means a straggler, a data-loader stall, or thermal throttling. **MFU** = tok/s × FLOPs per token ÷ hardware peak is the number to report — the lab computes it against a placeholder CPU peak of 100 GFLOP/s and against the 990 TFLOP/s of an H100 in the discussion. Add to those, at scale: memory headroom, the fraction of time in communication (Chapter 11), and expert load balance (Chapter 12).

## The idea in code

Imports for every snippet:

```python
import torch
from llm.config import preset
from llm.model import TinyLM
from llm.optim import lr_at, newton_schulz_orthogonalize, split_params, build_optimizer
from llm.train import TrainConfig, train, get_batch, load_checkpoint
from llm.pipeline import get_tokenizer, get_tokens, run_path
```

**A batch.** Windows from the packed stream; the target is the input shifted by one:

```python
tok = get_tokenizer()
train_tokens, val_tokens = get_tokens(tok)              # (357931,), (18838,) int64
x, y = get_batch(train_tokens, batch_size=4, seq_len=8, generator=torch.Generator().manual_seed(0))
print(x.shape, y.shape, torch.equal(x[:, 1:], y[:, :-1]))   # torch.Size([4, 8]) torch.Size([4, 8]) True
```

**Schedules.** `lr_at` returns the multiplier for a step; warmup is linear from 0:

```python
for step in (0, 5, 9, 50, 90, 99):                     # 100-step run, 10 warmup steps
    print(step, round(lr_at(step, 100, 1.0, warmup_steps=10, kind="cosine"), 3),
          round(lr_at(step, 100, 1.0, warmup_steps=10, kind="wsd"), 3))
# 0 0.1 0.1 | 5 0.6 0.6 | 9 1.0 1.0 | 50 0.628 1.0 | 90 0.127 0.5 | 99 0.1 0.05
```

**Newton–Schulz drives singular values toward 1.** Five iterations of matmuls, no SVD:

```python
G = torch.randn(64, 32)
print(torch.linalg.svdvals(G)[[0, -1]])                   # tensor([13.0050,  2.6853])  largest, smallest
O = newton_schulz_orthogonalize(G, steps=5)
print(torch.linalg.svdvals(O)[[0, -1]])                   # tensor([1.0722, 0.6819])   all within ~1.5× of 1
```

**Which parameters get Muon.** Matrices that are linear maps; everything else to AdamW:

```python
model = TinyLM(preset("nano", vocab_size=tok.vocab_size))
muon_p, adam_p = split_params(model)
print(len(muon_p), sum(p.numel() for p in muon_p), len(adam_p), sum(p.numel() for p in adam_p))
# 21 294912 8 84288   -> 21 projection matrices to Muon; embedding + 7 RMSNorm gains to AdamW
print([type(o).__name__ for o in build_optimizer(model, "muon", lr=1e-3, muon_lr=0.02)])   # ['Muon', 'AdamW']
```

**Twenty steps, with a checkpoint.** The whole loop is one call; `ckpt_path` saves on every eval:

```python
cfg = TrainConfig(steps=20, batch_size=8, seq_len=32, optimizer="adamw", lr=2e-3, warmup_steps=5,
                  log_every=5, eval_every=10, ckpt_path=run_path("snip10_ckpt.pt"))
hist = train(model, train_tokens, val_tokens, cfg)
# step 0 | loss 6.7776 | lr×0.200 | grad_norm 1.70 | ...   step 19 | loss 4.9407 | lr×0.110 | grad_norm 2.77
print(hist.step, [round(l, 2) for l in hist.train_loss])   # [0, 5, 10, 15, 19] [6.78, 6.1, 5.45, 4.97, 4.94]
```

**Resume.** A fresh model plus `resume_from` continues at the saved step with the saved history:

```python
m2 = TinyLM(preset("nano", vocab_size=tok.vocab_size))   # random weights...
step, h = load_checkpoint(run_path("snip10_ckpt.pt"), m2)  # ...overwritten from the checkpoint
print(step, len(h.step))                                    # 20 5
cfg2 = TrainConfig(**{**cfg.to_dict(), "steps": 30})
h2 = train(m2, train_tokens, val_tokens, cfg2, resume_from=run_path("snip10_ckpt.pt"), verbose=False)
print(h2.step)                                              # [0, 5, 10, 15, 19, 20, 25, 29]
```

## Worked example 🧪

WORKED_EXAMPLE_10

🎛️ In `interactive/10_training_dynamics.html`, a simulated run shows loss, gradient norm and LR side by side. Drag the peak LR up until the loss spikes, then try each remedy — longer warmup, clipping, z-loss, skipping the bad batch — and watch which ones recover. Switch the schedule between cosine and WSD and note where the final loss drop happens. The challenge is to find the highest LR that finishes without a spike using at most two remedies.

## Try it yourself ✍️

1. **Provoke a spike.** Train the nano model with `lr=3e-2` (30× the default) and `warmup_steps=0`. Plot loss and gradient norm. Then fix it with warmup alone, then with clipping at 0.5 alone. Which is more effective here?
2. **Gradient accumulation.** Run `batch_size=8, grad_accum=4` and `batch_size=32, grad_accum=1` with the same seed. Are the loss curves identical? (They should be close but not bit-identical — explain why from `get_batch`.)
3. **Muon's learning rate.** Sweep `muon_lr` in {0.005, 0.01, 0.02, 0.04, 0.08}. Muon is known to tolerate a wide range; where does it break?
4. **Give Muon the embedding.** Modify `split_params` in a copy so the embedding goes to Muon. What happens to the loss? This is why the exclusion exists.
5. **WSD in practice.** Train 300 steps with `schedule="wsd"` and a checkpoint at step 200 (`eval_every=100`). Then resume from that checkpoint with `steps=250, schedule="wsd"` — a 50-step decay branch. Compare its final loss with the 300-step run's. That is how 2025–2026 labs produce intermediate models.
6. **Autocast on CPU.** Set `dtype="bfloat16"` and compare tokens/second and final loss. On most CPUs it is slower; on Apple silicon or a recent Xeon it may not be.

## Check yourself ✅

<details><summary>1. Why divide the loss by grad_accum before calling backward()?</summary>

Because `.grad` accumulates a *sum* over the micro-batches. Dividing each micro-batch's loss by `grad_accum` makes the accumulated gradient the *mean* over the whole effective batch, which is what a single big batch would have produced, so the learning rate keeps its meaning.
</details>

<details><summary>2. A checkpoint saves the optimizer state. What goes wrong if you resume with only the weights?</summary>

AdamW's running moments *m* and *v* restart at zero, so the first steps after resuming behave like step 0 without warmup: the normaliser √*v* is tiny and the updates are far too large, giving a visible loss bump. Muon's momentum buffer resets similarly. Saving the state makes the resumed run continue smoothly (the lab checks this).
</details>

<details><summary>3. In one sentence each: what does Adam normalise, and what does Muon normalise?</summary>

Adam normalises each *element* of the update so every weight moves by about the same amount. Muon normalises each *matrix* of the update so every singular direction moves by the same amount — the update is orthogonal.
</details>

<details><summary>4. Why is WSD preferred over cosine for a run whose length might change?</summary>

Cosine's shape depends on the planned total steps, so extending the run or taking a model from the middle means the LR was decayed at the wrong time. WSD holds the LR constant until the end, so any stable-phase checkpoint can be given its own short decay to produce a finished model, and the run can be extended for free.
</details>

<details><summary>5. Your gradient norm has been rising slowly for 500 steps and the loss is fine. What do you do?</summary>

Treat it as an early warning of a spike. Check that the LR schedule is what you intended, consider lowering the peak LR or adding z-loss/QK-norm, and make sure checkpoints are frequent enough to roll back cheaply. Clipping is already bounding each step; the rising norm says the model's logits or activations are growing.
</details>

## Key takeaways

- One step = LR for this step → batch → forward → loss → backward (× accumulation) → clip → optimizer step → log; `train.py` does this in ~50 lines.
- Warmup protects the first steps; cosine and WSD decay the end. WSD lets you branch a finished model from any stable-phase checkpoint.
- AdamW decouples weight decay from the gradient and decays only matrices. Muon orthogonalises each matrix's momentum with five Newton–Schulz iterations and steps equally in every direction; embeddings and norms stay with AdamW.
- 🆕 Muon is the pretraining optimizer of Kimi K2, GLM-5 and DeepSeek-V4, with ~2× compute efficiency reported; FP8 is the default precision at scale and FP4 is being validated.
- Loss spikes come from LR, unbounded logits and bad batches; remedies are warmup, clipping, skipping, z-loss and QK-norm.
- Watch loss, gradient norm, LR, tokens/second and MFU. A checkpoint must include the optimizer state.

## Going deeper

- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW), 2019. https://arxiv.org/abs/1711.05101
- Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways" (§5.1: loss spikes, the skip-batches recipe, z-loss), 2022. https://arxiv.org/abs/2204.02311
- Jordan et al., "Muon: An optimizer for hidden layers in neural networks", 2024. https://kellerjordan.github.io/posts/muon/
- Hägele et al., "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations" (WSD schedules), 2024. https://arxiv.org/abs/2405.18392
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (§3.3: FP8 training at scale), Dec 2024. https://arxiv.org/abs/2412.19437
- 🆕 Liu et al., "Muon is Scalable for LLM Training" (Moonshot; the RMS-matching scale, weight decay, Kimi results), 2025. https://arxiv.org/abs/2502.16982
- 🆕 "SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales", July 2026. https://arxiv.org/abs/2607.20548
- 🆕 "MXFP4 pretraining on native FP4 hardware" (Hadamard rotations for stability), May 2026. https://arxiv.org/abs/2605.09825
- 🆕 PyTorch blog, "Using the Muon optimizer with DeepSpeed", 2026. https://pytorch.org/blog/using-muon-optimizer-with-deepspeed/
- Karpathy, modded-nanoGPT — the speedrun where Muon was born, records through 2026. https://github.com/KellerJordan/modded-nanogpt

← [Chapter 9](09-scaling-laws.md) · [Course home](../README.md) · [Chapter 11](11-distributed-training.md) →
