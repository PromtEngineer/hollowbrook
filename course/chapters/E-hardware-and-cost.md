# Appendix E: Hardware and cost guide (laptop → 8×H100 → cluster)

This appendix turns the course's arithmetic (Chapter 9: `C ≈ 6·N·D`) into money and time. Prices
are typical cloud list prices in 2026 and move constantly; treat every number as ±50 %.

## E.1 The one formula

```
time (s)  =  6 · N · D  /  (GPUs · peak FLOP/s · MFU)
cost      =  time (h) · GPUs · $/GPU-hour
```

Read this as: total compute is six times parameters times tokens; you deliver it at some fraction
(**MFU**, model FLOPs utilisation, typically 30–50 % for well-tuned dense runs, lower for MoE and
long context) of the hardware's peak.

## E.2 Reference hardware

| Device | Peak bf16 dense (approx.) | Memory | Bandwidth | Typical cloud price |
|---|---|---|---|---|
| Laptop CPU (4–8 cores) | 0.05–0.3 TFLOP/s | 16–64 GB | 50–100 GB/s | $0 |
| Apple M-series (GPU) | 5–30 TFLOP/s | 16–128 GB unified | 100–800 GB/s | $0 |
| RTX 4090 / 5090 | ~165 / ~200+ TFLOP/s | 24 / 32 GB | 1.0 / 1.8 TB/s | $0.5–1 /h |
| NVIDIA H100 (SXM) | ~990 TFLOP/s | 80 GB HBM3 | 3.35 TB/s | $2–3 /h |
| NVIDIA B200 | ~2.2 PFLOP/s (bf16), 2× that in FP8, 4× in FP4 (reported) | 180–192 GB | ~8 TB/s | $4–6 /h |
| AMD MI355X | comparable class to B200; native MXFP4 (reported) | 288 GB | ~8 TB/s | varies |

FP8 roughly doubles and FP4 roughly quadruples peak throughput over bf16 on hardware that
supports them natively, which is why FP8 became the 2026 default for large pretraining runs and
FP4/NVFP4/MXFP4 are the research frontier (Chapter 10).

## E.3 What each budget buys (Chinchilla-optimal, `D = 20·N`, 40 % MFU)

| Budget | Hardware | FLOPs delivered | Model (N) | Tokens (D) | Example |
|---|---|---|---|---|---|
| free, 10 min | laptop CPU | ~5·10¹² | 2.5 M | 2.5 M (TinyLM, over-trained) | this course |
| $2, 1 h | 1× RTX 4090 | ~2·10¹⁷ | ~40 M | ~800 M | GPT-2-small class in a few hours |
| $100, 4 h | 8× H100 | ~4.5·10¹⁸ | ~200 M | ~4 B | nanochat "$100 ChatGPT" (Oct 2025) |
| $10 k | 8× H100, 2 weeks | ~1.4·10²⁰ | ~1 B | ~20 B | a serious small research model |
| $1 M | 256× H100, 2 weeks | ~4·10²¹ | ~6 B | ~120 B | a 2023-class 7B (Chinchilla) |
| $10 M+ | 1024+ GPUs, weeks | 10²²–10²³ | 10–100 B (dense) or ~1 T (MoE) | trillions | 2025–2026 open-weight frontier |

Frontier labs deliberately **over-train**: Llama-3 8B saw 15 T tokens (~1,900 tokens per parameter,
not 20) because a smaller model that is cheaper to *serve* is worth extra training compute.
The 2026 open-weight models in the research notes (DeepSeek-V4 ~1.6 T total parameters,
Kimi K3 ~2.8 T, GLM-5.2 744 B) are MoE models: total parameters are large, *active* parameters
per token are 5–15 % of that, and the training cost formula uses the active count.

## E.4 Memory: will it fit?

Training memory per parameter with Adam in mixed precision ≈ **16 bytes** (2 weights bf16 +
2 grads + 4 master weights fp32 + 8 Adam states), plus activations (grows with batch × sequence).

- 7 B model, full fine-tune: 112 GB of states → does not fit one 80 GB GPU → FSDP over 2+ GPUs,
  or LoRA (Chapter 15) which only trains a few percent of the parameters.
- 70 B model: 1.1 TB of states → 16+ GPUs with sharding (Chapter 11).
- Inference is cheaper: weights only, 2 bytes/param in bf16 (140 GB for 70 B), 0.5 bytes in INT4,
  plus the KV cache (Chapter 7: `2 · layers · kv_heads · head_dim · bytes` per token).

## E.5 Cost of serving

Per generated token, a dense model does ~2·N FLOPs but is usually **memory-bandwidth bound**
(Chapter 7): every decode step reads all the weights. A 70 B bf16 model on one H100 is limited to
roughly `3.35 TB/s / 140 GB ≈ 24 tokens/s` for a single sequence; batching many users amortises the
weight reads, which is why serving systems batch aggressively. MoE, quantisation, speculative decoding
and KV-cache compression (DeepSeek-V4's CSA/HCA) all attack this bottleneck.

Rule of thumb for 2026 API prices: small models a few cents per million tokens, frontier models
a few dollars per million output tokens; "thinking" tokens are billed as output tokens, which is why
reasoning models can be an order of magnitude more expensive per question.

## E.6 Practical advice by budget

- **Laptop only.** Do the whole course. For anything bigger, rent a single GPU by the hour.
- **One consumer GPU.** Pretrain a 30–125 M model on a few billion tokens (FineWeb-Edu sample),
  fine-tune 1–8 B models with LoRA, run GRPO on 0.5–1.5 B models with verifiable rewards.
- **8× H100 for a day.** nanochat-scale pretraining from scratch, full-parameter SFT/DPO/GRPO on
  7–8 B models, on-policy distillation from a 32 B teacher.
- **Cluster.** Now the problems are Chapter 11's: parallelism strategy, data loading, checkpoint
  I/O, fault tolerance and, above all, evaluation infrastructure.
