# Appendix C: Glossary

Every bold term defined in the course, with the chapter(s) that define it. 461 entries.

## "

- **"aha moment"** — a reported training-log excerpt of R1-Zero pausing and restarting a derivation; evidence of more reflective phrasing, not proof of new capability. *(Ch. 19)*

## 1

- **16 bytes per parameter** — the training-state cost of a parameter under AdamW in mixed precision: 2 (bf16 weight) + 2 (bf16 gradient) + 4 (fp32 master weight) + 8 (Adam's two fp32 moments). *(Ch. 11)*
- **1F1B** — the "one forward, one backward" pipeline schedule that caps in-flight activations at the number of stages. *(Ch. 11)*

## 6

- **6·N·D rule** — training compute ≈ 6 FLOPs per parameter per token: 2 forward, 4 backward. *(Ch. 9)*

## A

- **action** — what the agent chooses; for a language model, the next token. *(Ch. 18)*
- **activation checkpointing** — discarding intermediate activations during the forward pass and recomputing them in the backward pass, trading ~30% compute for memory. *(Ch. 11)*
- **activation function** — a fixed non-linear curve (tanh, relu, silu) applied to a neuron's sum so that stacked layers can represent bends. *(Ch. 4)*
- **active parameters** — the parameters that participate in computing one token (the visited experts plus everything dense); governs per-token compute. *(Ch. 12)*
- **Adam** — an optimizer that normalises each parameter's step by a running estimate of its gradient's magnitude. *(Ch. 4, Ch. 10)*
- **AdamW** — Adam with decoupled weight decay; the default LLM optimizer since 2019. *(Ch. 4)*
- **add-k smoothing** — smoothing by pretending every possible (context, next) pair was seen k extra times. *(Ch. 1)*
- **adjusted base frequency (ABF)** — raising RoPE's θ so all channel pairs rotate more slowly and longer positions map onto angle ranges seen in training; Llama 3 used θ = 500k. *(Ch. 13)*
- **advantage** — the reward minus a baseline: how much better than expected an answer was. *(Ch. 18)*
- **advantage (group-relative)** — (r_i − mean(r)) / (std(r) + ε) over the G answers to one prompt; Dr. GRPO omits the division by std. *(Ch. 19)*
- **agent** — a model used in a loop: it writes a message, the environment (tools, tests, a browser) replies, it writes again, and the whole interaction is judged. *(Ch. 21)*
- **agent harness** — the program around a model that supplies tools, manages context, enforces permissions and loops until a task is done. *(Ch. 0)*
- **Agent2Agent protocol (A2A)** — the protocol (Google, April 2025; v1.0 reported April 2026) by which one agent sends tasks to another agent and exchanges messages and artefacts with it. *(Ch. 26, Ch. 28)*
- **agentic eval** — running whole trajectories in a sandbox and checking the outcome; the most relevant and the slowest, most expensive, highest-variance measurement. *(Ch. 23)*
- **agentic model** — a model trained on multi-turn interactions with tools and environments so that it can act, observe a result, and act again. *(Ch. 14)*
- **agentic RL** — RL over multi-turn interactions with tools and environments. *(Ch. 0)*
- **agentic safety** — whether a model acting with tools stays inside its instructions when instructions and task conflict; defended by weights and by the harness together. *(Ch. 22)*
- **alignment** — operationally, the property that a model's behaviour matches a written specification as measured by held-out evals; the "alignment tax" is the helpfulness lost (over-refusal) when safety data is over-weighted. *(Ch. 14, Ch. 22)*
- **alignment tax** — the measured capability cost of alignment training: task accuracy lost while principle adherence rose. *(Ch. 22)*
- **all-gather** — a collective in which every device receives every other device's shard, reassembling a full tensor. *(Ch. 11)*
- **all-reduce** — a collective operation after which every participant holds the sum (or mean) of everyone's tensor. *(Ch. 11)*
- **all-to-all** — a collective in which every device sends a different chunk to every other device. *(Ch. 11)*
- **allowed_special** — the encode flag that decides whether special-token strings in the input are recognised (trusted template) or spelled out as text (untrusted input). *(Ch. 2)*
- **anneal** — (also decay phase or cooldown) the final stretch of pretraining in which the learning rate decays to zero while the data mix shifts to higher-quality, re-weighted sources. *(Ch. 13)*
- **annotation guidelines** — the written document that tells a labeler what wins and why, with worked examples of every known trap; revised until inter-annotator agreement is acceptable. *(Ch. 16)*
- **arithmetic intensity** — FLOPs performed per byte moved from memory; for batch-`B` decode it is about `B`. *(Ch. 7)*
- **asynchronous rollouts** — generating trajectories, running environments and training in overlapping pipelines rather than in lock-step, so the trainer never waits for the slowest environment. *(Ch. 21)*
- **attention** — a soft, learned lookup: each token scores its query against every earlier token's key and takes a weighted average of their values. *(Ch. 5)*
- **attention checks** — items designed so that a labeler who is not reading gets them wrong. *(Ch. 16)*
- **attention sink** — a head that parks most of its weight on a fixed position (usually the first token) when it has nothing useful to look up, because softmax weights must sum to 1. *(Ch. 5)*
- **autoregressive** — generating one token at a time, left to right, each conditioned on the previous ones. *(Ch. 12)*
- **auxiliary-loss-free balancing** — DeepSeek-V3's scheme: a per-expert bias added to routing scores and adjusted by load, so nothing competes with the main loss. *(Ch. 12)*

## B

- **backpropagation** — the algorithm that applies the chain rule over a computation graph to obtain every parameter's gradient in one backward walk. *(Ch. 4)*
- **backward pass** — computing the gradient of the loss with respect to every parameter by walking the computation graph in reverse. *(Ch. 4)*
- **band** — a contiguous group of r rows of a MinHash signature; a pair is a candidate if any band matches exactly, P = 1 − (1 − J^r)^b. *(Ch. 8)*
- **base checkpoint** — the annealed, context-extended model released as "the base model", from which all post-training and distillation branch. *(Ch. 13)*
- **base model** — the checkpoint after pretraining: continues any text but has no notion of a user or a task. *(Ch. 0, Ch. 14)*
- **baseline** — a value subtracted from the reward before it multiplies the score function; it leaves the expected gradient unchanged and lowers its variance. *(Ch. 18)*
- **best-of-N** — sampling N answers and keeping the one a reward function scores highest; the simplest way to optimise against a reward, and the setting in which reward overoptimisation is measured. *(Ch. 17)*
- **bf16** — bfloat16, a 16-bit floating-point format with float32's exponent range and 8 bits of mantissa; the default for training and storing models. *(Ch. 6)*
- **bias** — a learned constant added to a neuron's weighted sum. *(Ch. 4)*
- **bigram** — an n-gram model with one token of context (n = 2). *(Ch. 1)*
- **bits** — the unit of −log₂ P; a probability-½ event costs 1 bit. *(Ch. 1)*
- **boilerplate** — navigation menus, footers, cookie notices and other text repeated on every page of a site. *(Ch. 8)*
- **bootstrap confidence interval** — resample the per-item scores with replacement many times, take the mean each time, and report the central 95% of those means; half-width ≈ 1/√n near accuracy 0.5. *(Ch. 23)*
- **Bradley–Terry model** — a model of pairwise comparisons in which each item has a hidden score and P(a beats b) = σ(r_a − r_b), the sigmoid of the score difference. *(Ch. 17)*
- **branch point** — a checkpoint on the WSD stable plateau from which an anneal can be started, making mid-training experiments cheap to repeat. *(Ch. 13)*
- **Byte Latent Transformer (BLT)** — a tokenizer-free model that reads bytes and groups them into variable-size patches based on next-byte predictability. *(Ch. 2)*
- **byte-pair encoding (BPE)** — a compression algorithm that builds a subword vocabulary by repeatedly merging the most frequent adjacent pair of symbols. *(Ch. 2)*

## C

- **catastrophic forgetting** — the loss of earlier capabilities when a network is trained further on a narrower or different distribution. *(Ch. 13, Ch. 15)*
- **causal mask** — setting scores for future positions to −∞ so that token i can only attend to tokens 0..i. *(Ch. 5)*
- **chain rule** — the derivative of a composition is the product of the derivatives along the path. *(Ch. 4)*
- **chain rule of probability** — P(w₁…w_T) = ∏ P(wᵢ | w₁…wᵢ₋₁): a sequence's probability is the product of each token's probability given the tokens before it. *(Ch. 1)*
- **character-level tokens** — one token per character: tiny vocabulary, long sequences. *(Ch. 2)*
- **chat template** — the fixed convention for writing a multi-turn conversation as one token sequence, using special tokens for roles and turn boundaries. *(Ch. 14)*
- **checkpoint** — a saved snapshot of model weights, optimizer state, step count, config and history, sufficient to resume training. *(Ch. 10, Ch. 11)*
- **checkpointing (harness)** — persisting enough state (the plan and progress files) that a fresh process can continue the job. *(Ch. 27)*
- **Chinchilla** — the 2022 DeepMind study finding that compute-optimal training scales N and D together, about 20 tokens per parameter. *(Ch. 9)*
- **CISPO** — a 2025 variant that clips the importance weight itself rather than zeroing token gradients, so rare tokens keep contributing. *(Ch. 19)*
- **clean git state per session** — the practice of starting each session from a committed tree and committing its work, so a bad session can be reverted. *(Ch. 27)*
- **client** — in MCP, the object inside the host that owns one connection to one server and speaks JSON-RPC to it. *(Ch. 26)*
- **clip-higher** — DAPO's asymmetric clipping with ε_high > ε_low, giving low-probability tokens of good answers more room to grow. *(Ch. 19)*
- **Cohen's kappa** — inter-annotator agreement corrected for chance: κ = (p_o − p_e)/(1 − p_e), where p_o is observed agreement and p_e the agreement expected from the two annotators' label frequencies. *(Ch. 16)*
- **cold start** — a small SFT stage on a few thousand long-reasoning examples, run before RL so that RL starts from a readable format (DeepSeek-R1's term). *(Ch. 14, Ch. 15)*
- **Common Crawl** — a non-profit monthly crawl of the public web, published as WARC files; the raw source of nearly all open pretraining datasets. *(Ch. 8)*
- **communication versus compute** — the ratio that decides MFU: per step, compute scales with batch × parameters while gradient traffic scales only with parameters. *(Ch. 11)*
- **compression ratio** — bytes of text per token; higher means each token covers more text, and it depends on the corpus. *(Ch. 2)*
- **compute-bound** — limited by the chip's arithmetic rate; prefill and training are the canonical cases. *(Ch. 7)*
- **compute-optimal frontier** — the line through the minima of the iso-FLOP curves: the best (N, D) for every budget. *(Ch. 9)*
- **constitution** — the short written document of principles that drives CAI; in 2026 published as a model spec by several labs. *(Ch. 22)*
- **Constitutional AI (CAI)** — Anthropic's 2022 recipe: write principles as a document, have the model critique and revise its own drafts against them (SFT data), then have an AI judge rank samples against them (preference data). *(Ch. 22)*
- **contamination** — evaluation text present in training data, which inflates benchmark scores. *(Ch. 8, Ch. 23)*
- **context** — the tokens that come before the position being predicted. *(Ch. 1)*
- **context parallelism (CP)** — splitting one long sequence across devices and passing key/value blocks around a ring (Ring Attention) so attention can span the whole sequence. *(Ch. 11)*
- **context rot** — the degradation of an agent's judgement as its window fills with stale tool output, even when the relevant facts are still present. *(Ch. 27)*
- **contextual bandit** — an RL problem with one decision per episode and a context (the prompt); the structure of single-answer LLM RL. *(Ch. 18)*
- **continuous batching** — scheduling at the granularity of one decode step so a finished sequence's slot is refilled immediately. *(Ch. 7)*
- **cosine decay** — a learning-rate schedule that follows half a cosine from peak down to a minimum at the last step. *(Ch. 10)*
- **cosine similarity** — the dot product divided by both vectors' norms; the cosine of the angle between them, from −1 to +1. *(Ch. 3)*
- **cost control** — the limits on turns, sessions, tool-result size, context budget and estimated tokens that bound what a run can spend. *(Ch. 27)*
- **credit assignment** — deciding which turn (or token) of a trajectory deserves how much of the trajectory's reward; the library gives every generated token the same advantage. *(Ch. 21)*
- **critique and revision** — stage 1 of CAI: the model is shown a principle, critiques its own draft against it and rewrites; revisions become SFT data. *(Ch. 22)*
- **cross-entropy** — the average surprisal per token of a model on a text; the training loss of every neural language model. *(Ch. 1, Ch. 4)*
- **CSA / HCA** — DeepSeek-V4's Compressed Sparse Attention (compress m tokens per KV entry, then top-k selection plus a window) and Heavily Compressed Attention (stronger compression, dense attention), interleaved across layers. *(Ch. 12)*
- **curation** — the data pipeline (filtering, deduplication, scrubbing, mixing) that turns raw crawled text into a pretraining set. *(Ch. 0)*
- **curriculum** — ordering the data mix over training, e.g. saving the highest-quality data for the final annealing phase. *(Ch. 8)*

## D

- **DAPO** — a 2025 GRPO recipe with clip-higher, dynamic sampling, token-level loss, overlong shaping and no KL term. *(Ch. 19)*
- **data curation** — the pipeline that turns raw crawled text into a training set: extract, filter, deduplicate, classify, scrub, decontaminate, mix. *(Ch. 8)*
- **data mixing** — choosing per-source sampling weights (web, code, maths, books, synthetic) for the final training stream. *(Ch. 8)*
- **data parallelism (DP)** — every device holds a full model copy and trains on a different slice of each batch; gradients are averaged every step. *(Ch. 11)*
- **DCLM** — DataComp-LM: a 2024 benchmark and dataset whose best recipe keeps the top 10% of Common Crawl by a fastText classifier trained on instruction-style text. *(Ch. 8)*
- **debate** — several agents answer, then see each other's answers and revise; demonstrates that agreement is not correctness. *(Ch. 28)*
- **decode** — the second phase of generation: one forward pass per new token, reading the KV cache; memory-bound at small batch. *(Ch. 7)*
- **decoder-only** — a Transformer that uses causal attention throughout and is trained to predict the next token. *(Ch. 5)*
- **decontamination** — removing training documents that overlap with evaluation sets, usually by n-gram overlap. *(Ch. 8)*
- **decoupled weight decay** — shrinking weights by lr × wd × w separately from the Adam update, rather than adding wd × w to the gradient (AdamW). *(Ch. 10)*
- **deduplication** — removing documents that are copies or near-copies of others. *(Ch. 8)*
- **DeepSeekMoE** — the MoE design with many fine-grained routed experts plus shared experts, used by DeepSeek-V2/V3/V4. *(Ch. 12)*
- **derivative** — the slope of a function at a point: how much the output changes per tiny change of the input. *(Ch. 4)*
- **diffusion language model** — a model that generates by iteratively un-masking or denoising a whole sequence in parallel rather than left to right. *(Ch. 12)*
- **digit chunking** — the GPT-4 pre-tokenizer rule that numbers are split into groups of at most three digits. *(Ch. 2)*
- **dimension** — the number of coordinates in a vector; d_model for a model's embeddings. *(Ch. 3)*
- **Direct Preference Optimization (DPO)** — a method that turns a preference pair directly into a loss on the policy's own log-probabilities, using a frozen reference model instead of a separate reward model and RL. *(Ch. 17)*
- **distillation** — any training procedure in which a student model learns from the outputs (logits, samples or per-token scores) of a teacher model rather than only from a dataset. *(Ch. 20)*
- **distillation of a labelling function** — labelling a sample with an expensive model, then training a cheap model to imitate the labels and score everything. *(Ch. 8)*
- **distilled data** — a stronger model's answers to your prompts, used as SFT targets for a weaker model (off-policy distillation). *(Ch. 15)*
- **dot product** — the sum of coordinate-wise products of two vectors; large when they align, but grows with their lengths. *(Ch. 3)*
- **DPO** — direct preference optimization: shifting the policy toward preferred answers without a separate reward model. *(Ch. 0, Ch. 14)*
- **Dr. GRPO** — a 2025 correction removing GRPO's per-answer length normalisation and std normalisation, which biased training toward long wrong answers. *(Ch. 19)*
- **draft model** — the small model that proposes tokens in speculative decoding. *(Ch. 7)*
- **DSA** — DeepSeek Sparse Attention (2025): a Lightning Indexer scores all keys cheaply, full attention runs over the top-k. *(Ch. 12)*
- **DualPipe** — DeepSeek-V3's pipeline schedule that overlaps the forward and backward passes of different micro-batches to hide expert-parallel communication. *(Ch. 11)*
- **duplicated work** — the failure where a vague split makes workers investigate the same thing; symptom: overlapping reports. *(Ch. 28)*
- **dynamic sampling** — skipping rollout groups whose rewards are all equal (zero advantages) and sampling more until the batch is full of informative groups. *(Ch. 19)*

## E

- **effective branching factor** — the plain reading of perplexity: how many options the model is, on average, choosing between. *(Ch. 1)*
- **elicitation** — in MCP, a server asking the human a question mid-call, routed through the host. *(Ch. 26)*
- **embedding** — a learned table mapping each token id to a vector of d numbers; the first layer of a language model. *(Ch. 3)*
- **embedding matrix** — the (V × d) matrix E whose row t is the vector for token t. *(Ch. 3)*
- **entropy** — `−Σ p_i log₂ p_i`, the average surprise of a distribution in bits; 0 when certain, `log₂ V` when uniform. *(Ch. 7, Ch. 19)*
- **entropy collapse** — the failure in which the policy's entropy falls to near zero early in RL, every group becomes all-equal and learning stops. *(Ch. 19)*
- **environment** — any object with `reset()` and `step(text)`; it starts the episode, answers tool calls with observations, grades final answers, and says when the episode is done. *(Ch. 21)*
- **EOS token** — the end-of-sequence special token placed between packed documents. *(Ch. 8)*
- **eval-driven development** — write and freeze the eval before training, decontaminate against it, run every checkpoint through it, change one thing at a time, read the failures. *(Ch. 23)*
- **evals** — measurements of a checkpoint on benchmarks, with judges and with users; feed back into every training stage. *(Ch. 0)*
- **evaluation** — producing numbers about a model's quality that survive the known traps: an unchecked task, a too-small sample, contamination, and gaming. *(Ch. 23)*
- **evaluator** — the judging agent, which sees only the task and the candidate; replaceable by a program via `accept_fn`. *(Ch. 28)*
- **exact deduplication** — dropping documents whose normalised text hashes to a value already seen. *(Ch. 8)*
- **exact-match eval** — generate one answer per item and have a program compare it with the reference; the verifiable reward reused as a metric. *(Ch. 23)*
- **expert parallelism (EP)** — spreading the experts of a mixture-of-experts layer across devices and routing tokens to them with all-to-all communication. *(Ch. 11)*
- **exposure bias** — the failure of a model trained only on good text to recover after its own mistakes at generation time, because it never saw the states that follow a mistake; errors then compound. *(Ch. 20)*

## F

- **fan-out** — running the workers concurrently (threads here), so their latencies overlap. *(Ch. 28)*
- **fault tolerance** — the checkpointing, asynchronous saving and elastic-restart machinery that lets a run survive hardware failing every few hours at cluster scale. *(Ch. 11)*
- **fine-grained experts** — many small experts (e.g. 256, top-8) instead of a few large ones, giving the router far more combinations. *(Ch. 12)*
- **FineWeb-Edu** — a 1.3T-token subset of FineWeb selected by a classifier trained to imitate Llama-3-70B's 0–5 "educational value" ratings. *(Ch. 8)*
- **FlashAttention** — an exact attention implementation that processes keys in blocks and never materialises the T × T score matrix. *(Ch. 5)*
- **FLOP** — one floating-point multiply or add; training costs about `6·N·D` FLOPs for `N` parameters and `D` tokens. *(Ch. 6, Ch. 9)*
- **forward KL divergence** — `KL(teacher ‖ student)`: the teacher-weighted log-ratio of teacher to student probabilities; mode-covering, used in logit distillation. *(Ch. 20)*
- **forward pass** — running the model on an input to produce a prediction. *(Ch. 4)*
- **fp32 / bf16** — 32-bit floats (8 exponent, 23 mantissa bits) and bfloat16 (8 exponent, 7 mantissa bits): same range, ~3 significant digits, half the memory. *(Ch. 10)*
- **FP8 / FP4** — 8-bit (e4m3, e5m2) and 4-bit (NVFP4, MXFP4) floating-point formats with block scaling; FP8 is the 2026 training default at scale, FP4 is being validated. *(Ch. 10)*
- **FSDP** — Fully Sharded Data Parallel, PyTorch's implementation of ZeRO stage 3. *(Ch. 11)*
- **full fine-tuning** — updating every parameter of the model during SFT, at the cost of a full copy of the weights and optimizer state per fine-tune. *(Ch. 15)*

## G

- **gated DeltaNet** — a linear-attention variant whose update overwrites the part of the state matching the current key (delta rule) and adds a gate. *(Ch. 12)*
- **generalization** — how well a model does on data it was not trained on, as opposed to memorising its training data. *(Ch. 1)*
- **generative reward model** — an LLM that writes a critique and then a verdict instead of emitting a scalar score (also called a critic model). *(Ch. 17)*
- **generator/evaluator** — a loop in which a generator proposes a candidate and a fresh-context evaluator judges it, until ACCEPT or `max_rounds`. *(Ch. 28)*
- **GMPO** — geometric-mean policy optimisation, a 2025 variant optimising the geometric mean of per-token weighted ratios for a narrower ratio range. *(Ch. 19)*
- **gold items** — items with a known correct label mixed blind into a labeling stream to measure each annotator's accuracy continuously. *(Ch. 16)*
- **Goodhart's law** — "when a measure becomes a target it ceases to be a good measure": optimising a proxy reward eventually improves the proxy without improving the goal. *(Ch. 17)*
- **gradient** — the vector of derivatives of the loss with respect to every parameter; points uphill. *(Ch. 4)*
- **gradient accumulation** — summing gradients over several micro-batches before one optimizer step, to get a larger effective batch in the same memory. *(Ch. 10)*
- **gradient bucketing** — grouping gradients into large buckets so each all-reduce is one big message instead of many small ones. *(Ch. 11)*
- **gradient clipping** — scaling all gradients down so their global L2 norm does not exceed a threshold (1.0); the pre-clip norm is a key health signal. *(Ch. 10)*
- **gradient dead zone** — the set of tokens whose gradient the clip has zeroed, disproportionately rare tokens of good answers under a symmetric clip. *(Ch. 19)*
- **gradient descent** — repeatedly stepping every parameter a small amount against its gradient. *(Ch. 4)*
- **gradient norm** — the global L2 norm of all parameter gradients before clipping; a slow rise is the usual early warning of a loss spike. *(Ch. 10)*
- **greedy decoding** — always choosing the highest-probability next token; deterministic and prone to loops. *(Ch. 1, Ch. 7)*
- **Group Relative Policy Optimization (GRPO)** — a policy-gradient method that samples a group of answers per prompt and uses the group's mean reward as the baseline, removing the critic network. *(Ch. 19)*
- **grouped-query attention (GQA)** — h query heads share h_kv < h key/value heads, shrinking k/v parameters and the KV cache. *(Ch. 5)*
- **GSPO** — a 2025 variant using one sequence-level importance ratio (the geometric mean of per-token ratios) per answer, clipped with a tiny ε. *(Ch. 19)*

## H

- **hand-off** — what one stage passes to the next; should be a document (file, task object) so it can be checked and resumed. *(Ch. 28)*
- **harness** — the program around a model that decides what it may do, verifies what it did, persists what happened and stops it when it should; "the harness is the product". *(Ch. 27)*
- **held-out data** — text kept out of training and used only for measurement. *(Ch. 1)*
- **heuristic filter** — a hand-written rule that drops a document for a measurable property (length, symbol ratio, repeated lines), as in the Gopher and C4 rule sets. *(Ch. 8)*
- **hook** — a function the harness owner attaches to a point in the loop; code, not prompt text, so it cannot be argued with. *(Ch. 27)*
- **host** — in MCP, the application that owns the model and the agent loop and decides what the model may do. *(Ch. 26)*
- **hybrid** — an architecture that mixes many linear/SSM layers with a few full-attention layers kept for exact recall. *(Ch. 12)*
- **hyper-connections** — replacing the single residual stream with several streams mixed by small learned matrices at every layer. *(Ch. 12)*
- **hyperparameter transfer** — tuning learning rate and related settings on a small proxy model and reusing them at scale, enabled by muP. *(Ch. 9)*

## I

- **idempotent** — a tool call that produces the same effect and result if repeated with the same arguments, and can therefore be retried safely. *(Ch. 26)*
- **implicit reward** — in DPO, β·(log π_θ(y|x) − log π_ref(y|x)): how much more likely the policy makes an answer than the reference does. *(Ch. 17)*
- **importance weight** — the ratio π_θ/π_old that corrects a gradient computed under the current policy for samples drawn from an older one, allowing sample re-use. *(Ch. 18)*
- **induction heads** — attention heads that find an earlier occurrence of the current token and attend to what followed it; the mechanism behind in-context copying. *(Ch. 6)*
- **inference-aware scaling law** — a scaling law that minimises training FLOPs plus expected lifetime serving FLOPs (2·N per served token). *(Ch. 9)*
- **initializer** — a read-only agent that turns a task into a written plan and initial state on the first run. *(Ch. 27)*
- **initializer/coder pattern** — one initializer writes the plan; short-lived coder sessions each pick up the files, do bounded work and hand to a verifier. *(Ch. 27)*
- **instruct model** — a model that follows instructions and chats, made by SFT and preference learning on a base model. *(Ch. 0, Ch. 14)*
- **instruction data** — (prompt, response) demonstrations used for SFT; sourced from humans, synthetic generation, distillation from a stronger model, or rejection sampling. *(Ch. 15)*
- **interleaved pipeline** — a pipeline schedule in which each device holds several non-contiguous chunks of layers, shrinking the bubble at the cost of more communication. *(Ch. 11)*
- **IPO** — a DPO variant that replaces the sigmoid loss by a squared loss asking the log-ratio gap to equal 1/(2β), more robust to noisy labels. *(Ch. 17)*
- **irreducible loss E** — the constant floor in L = E + A/N^α + B/D^β: the entropy of the text itself, which no model can beat. *(Ch. 9)*
- **isError** — the flag on an MCP `tools/call` result that marks the text as a tool-level error the model should read, as opposed to a protocol-level JSON-RPC error the client raises. *(Ch. 26)*
- **iso-FLOP curve** — the loss of models that all cost the same compute C, plotted against model size; U-shaped, with the best model at the bottom. *(Ch. 9)*

## J

- **Jaccard similarity** — |A ∩ B| / |A ∪ B| for two sets; the fraction of shingles two documents share. *(Ch. 8)*
- **jailbreak** — an input crafted to make a model violate its principles: role-play framings, encodings, many-shot compliance examples, or instructions hidden in tool results. *(Ch. 22)*
- **JSON-RPC 2.0** — the message format under MCP: a request has an id, a method and params; a response carries the same id with a result or an error object. *(Ch. 26)*
- **judge** — any function `(prompt, answer_a, answer_b) → "A" | "B" | "tie"`; a rule, a model, or a human. *(Ch. 23)*

## K

- **key** — the vector a token advertises to be matched against queries. *(Ch. 5)*
- **Kimi Delta Attention** — Kimi Linear's linear-attention layer with a delta-rule state update, used 3:1 with MLA layers. *(Ch. 12)*
- **KL divergence** — a measure of how far one distribution is from another; for a policy and reference, the average of log π_θ(y) − log π_ref(y) over the policy's own samples. *(Ch. 17)*
- **KTO** — a preference method that needs only single answers labelled good or bad rather than pairs, with a loss shaped by prospect-theory loss aversion. *(Ch. 17)*
- **KV cache** — the stored keys and values of past tokens, so each generation step only computes the newest token's q, k, v. *(Ch. 5, Ch. 7)*

## L

- **language identification** — assigning each document a language and confidence, e.g. with fastText; documents below a threshold are dropped. *(Ch. 8)*
- **language model** — a function from a sequence of tokens to a probability distribution over the next token. *(Ch. 1)*
- **large language model (LLM)** — a neural network trained on very large amounts of text to predict the next token, then post-trained to follow instructions, reason and act. *(Ch. 0)*
- **layer** — a set of neurons that read the same inputs; in matrix form, one matrix multiply plus an activation. *(Ch. 4)*
- **LayerNorm** — normalisation that subtracts a vector's mean, divides by its standard deviation, then applies a learned scale and shift. *(Ch. 6)*
- **learning rate** — the multiplier on the gradient (or optimizer update) that sets the size of each step. *(Ch. 4)*
- **learning-rate schedule** — a rule for changing the learning rate over training, typically warmup then decay. *(Ch. 4)*
- **least privilege** — giving the agent only the tools and the write targets a task needs, so that a persuaded model cannot do more damage than the task allows. *(Ch. 26)*
- **length bias** — a judge's preference for the longer answer, independent of correctness. *(Ch. 23)*
- **length control** — keeping RL-trained answers from growing without bound, via overlong shaping, un-normalised losses or length budgets. *(Ch. 19)*
- **Lightning Indexer** — the small, low-precision scoring module in DSA that selects which keys each query attends to. *(Ch. 12)*
- **likelihood displacement** — the DPO failure in which the pairs become well separated while the chosen answers themselves become less likely than under the reference, because only the difference of log-probabilities is constrained. *(Ch. 17)*
- **Likert rating** — a label format that scores one answer on a fixed scale (1–5 or 1–7); cheap but prone to annotator drift and ceiling effects. *(Ch. 16)*
- **linear attention** — attention without the softmax, which can be computed as a recurrence with a fixed-size state. *(Ch. 12)*
- **linear/state-space layer** — a sequence layer (Mamba-2, gated DeltaNet, linear attention) that keeps a fixed-size state instead of a growing KV cache. *(Ch. 12)*
- **LLM-as-judge** — using a prompted language model to compare or grade answers, as an evaluation tool or as a training reward. *(Ch. 17, Ch. 23)*
- **LLM-as-labeler** — using a language model, prompted with the guidelines and the candidates, to produce labels in place of a human; also called LLM-as-judge. *(Ch. 16)*
- **load-balancing loss** — an auxiliary loss (E · Σ f_e · P_e) that pushes the router to spread tokens evenly over experts. *(Ch. 12)*
- **locality-sensitive hashing (LSH)** — bucketing signatures by bands so that similar documents collide and only bucket-mates are compared. *(Ch. 8)*
- **log-probability** — the logarithm of a probability; turns the chain-rule product into a sum and avoids underflow. *(Ch. 1)*
- **logit distillation** — Hinton's 2015 recipe: on fixed text, train the student's next-token distribution to match the teacher's whole distribution with a forward KL loss. *(Ch. 20)*
- **logits** — the `V` unnormalised scores per position produced by the output head before the softmax. *(Ch. 6)*
- **long-context extension** — a short, low-LR training phase on long documents after re-scaling the positional encoding, teaching the model to use positions beyond its pretraining length. *(Ch. 13)*
- **LoRA (low-rank adaptation)** — parameter-efficient fine-tuning that freezes each weight matrix W and trains a rank-r update B·A (scaled by α/r) added to it; B starts at zero and the update is merged into W after training. *(Ch. 15)*
- **loss curve** — training or validation loss plotted against steps or tokens; read on a log-token axis for its power-law shape. *(Ch. 9)*
- **loss function** — a function of prediction and truth that returns one number, zero when perfect and larger the worse the prediction. *(Ch. 4)*
- **loss mask** — a per-token 0/1 flag saying whether that token's prediction contributes to the training loss; in SFT only assistant tokens and their closing `<|end|>` are 1. *(Ch. 14)*
- **loss mask (trajectory)** — a 0/1 vector over the trajectory's positions that is 1 only on tokens the policy generated; prompt, role tags and tool results are 0. *(Ch. 21)*
- **loss spike** — a sudden jump in training loss, from too-high LR, unbounded logits or a bad batch; may recover or diverge. *(Ch. 10)*
- **lost context** — the failure where a later stage never learned what an earlier one knew because it lived in a context, not a document. *(Ch. 28)*

## M

- **majority voting** — sampling several answers and returning the most common extracted final answer (self-consistency); test-time compute without a verifier. *(Ch. 19)*
- **Mamba-2** — a selective state-space model whose scan is a linear-attention-style recurrence with input-dependent decay. *(Ch. 12)*
- **manifold-constrained hyper-connections (mHC)** — DeepSeek-V4's hyper-connections with the mixing matrices constrained for stability at depth. *(Ch. 12)*
- **Markov assumption** — the (false but useful) assumption that only the last n − 1 tokens affect the next one. *(Ch. 1)*
- **master copy** — the fp32 copy of the weights kept for the optimizer update while forward/backward run in bf16 or FP8. *(Ch. 10, Ch. 11)*
- **maximum-likelihood estimate** — the parameter choice that makes the training data most probable; for an n-gram it is the plain count ratio count(c, w) / count(c). *(Ch. 1)*
- **mean squared error (MSE)** — the average squared difference between predictions and targets. *(Ch. 4)*
- **memorisation** — a model reproducing training text verbatim; grows with how often the text was repeated in training. *(Ch. 8)*
- **memory-bound** — limited by how fast bytes can be read from memory rather than by arithmetic; batch-1 decode is the canonical case. *(Ch. 7)*
- **merge** — one learned BPE rule: a pair of symbol ids and the new id that replaces them; the ordered merge list is the tokenizer. *(Ch. 2)*
- **MFU** — model FLOPs utilisation: the useful 6·N·D FLOPs achieved divided by the hardware's peak FLOP/s; ~40% is good for a large run. *(Ch. 9)*
- **mHC** — manifold-constrained hyper-connections: several parallel residual streams with learned mixing, used by DeepSeek-V4 in place of plain residual addition. *(Ch. 6)*
- **micro-batch** — a slice of the batch that flows through the pipeline as a unit so that stages can work concurrently. *(Ch. 11)*
- **mid-training** — the final stretch of pretraining on a smaller, higher-quality mix with decaying learning rate, often with context extension. *(Ch. 0, Ch. 13)*
- **min-p** — keep tokens whose probability is at least `min_p` times the top token's; the cut-off adapts to the model's confidence. *(Ch. 7)*
- **MinHash** — for each of k hash functions, the minimum hash over a document's shingles; two signatures agree in a fraction ≈ Jaccard of positions. *(Ch. 8)*
- **minibatch** — a small random sample of training examples used for one step. *(Ch. 4)*
- **mixed-precision training** — computing in bfloat16 while keeping fp32 master weights so small updates are not rounded away. *(Ch. 11)*
- **MLA** — Multi-head Latent Attention: store one compressed latent per token per layer and up-project keys and values on the fly, shrinking the KV cache. *(Ch. 12)*
- **mode-covering** — a divergence minimised by spreading probability mass over every region where the target has mass (forward KL). *(Ch. 20)*
- **mode-seeking** — a divergence minimised by putting probability mass only where the target's mass is high, ignoring the rest (reverse KL). *(Ch. 20)*
- **Model Context Protocol (MCP)** — the open standard (November 2024, revision 2025-11-25) by which an agent lists and calls tools, reads resources and fetches prompts from a separate server over JSON-RPC. *(Ch. 26)*
- **model FLOPs utilisation (MFU)** — the fraction of the hardware's peak FLOP/s spent on the model's own 6·N·D operations; 40–50% is good at scale. *(Ch. 11)*
- **momentum** — an optimizer that steps along a running average (velocity) of past gradients. *(Ch. 4)*
- **monitoring** — logging and auditing agent transcripts so that violations are found after the fact even when the weights and the gates missed them. *(Ch. 22)*
- **multi-agent system** — a program that runs several agents, each with its own context, and combines their outputs; the design decisions are who sees what and how outputs combine. *(Ch. 28)*
- **multi-head attention** — running h independent attentions on slices of width head_dim = d/h and concatenating their outputs. *(Ch. 5)*
- **multi-head latent attention (MLA)** — caching a small compressed latent per token and reconstructing per-head keys and values from it (DeepSeek-V2/V3). *(Ch. 5)*
- **multi-layer perceptron (MLP)** — a chain of layers, each feeding the next. *(Ch. 4)*
- **multi-query attention (MQA)** — GQA with a single key/value head. *(Ch. 5)*
- **multi-token prediction (MTP)** — extra heads trained to predict tokens t+2, t+3, … from the same hidden state, adding training signal and a speculative-decoding draft. *(Ch. 12)*
- **Muon** — a 2024–2026 optimizer that orthogonalises the momentum-averaged gradient of each weight matrix before stepping. *(Ch. 4, Ch. 10)*
- **muP** — Maximal Update Parametrisation: scaling initialisation and per-layer learning rates with width so the optimal learning rate is width-independent. *(Ch. 9)*

## N

- **n-gram model** — a language model whose context is the previous n − 1 tokens, estimated by counting. *(Ch. 1)*
- **n-gram overlap** — the decontamination test: any document sharing an n-word window (13 in production) with an eval item is removed. *(Ch. 8)*
- **nats** — the unit of cross-entropy loss when the logarithm is natural; an untrained model scores `ln V` nats. *(Ch. 6)*
- **near-duplicate** — a document that differs from another only by a few words, a date or a byline. *(Ch. 8)*
- **neuron** — a unit that computes an activation function of a weighted sum of its inputs plus a bias. *(Ch. 4)*
- **Newton–Schulz iteration** — a fixed polynomial of matrix multiplies, applied a few times, that pushes every singular value toward 1 without computing an SVD. *(Ch. 10)*
- **next-token predictor** — a function from a sequence of tokens to a probability distribution over the next token; the core of every LLM. *(Ch. 0)*
- **norm** — the length of a vector, the square root of its dot product with itself. *(Ch. 3)*
- **notification** — a JSON-RPC request without an id, which receives no reply (e.g. `notifications/initialized`). *(Ch. 26)*
- **NSA** — Native Sparse Attention (2025): compressed, selected and sliding-window branches gated per query, trained sparse from the start. *(Ch. 12)*

## O

- **observability** — being able to answer what the agent did and why it cost that much without re-running it: the event stream, `harness.log`, backend records. *(Ch. 27)*
- **off-policy** — training on text that was not produced by the policy being trained (a dataset, or the teacher's samples). *(Ch. 20)*
- **off-policy data** — training data produced by something other than the current model: humans, a teacher model, or an earlier checkpoint. *(Ch. 14)*
- **off-policy lag** — in asynchronous training, the gap between the policy that produced a trajectory and the one being updated; makes PPO's importance ratio differ from 1. *(Ch. 21)*
- **on-policy** — training on text the policy being trained produced itself. *(Ch. 20)*
- **on-policy data** — training data sampled from the model being trained at (or near) its current weights. *(Ch. 14)*
- **on-policy distillation** — transferring a teacher's skills to a student by having the student sample and the teacher grade each token. *(Ch. 0, Ch. 14)*
- **on-policy distillation (OPD)** — distillation in which the student samples its own answers and the teacher scores every sampled token; minimises the reverse KL from student to teacher with a REINFORCE-style per-token update. *(Ch. 20)*
- **on-policy pairs** — preference pairs whose chosen and rejected answers were both sampled from the model being trained and graded afterwards, so the rejected answers are the model's real mistakes. *(Ch. 17)*
- **one-hot vector** — a vector of length V that is zero everywhere except for a single 1 at the token's index. *(Ch. 3)*
- **optimizer** — the rule that turns gradients into parameter changes (SGD, momentum, Adam, AdamW, Muon). *(Ch. 4)*
- **orchestrator** — the agent that splits a task into subtasks and merges the workers' reports. *(Ch. 28)*
- **orchestrator–workers** — the pattern split → parallel workers → merge, implemented by `orchestrate()`. *(Ch. 28)*
- **ORPO** — a reference-free method that adds an odds-ratio preference penalty to the SFT loss so both can be trained in one pass from a base model. *(Ch. 17)*
- **orthogonal matrix** — a matrix that only rotates/reflects, never stretches: U Vᵀ from the SVD of the momentum matrix. *(Ch. 10)*
- **over-refusal** — declining harmless requests that share surface features with harmful ones; reported alongside harmful-compliance rate because either alone can be optimised trivially. *(Ch. 22)*
- **overfitting** — training loss keeps improving while performance on unseen data gets worse. *(Ch. 4)*
- **overlap** — starting the all-reduce of already-finished gradient buckets while the backward pass is still computing later ones. *(Ch. 11)*
- **overtraining** — training a model on far more tokens than the compute-optimal count (e.g. 1,900 tokens/parameter for Llama-3 8B) to get a smaller, cheaper-to-serve model. *(Ch. 9)*

## P

- **packing** — concatenating documents into one token stream separated by EOS so no padding is wasted. *(Ch. 8, Ch. 15)*
- **paged attention** — allocating the KV cache in fixed-size blocks, like virtual memory, so variable-length sequences pack without fragmentation. *(Ch. 7)*
- **pair accuracy** — the fraction of preference pairs on which a reward model or policy scores the chosen answer above the rejected one; 0.5 is chance. *(Ch. 17)*
- **pairwise comparison** — a label format that shows two answers to one prompt and asks which is better or whether they tie; the format of DPO data. *(Ch. 16)*
- **parallel thinking** — producing several reasoning paths in one inference pass and merging them, reported to beat one long chain at equal token budget. *(Ch. 19)*
- **pass@N** — the fraction of prompts for which at least one of N samples is correct; an upper bound on best-of-N selection and the standard probe for whether RL added capability or only sharpened. *(Ch. 19)*
- **per-token advantage** — in OPD, `A_t = log π_t(y_t) − log π_s(y_t)`: how much more the teacher liked the student's token than the student itself did; positive means push the token up. *(Ch. 20)*
- **permission gate** — a harness rule requiring human approval before destructive or irreversible actions. *(Ch. 22, Ch. 27)*
- **permission policy** — `allow_all`, `allow_read_only` (only tools flagged `read_only`) or `ask` (defer to a `permission_fn`; deny if none). *(Ch. 27)*
- **perplexity** — 2 raised to the average bits per token; the effective number of equally likely choices per step; lower is better. *(Ch. 1, Ch. 10, Ch. 23)*
- **PII** — personally identifiable information (e-mails, phone numbers, IDs) scrubbed or replaced with placeholder tokens before training. *(Ch. 8)*
- **pipeline** — agents run in sequence, each stage's output being the next stage's whole input. *(Ch. 28)*
- **pipeline bubble** — the idle time in a pipeline while stages wait for each other; fraction (p−1)/(m+p−1) for p stages and m micro-batches. *(Ch. 11)*
- **pipeline parallelism (PP)** — assigning consecutive layers to consecutive devices and streaming micro-batches through them. *(Ch. 11)*
- **plan file** — `PLAN.md`: the task decomposed into steps, written once by the initializer, read by every session, read-only for the agent. *(Ch. 27)*
- **planner/generator/evaluator** — a three-agent variant: a planner writes acceptance criteria once, a generator produces candidates, a fresh-context evaluator judges them. *(Ch. 27)*
- **policy** — the model being trained, written π_θ; in preference learning and RL it is the distribution over answers given a prompt. *(Ch. 17)*
- **policy gradient theorem** — ∇J = E_{y~π}[R ∇log π(y)]: the gradient of expected reward is the reward-weighted score function, estimable by sampling. *(Ch. 18)*
- **position bias** — a judge's tendency to prefer whichever answer is shown first; detected by judging both orders (the swap test). *(Ch. 16, Ch. 23)*
- **position interpolation** — feeding RoPE scaled-down positions so a longer window fits inside the trained angle range. *(Ch. 13)*
- **post-norm** — the original 2017 arrangement `x = norm(x + f(x))`, normalising the stream after each addition. *(Ch. 6)*
- **post-tool hook** — a hook run on a tool result; may return a replacement (truncate, redact, log) and runs even for blocked calls. *(Ch. 27)*
- **post-training** — every training stage after pretraining (SFT, preference learning, RLVR, distillation, safety), using far less data and aimed at changing behaviour rather than knowledge. *(Ch. 14)*
- **pre-norm** — the modern arrangement `x = x + f(norm(x))`, normalising only the copy a sub-layer reads; trains stably at depth. *(Ch. 6)*
- **pre-tokenization** — splitting text with a regular expression into chunks that BPE merges never cross (words with a leading space, contractions, digit groups, punctuation). *(Ch. 2)*
- **pre-tool hook** — a hook run before a tool call; returns `None` to allow or a string to block the call with that reason. *(Ch. 27)*
- **preference learning** — training from pairs of answers where one was preferred, via a reward model or DPO. *(Ch. 0)*
- **preference pair** — (prompt, chosen, rejected): two answers to one prompt and the verdict that the first is better; the unit of preference-learning data. *(Ch. 16)*
- **prefill** — the first phase of generation: one forward pass over the whole prompt that fills the KV cache; compute-bound. *(Ch. 7)*
- **pretraining** — training on trillions of tokens with the next-token objective; produces the base model. *(Ch. 0)*
- **pretraining loop** — the program that repeatedly samples a batch, runs forward, computes the loss, backpropagates, clips and updates the parameters. *(Ch. 10)*
- **principal component analysis (PCA)** — projecting points onto the directions of greatest spread, used to draw high-dimensional vectors in 2-D. *(Ch. 3)*
- **probability distribution** — a set of non-negative numbers, one per possible outcome, that add up to 1. *(Ch. 1)*
- **probability ratio** — ρ = π_θ(a)/π_old(a), how much a token's probability has changed since the policy that produced the sample. *(Ch. 18)*
- **progress file** — `PROGRESS.md`: appended by the harness after each session with the model's summary, the verifier's PASS/FAIL and the test output tail. *(Ch. 27)*
- **prompt caching** — reusing the KV cache of a shared prompt prefix across requests so it is prefilled once. *(Ch. 7)*
- **prompt injection** — a jailbreak delivered through content the model reads rather than the user's message, such as a web page or a tool result. *(Ch. 22, Ch. 26)*
- **prompts** — in MCP, reusable prompt templates with arguments that a server offers. *(Ch. 26)*
- **Proximal Policy Optimization (PPO)** — a policy-gradient method that clips the per-token probability ratio to a window around 1 so one update cannot move the policy too far from the sampler. *(Ch. 18)*

## Q

- **QK-norm** — normalising queries and keys before the attention dot product so attention logits stay bounded. *(Ch. 10)*
- **quality classifier** — a small model scoring documents for training value, trained on labels from an expensive judge (FineWeb-Edu, DCLM). *(Ch. 8)*
- **quantization** — storing weights (or activations) in fewer bits with per-group scale factors; fewer bytes means faster memory-bound decode. *(Ch. 7)*
- **query** — the vector a token uses to ask "what am I looking for?". *(Ch. 5)*

## R

- **R1-Zero** — DeepSeek's model trained by GRPO directly on a base model with only accuracy and format rewards, in which long reasoning emerged. *(Ch. 19)*
- **random windows** — batching by taking `batch_size` random `seq_len`-token slices of the packed stream; the target is the slice shifted one token right. *(Ch. 10)*
- **rank** — the number r of independent directions in a LoRA update; r · (d_in + d_out) trainable numbers per layer instead of d_in · d_out. *(Ch. 15)*
- **ranking** — a label format that orders k answers to one prompt; each ranking expands into all of its pairwise comparisons for training. *(Ch. 16)*
- **reasoning model** — a model trained with RLVR to produce intermediate steps before its answer. *(Ch. 0, Ch. 14)*
- **red-teaming** — the organised search, by people or attacker models, for inputs that make a model violate its principles. *(Ch. 22)*
- **reference** — a frozen copy of the policy taken before training (π_ref), against which DPO measures how far the policy has moved. *(Ch. 17)*
- **REINFORCE** — the algorithm that follows the policy gradient estimated from a few samples, optionally with a baseline. *(Ch. 18)*
- **reinforcement learning (RL)** — learning from scores on the model's own attempts rather than from fixed targets: sample, score, make higher-scoring behaviour more likely. *(Ch. 18)*
- **reinforcement learning from AI feedback (RLAIF)** — stage 2 of CAI: an AI judge reading the principles scores or ranks samples, and the verdicts become preference pairs for a reward model or DPO. *(Ch. 22)*
- **reinforcement learning with verifiable rewards (RLVR)** — RL where the reward comes from an automatic checker (tests, answer keys); the recipe behind reasoning models. *(Ch. 0, Ch. 14, Ch. 19)*
- **rejection sampling** — sampling several answers from the model and keeping those a verifier or judge accepts, as SFT data or preference pairs. *(Ch. 15, Ch. 16)*
- **rejection-sampling SFT** — sequence-level distillation with a verifier: sample answers from the teacher, keep the correct ones, SFT the student on them; how the DeepSeek-R1-Distill models were made. *(Ch. 20)*
- **repetition penalty** — dividing (positive) or multiplying (negative) the logits of tokens already in the sequence to discourage loops. *(Ch. 7)*
- **replay** — keeping a slice of the original pretraining data in a continued-training mix so old capabilities are not overwritten. *(Ch. 13, Ch. 15)*
- **residual stream** — the width-`d` vector per token that flows straight through the model; every sub-layer reads it (through a norm) and writes to it by addition, and it is never overwritten. *(Ch. 6)*
- **resources** — in MCP, read-only data a server exposes by URI for the host to attach to the context. *(Ch. 26)*
- **resuming** — starting a new session from the files alone; in `MiniHarness` it is the same function as `run_session()`. *(Ch. 27)*
- **return** — the sum of future rewards from a state; equal to the final reward when there is only one. *(Ch. 18)*
- **reverse KL** — `KL(student ‖ teacher)` measured on the student's own samples; mode-seeking, the objective of on-policy distillation. *(Ch. 20)*
- **reward** — the score assigned to a trajectory; for a language model one number for the whole answer, arriving at the end. *(Ch. 18)*
- **reward hacking** — a policy achieving high reward by exploiting flaws in the reward rather than doing the intended task. *(Ch. 17)*
- **reward model** — a model trained to score answers by how much humans (or judges) would prefer them. *(Ch. 0, Ch. 14, Ch. 17)*
- **reward shaping** — adding small auxiliary terms (format, length) to a sparse reward so early training has a gradient; dangerous when they outweigh correctness. *(Ch. 19)*
- **ridge point** — a chip's peak FLOP/s divided by its memory bandwidth (~300 FLOP/byte on an H100); below it a workload is memory-bound, above it compute-bound. *(Ch. 7)*
- **ring all-reduce** — an all-reduce over a ring of N devices (reduce-scatter then all-gather) in which each device moves 2(N−1)/N × the tensor size, independent of N. *(Ch. 11)*
- **RMSNorm** — normalisation that divides a vector by its root-mean-square and multiplies by a learned gain; no mean subtraction, no bias; the 2026 default. *(Ch. 6)*
- **rollout-as-a-service** — an infrastructure pattern in which a separate service provisions, resets and tears down environments for the trainer over the network. *(Ch. 21)*
- **roofline** — the model that bounds a workload's speed by `min(peak compute, bandwidth × intensity)`. *(Ch. 7)*
- **rotary position embeddings (RoPE)** — rotating pairs of channels of q and k by angles proportional to position so that scores depend only on relative offset. *(Ch. 5)*
- **router** — the small linear layer in a mixture-of-experts block that scores each token against every expert. *(Ch. 12)*
- **rubric** — a checklist of yes/no (or scored) properties of a response, such as "final answer is correct", that turns a judgement into items two labelers can compare. *(Ch. 16)*
- **rubric reward** — a checklist of yes/no criteria applied to an answer and averaged into a score; the 2025–2026 middle ground between verifiers and learned reward models. *(Ch. 17)*
- **runaway cost** — the failure where every extra agent adds a context that is re-read each turn, doubling the bill without moving the score. *(Ch. 28)*

## S

- **safety evals** — measurements of a model's behaviour against its principles: harmful-compliance rate, over-refusal rate, sycophancy rate, harmful-capability uplift, agentic-safety tests. *(Ch. 22)*
- **safety training** — post-training that shapes refusals, honesty and behaviour under autonomy. *(Ch. 0)*
- **sampling** — choosing the next token at random in proportion to its probability. *(Ch. 1, Ch. 26)*
- **sandbox** — an isolated process or container in which an agent's actions cannot reach anything outside, that can be reset to a known state for every rollout. *(Ch. 21, Ch. 26)*
- **scaling law** — an empirical power-law relationship between a model's loss and its parameters N, training tokens D, or compute C, holding across many orders of magnitude. *(Ch. 9)*
- **score-function trick** — the identity ∇π = π ∇log π, which turns the gradient of an expected reward into an expectation over the policy's own samples. *(Ch. 18)*
- **Self-Instruct** — a synthetic-data method in which a model, seeded with a few tasks, invents new prompts and answers them; Evol-Instruct rewrites prompts to be progressively harder. *(Ch. 15)*
- **self-preference** — a model judge's preference for answers in its own style. *(Ch. 23)*
- **self-preference bias** — a judge's tendency to prefer text in its own style, including its own outputs; mitigated by judging with a different model family. *(Ch. 16)*
- **sequence parallelism** — (Megatron sense) splitting the norm/dropout operations that TP leaves replicated along the sequence axis; (long-context sense) see context parallelism. *(Ch. 11)*
- **sequence-level distillation** — training the student on whole answers sampled from the teacher instead of on the teacher's logits. *(Ch. 20)*
- **server** — in MCP, a separate program that exposes tools, resources and prompts over a transport. *(Ch. 26)*
- **SGD** — stochastic gradient descent: gradient descent where each gradient is estimated on a random minibatch. *(Ch. 4)*
- **shared expert** — a DeepSeekMoE expert that every token visits regardless of the router, holding knowledge all tokens need. *(Ch. 12)*
- **shared state** — any store more than one agent reads and writes (a file, a git branch, a task object); files are the default on one machine. *(Ch. 28)*
- **shingle** — one of the overlapping n-word windows of a document; the document is represented as its set of shingles. *(Ch. 8)*
- **sigmoid** — the S-shaped function σ(z) = 1 / (1 + e^(−z)) that maps any real number to a probability between 0 and 1. *(Ch. 17)*
- **signature** — a fixed-length summary of a document (64 MinHash values) from which similarity can be estimated. *(Ch. 8)*
- **SimPO** — a reference-free DPO variant whose reward is the average per-token log-probability of the answer, with a required margin γ. *(Ch. 17)*
- **singular values** — the stretch factors of a matrix along its principal directions; an orthogonal matrix has all singular values equal to 1. *(Ch. 10)*
- **skill** — a folder of instructions (a `SKILL.md` plus any scripts) whose one-line description is always in context but whose body is loaded only when the task matches. *(Ch. 27)*
- **sliding window** — a sparse-attention pattern in which each query sees only the most recent w keys. *(Ch. 12)*
- **sliding-window attention** — a mask that lets each query see only the most recent w keys, bounding the cache and the score matrix. *(Ch. 5)*
- **smoothing** — any method that moves a little probability mass from seen events to unseen ones so nothing has probability exactly zero. *(Ch. 1)*
- **softmax** — exponentiate each score and divide by the sum, turning a row of scores into positive weights that sum to 1. *(Ch. 5, Ch. 7)*
- **sparse attention** — attention in which each query scores only a subset of the keys, chosen by position (windows) or by content (NSA, DSA). *(Ch. 12)*
- **sparsity** — the problem that the number of possible contexts grows exponentially with context length, so most are never observed in training. *(Ch. 1)*
- **special token** — a reserved string such as `<|user|>` with its own id that is never split into pieces; used for chat structure and document boundaries. *(Ch. 2)*
- **special tokens** — vocabulary ids reserved for structure (roles, turn ends, padding) that are never produced by encoding ordinary text. *(Ch. 14)*
- **speculative decoding** — a fast draft model proposes `k` tokens and the target verifies them in one pass, accepting the matching prefix; lossless. *(Ch. 7, Ch. 12)*
- **state** — what the agent sees before choosing an action; for a language model, the prompt plus the tokens generated so far. *(Ch. 18)*
- **static batching** — running a fixed group of requests together until the longest finishes. *(Ch. 7)*
- **stop token** — a vocabulary entry (TinyLM: `<|end|>`) that a post-trained model emits to end its turn, halting the decode loop. *(Ch. 7)*
- **Storyland** — the course's synthetic corpus: templated stories about a fixed cast plus two-digit arithmetic and Q&A. *(Ch. 0)*
- **Streamable HTTP** — MCP's transport for remote servers: one HTTP endpoint that can stream replies, with OAuth for authorisation. *(Ch. 26)*
- **strong-to-weak distillation** — training a smaller model from a larger one's outputs (off-policy) and then its token-level judgements (on-policy), as in the Qwen3 recipe. *(Ch. 14)*
- **student** — the model being trained in distillation, usually smaller than the teacher. *(Ch. 20)*
- **sub-agent** — a fresh `Agent` with an empty context, usually fewer tools and the same policy, whose final text alone returns to the parent; context and risk isolation. *(Ch. 27)*
- **subword tokens** — pieces between characters and words: common words are one token, rare words are split into common pieces. *(Ch. 2)*
- **superposition** — storing more features than dimensions by using overlapping directions in the residual stream; why single neurons rarely mean one thing. *(Ch. 6)*
- **supervised fine-tuning (SFT)** — training on prompt → answer examples in a chat format; produces the instruct model. *(Ch. 0, Ch. 14, Ch. 15)*
- **surprisal** — −log₂ P(token | context): how surprised the model was by the token that actually came. *(Ch. 1)*
- **swap test** — judge (a, b), then (b, a) and translate back; a consistent judge gives the same winner, a position-biased one contradicts itself. *(Ch. 23)*
- **SwiGLU** — the gated MLP `down(silu(gate·x) ⊙ up·x)`; three matrices, hidden width `8/3·d` to match a `4d` ReLU MLP's parameter count. *(Ch. 6)*
- **sycophancy** — a preference-trained model's drift toward telling users what they want to hear: agreeing with wrong premises, reversing correct answers under pushback. *(Ch. 22)*
- **sycophantic evaluator** — an evaluator that accepts whatever it is shown; the pattern's characteristic failure, defended against by a program judge or a rubric. *(Ch. 28)*
- **synthetic rephrasing** — having an LLM rewrite low-quality pages or generate Q&A/summaries from high-quality ones to add diverse tokens (Nemotron-CC). *(Ch. 8)*

## T

- **target model** — the large model whose output distribution speculative decoding reproduces exactly. *(Ch. 7)*
- **tasks** — in the 2025-11-25 MCP revision, a way to start a long-running operation and poll or be notified about it instead of blocking a call. *(Ch. 26)*
- **teacher** — the frozen, usually larger model whose outputs supervise the student in distillation. *(Ch. 20)*
- **temperature** — a knob that reshapes the distribution before sampling: below 1 sharpens it toward the top token, above 1 flattens it. *(Ch. 1, Ch. 7)*
- **temperature (in distillation)** — a divisor τ applied to both models' logits before the softmax; τ > 1 flattens the distributions so the student also learns which wrong tokens are nearly right. *(Ch. 20)*
- **tensor parallelism (TP)** — splitting individual weight matrices across devices (columns for the first matmul, rows for the second) with an all-reduce of activations to combine partial sums. *(Ch. 11)*
- **test-time compute** — spending more inference compute per question (more samples, longer or parallel reasoning) to raise accuracy. *(Ch. 19)*
- **text extraction** — turning HTML into main-body text (e.g. with trafilatura), dropping boilerplate. *(Ch. 8)*
- **tied embeddings** — using the embedding matrix as the output projection too, so logits are dot products with token rows; saves V × d parameters. *(Ch. 3, Ch. 6)*
- **TinyLM** — the decoder-only Transformer you build and train through every stage of the course. *(Ch. 0)*
- **token** — the unit of text a model reads and predicts; a word or punctuation mark in Chapter 1, a learned sub-word piece from Chapter 2 on. *(Ch. 1)*
- **token-level loss** — averaging the clipped objective over every answer token in the batch, so long answers weigh more (DAPO); the original GRPO averaged within each answer first. *(Ch. 19)*
- **tokenizer** — the fixed mapping from text to integer ids and back that a model is trained on. *(Ch. 0, Ch. 2)*
- **tool call** — an assistant turn in which the model emits a structured request (a JSON name and arguments) for the harness to execute, answered by a tool-result turn. *(Ch. 14, Ch. 21)*
- **tool design** — the craft of naming, describing, typing and error-messaging a tool so that a model can use it correctly without training on it. *(Ch. 26)*
- **tool search** — keeping a large tool catalogue outside the context and giving the model one tool that returns the schemas of the few tools matching a query. *(Ch. 26)*
- **tool-use bonus** — a small shaping reward paid when the tool was used correctly and the final answer is right; paid unconditionally it invites reward hacking. *(Ch. 21)*
- **top-k** — sampling restricted to the `k` most probable tokens, renormalised. *(Ch. 7)*
- **top-k sparse attention** — 🆕 each query attends to a small selected subset of keys (NSA, DeepSeek Sparse Attention, DeepSeek-V4 CSA/HCA). *(Ch. 5)*
- **top-p** — nucleus sampling: keep the smallest set of tokens whose cumulative probability reaches `p`. *(Ch. 7)*
- **training step** — one trip round forward → loss → backward → optimizer step. *(Ch. 4)*
- **train–test overlap** — evaluation items present in the training data; the target of decontamination. *(Ch. 8)*
- **trajectory** — the full sequence of states and actions in one episode; for a language model, one complete answer ending in `<|end|>`. *(Ch. 18, Ch. 21)*
- **Transformer block** — one attention sub-layer and one MLP sub-layer, each wrapped as `x = x + f(norm(x))`; the unit that is stacked `L` times to make a language model. *(Ch. 6)*
- **transport** — the channel MCP messages travel over: stdio (one JSON object per line on a subprocess's pipes) or Streamable HTTP. *(Ch. 26)*
- **trigram** — an n-gram model with two tokens of context (n = 3). *(Ch. 1)*
- **turn-level reward** — a reward returned by the environment for one turn (e.g. a well-formed tool call), summed with the others into the trajectory reward. *(Ch. 21)*
- **two-stage (distillation recipe)** — off-policy SFT on teacher samples first, to bring the student into the teacher's style, then on-policy distillation. *(Ch. 20)*

## U

- **unit-test eval** — an exact-match eval for code: the generated program is graded by running hidden tests. *(Ch. 23)*
- **up-sampling** — repeating a scarce high-value source (e.g. Wikipedia) several times in the mix. *(Ch. 8)*

## V

- **validation set** — held-out data never trained on, used to detect overfitting and choose models. *(Ch. 4)*
- **value** — the vector a token contributes to the weighted average when it is attended to. *(Ch. 5)*
- **value function** — a learned estimate of the expected reward from a state, used by PPO as its baseline (the critic). *(Ch. 18)*
- **vector** — an ordered list of numbers, thought of as a point or arrow in d-dimensional space. *(Ch. 3)*
- **verbosity bias** — a judge's tendency to prefer the longer answer regardless of content; invisible to the swap test, caught by comparing win-rate against length difference. *(Ch. 16)*
- **verifiable reward** — a reward computed by a program that checks the answer (1 if correct, 0 otherwise); needs no training and is the basis of RLVR. *(Ch. 17)*
- **verification loop** — the rule that a task is never marked done on the model's say-so; the harness runs tests (or another checker) and records the verdict next to the claim. *(Ch. 27)*
- **verifier** — a program that decides whether an answer is correct, producing the verifiable reward. *(Ch. 19, Ch. 27)*
- **vocabulary size trade-off** — a bigger vocabulary gives fewer tokens per text but a bigger V × d embedding matrix and rarer, less well-trained tokens. *(Ch. 2)*

## W

- **WARC** — the archive format Common Crawl uses; stores full HTTP responses including raw HTML. *(Ch. 8)*
- **warmup** — a linear ramp of the learning rate from 0 over the first steps, protecting the model while Adam's statistics are unreliable. *(Ch. 10)*
- **weight** — a learned multiplier on an input to a neuron; the entries of a layer's matrix. *(Ch. 4)*
- **weight decay** — shrinking every parameter slightly toward zero on each step to discourage large weights. *(Ch. 4)*
- **word-level tokens** — one token per word: short sequences, unbounded vocabulary, unknown-word problem. *(Ch. 2)*
- **worker** — an agent that does one subtask with only that subtask in its context. *(Ch. 28)*
- **WSD** — warmup–stable–decay: flat learning rate after warmup, then a short linear decay to zero; any stable-phase checkpoint can be given its own decay. *(Ch. 10)*

## Y

- **YaRN** — a context-extension method that interpolates only the slow RoPE pairs, leaves fast pairs untouched, and adds an attention-temperature correction. *(Ch. 13)*

## Z

- **z-loss** — an auxiliary loss 1e-4 · (log Z)² on the softmax normaliser Z that keeps logits from drifting to huge values. *(Ch. 10)*
- **ZeRO** — Zero Redundancy Optimizer: sharding optimizer states (stage 1), gradients (stage 2) and weights (stage 3) across the data-parallel group instead of replicating them. *(Ch. 11)*
