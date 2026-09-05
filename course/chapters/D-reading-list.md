# Appendix D: Reading list, 2017 → September 2026

Grouped by chapter. 🆕 marks 2025–2026 work. Items marked *(reported)* come from secondary
sources (blogs, aggregators) rather than a primary paper; check the primary source before citing.
Read the ★ items first; they are the ones the course's code follows most closely.

## Foundations (Chapters 1–7)
- ★ Vaswani et al., *Attention Is All You Need* (2017) — the Transformer.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019) — byte-level BPE, scale.
- Brown et al., *Language Models are Few-Shot Learners* (GPT-3, 2020).
- Su et al., *RoFormer: Rotary Position Embedding* (2021).
- Shazeer, *GLU Variants Improve Transformer* (2020) — SwiGLU.
- Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019).
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (2023).
- DeepSeek-AI, *DeepSeek-V2* (2024) — Multi-head Latent Attention (MLA).
- Dao et al., *FlashAttention* (2022), *FlashAttention-2* (2023); Shah et al., *FlashAttention-3* (2024).
- Holtzman et al., *The Curious Case of Neural Text Degeneration* (2019) — nucleus sampling.
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2022).
- ★ Karpathy, *nanoGPT* / *Let's build GPT* (2023) and *nanochat* (Oct 2025, https://github.com/karpathy/nanochat).
- 🆕 Pagnoni et al., *Byte Latent Transformer* (2024/25) — tokenizer-free models.

## Pretraining data (Chapter 8)
- Raffel et al., *T5 / C4* (2020) — the C4 heuristics.
- Rae et al., *Gopher* (2021) — the Gopher quality rules.
- Lee et al., *Deduplicating Training Data Makes Language Models Better* (2021).
- ★ Penedo et al., *The FineWeb Datasets* (2024) — FineWeb / FineWeb-Edu.
- Li et al., *DataComp-LM (DCLM)* (2024).
- ★ Su et al., *Nemotron-CC* (2024, https://arxiv.org/abs/2412.02595) — synthetic rephrasing at scale.
- 🆕 *FineInstructions* (Jan 2026, https://arxiv.org/abs/2601.22146).
- 🆕 *Data Darwinism II: DataEvolve* (2026, https://arxiv.org/abs/2603.14420).
- 🆕 *How Can We Synthesize High-Quality Pretraining Data?* (2026, https://arxiv.org/abs/2604.13977).

## Scaling and training (Chapters 9–11)
- Kaplan et al., *Scaling Laws for Neural Language Models* (2020).
- ★ Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022).
- Yang et al., *Tensor Programs V: μP* (2022) — hyperparameter transfer.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW, 2017).
- ★ Jordan et al., *Muon* (2024, https://github.com/KellerJordan/modded-nanogpt); Liu et al., *Muon is Scalable for LLM Training* (Moonshot, 2025).
- 🆕 *SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales* (July 2026, https://arxiv.org/abs/2607.20548).
- Hägele et al., *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* (2024) — WSD schedules.
- Rajbhandari et al., *ZeRO* (2019); Shoeybi et al., *Megatron-LM* (2019); Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters* (2021).
- DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024) — FP8 training, MTP, MoE at scale.
- 🆕 *Pretraining Large Language Models with MXFP4 on Native FP4 Hardware* (2026, https://arxiv.org/abs/2605.09825).
- 🆕 NVIDIA NVFP4 pretraining recipe *(reported, 2025–26)*.

## Architectures (Chapter 12)
- Shazeer et al., *Outrageously Large Neural Networks* (2017) — MoE; Fedus et al., *Switch Transformers* (2021).
- Dai et al., *DeepSeekMoE* (2024).
- Gloeckle et al., *Better & Faster LLMs via Multi-token Prediction* (2024).
- Gu & Dao, *Mamba* (2023); Dao & Gu, *Mamba-2* (2024); NVIDIA, *Nemotron-H* (2025, https://arxiv.org/abs/2504.03624).
- Yuan et al., *Native Sparse Attention (NSA)* (2025); DeepSeek-AI, *DeepSeek-V3.2* (DSA, 2025).
- 🆕 ★ DeepSeek-AI, *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence* (2026, https://arxiv.org/abs/2606.19348).
- 🆕 Nie et al., *LLaDA* (2025); *LLaDA-MoE* (2025, https://arxiv.org/abs/2509.24389); *SparseD* (ICLR 2026) — diffusion LMs.
- 🆕 Open-weight landscape summaries *(reported)*: https://wavect.io/blog/open-weight-llm-comparison-2026/

## Post-training (Chapters 14–17, 20, 22)
- ★ Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT, 2022).
- Wang et al., *Self-Instruct* (2022); Xu et al., *WizardLM / Evol-Instruct* (2023).
- Hu et al., *LoRA* (2021).
- ★ Rafailov et al., *Direct Preference Optimization* (2023).
- Azar et al., *IPO* (2023); Ethayarajh et al., *KTO* (2024); Meng et al., *SimPO* (2024); Hong et al., *ORPO* (2024).
- Bai et al., *Constitutional AI* (2022); Lee et al., *RLAIF* (2023).
- Gao et al., *Scaling Laws for Reward Model Overoptimization* (2022) — reward hacking.
- 🆕 ★ Thinking Machines, *On-Policy Distillation* (Oct 2025, https://thinkingmachines.ai/blog/on-policy-distillation/).
- 🆕 *Rethinking On-Policy Distillation* (2026, https://arxiv.org/abs/2604.13016); *Lightning OPD* (https://arxiv.org/abs/2604.13010); *Uni-OPD* (https://arxiv.org/abs/2605.03677); OPD survey (https://arxiv.org/abs/2606.22793).
- 🆕 *OpenRubrics* (ACL 2026, https://aclanthology.org/2026.acl-long.791/); *Many Voices, One Reward* (2026, https://arxiv.org/abs/2607.01830).
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023).

## Reinforcement learning (Chapters 18–19, 21)
- Williams, *REINFORCE* (1992); Sutton & Barto, *Reinforcement Learning: An Introduction* (2018) — chapter 13 only.
- ★ Schulman et al., *Proximal Policy Optimization* (2017); Schulman, *Approximating KL Divergence* (blog, 2020).
- ★ Shao et al., *DeepSeekMath* (2024) — GRPO; DeepSeek-AI, *DeepSeek-R1* (2025) — RLVR at scale.
- ★ Yu et al., *DAPO* (2025); Liu et al., *Understanding R1-Zero-Like Training* (Dr. GRPO, 2025); Zheng et al., *GSPO* (Qwen, 2025); MiniMax, *CISPO* (2025).
- 🆕 *Geometric-Mean Policy Optimization* (2025, https://arxiv.org/abs/2507.20673); *Train Less, Learn More* adaptive rollouts (2026, https://arxiv.org/abs/2602.14338).
- 🆕 Surveys *(reported)*: https://www.turingpost.com/p/reasoning-rl-in-2026
- 🆕 Agentic RL: *AgentRL* (https://arxiv.org/abs/2510.04206), *SkyRL-Agent* (https://arxiv.org/abs/2511.16108), *SENTINEL* (https://arxiv.org/abs/2606.12908), *RODS* (https://arxiv.org/abs/2606.19047), verl agentic RL docs (https://verl.readthedocs.io/en/latest/start/agentic_rl.html), Wolfe, *Agentic RL: Frameworks and Best Practices* (https://cameronrwolfe.substack.com/p/agentic-rl).
- 🆕 Test-time compute: *ParaThinker* (https://arxiv.org/abs/2509.04475), *Parallel Test-Time Scaling for Latent Reasoning* (https://arxiv.org/abs/2510.07745), *Fork-Think with Confidence* (https://arxiv.org/abs/2606.31484).

## Evaluation (Chapter 23)
- Jimenez et al., *SWE-bench* (2023); OpenAI, *SWE-bench Verified* (2024).
- 🆕 *Terminal-Bench* (2025–26); *Humanity's Last Exam* (2025); ARC Prize, *ARC-AGI-3* (2026).
- 🆕 *BenchJack* (https://arxiv.org/abs/2605.12673); *Coding Benchmarks Are Misaligned with Agentic SE* (https://arxiv.org/abs/2606.17799); *Benchmark Health Index* (https://arxiv.org/abs/2602.11674); *Cross-Context Verification* of contamination (https://arxiv.org/abs/2603.21454).

## Agents and harnesses (Chapters 24–28)
- ★ Anthropic, *Building effective agents* (Dec 2024); *Effective context engineering for AI agents* (Sept 2025); *Effective harnesses for long-running agents* (Nov 2025).
- 🆕 Three-agent (planner / generator / evaluator) harness *(reported, InfoQ Apr 2026)*: https://www.infoq.com/news/2026/04/anthropic-three-agent-harness-ai/
- 🆕 Osmani, *Long-running agents* (https://addyosmani.com/blog/long-running-agents/).
- ★ Model Context Protocol specification (2025-11-25 revision, https://modelcontextprotocol.io); Google, *Agent2Agent protocol* (2025; v1.0 Apr 2026 *(reported)*).
- 🆕 *Security Threat Modeling for Emerging AI-Agent Protocols* (https://arxiv.org/abs/2602.11327); *Governance Gaps in Agent Interoperability Protocols* (https://arxiv.org/abs/2606.31498).
- 🆕 Context rot: *Diagnosing and Mitigating Context Rot in Long-horizon Search* (https://arxiv.org/abs/2606.29718); *LOCA-bench* (https://arxiv.org/abs/2602.07962).
- Yao et al., *ReAct* (2022); Schick et al., *Toolformer* (2023).

## Staying current

Read primary sources: arXiv listings for cs.CL and cs.LG, the technical reports that accompany
open-weight releases (DeepSeek, Qwen, Kimi, GLM, Llama, Gemma, Mistral), and the engineering blogs of
Anthropic, OpenAI, Google DeepMind, Meta and Thinking Machines. Aggregators are useful for discovery
and unreliable for numbers.
