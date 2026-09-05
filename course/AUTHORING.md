# How this course is written (style guide)

This file is the contract every chapter, lab and interactive follows. It exists so the material
feels like one voice, and so a beginner never hits an undefined word.

## The reader

Assume the reader:
- can write a Python function, a loop and a class, and has run `pip install`;
- has **not** taken a machine-learning, linear-algebra or probability course (Appendix A refreshes what is needed);
- is smart, busy, and will stop reading the moment something is unexplained.

## The five-beat chapter structure

Every chapter has exactly this skeleton, in this order:

1. **Why this matters** (3–6 sentences). The problem this chapter solves, in plain words, with one concrete example.
2. **The idea in pictures** 📐. A diagram (Mermaid or SVG) *before* any equation or code. Explain the picture in prose.
3. **The idea in code**. Small snippets (≤ 25 lines each) that build up to the library function. Every snippet is runnable as-is after the imports shown at the top of the chapter. Snippets are copied from `course/llm/` or `course/labs/` and are covered by tests.
4. **Worked example** 🧪. Point at the lab, show the expected output (numbers included), and explain what the reader should look at.
5. **Try it yourself** ✍️, **Check yourself** ✅ (5 questions with answers in `<details>` blocks), **Key takeaways** (≤ 6 bullets), **Going deeper** (3–8 references with year; 2025–2026 items marked 🆕).

Sections may have sub-sections, but the five beats never change order.

## Rules of explanation

- **Define before use.** The first time a term appears, bold it and define it in the same sentence. Add it to the glossary (Appendix C).
- **Intuition → picture → math → code.** Never the other way round.
- **One new idea per paragraph.**
- **Every equation gets a plain-English reading** directly under it ("read this as: ...").
- **Every number is checked.** If the chapter says "about 5 minutes on a laptop CPU", a lab measured it.
- **Analogies are labelled as analogies** and their limits are stated in one sentence.
- **Say what is settled and what is open.** 2026 research findings are reported as "the current evidence suggests", with a citation.
- **No hand-waving words.** Banned without explanation: "simply", "obviously", "it turns out", "magic".

## Code rules

- All snippets import from the library: `from llm.model import TinyLM` etc. Snippets must run on CPU in under a minute.
- Prefer showing the *shape* of tensors in comments: `# (batch, seq, d_model)`.
- Every lab prints a short **✅ checks passed** line at the end and saves any figure to `course/figures/generated/`.
- Labs accept `--quick` to run in under 60 seconds and `--full` for the "real" run (still minutes, not hours).

## Diagram rules

- Pipeline/flow diagrams: Mermaid (renders on GitHub). Mechanism diagrams (attention, residual stream, KV cache, MoE routing, GRPO groups): SVG files in `course/figures/`, 900 px wide, dark text on white, with a caption in the chapter.
- Every diagram is referenced from the prose ("In the figure, the arrow from...").

## Interactive rules

- Plain HTML + CSS + JS in one file, no build step, no network requests, works from `file://`.
- Opens with a two-sentence "what to try" box and has at least one **Challenge** the reader can check.
- Default state already shows something interesting.

## Multiple passes

Every chapter goes through three passes after drafting:
1. **Beginner pass** — a reviewer who pretends to know nothing flags every undefined term or leap.
2. **Accuracy pass** — every claim, citation and formula checked; every snippet executed.
3. **Consistency pass** — names, notation and links match the library and the other chapters.

Notation used everywhere: `B` batch, `T` sequence length, `d` model width (`d_model`), `h` heads,
`V` vocab size, `N` parameters, `D` training tokens, `C` compute in FLOPs, `x` the residual stream,
`π_θ` the policy (the model being trained), `π_ref` the frozen reference.
