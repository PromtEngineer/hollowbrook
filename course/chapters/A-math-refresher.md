# Appendix A: Math refresher

**Read this when a chapter uses a symbol you do not recognise.** Nothing here needs more than
secondary-school algebra. Every idea is introduced with a picture in words first.

---

## A.1 Vectors: lists of numbers with a direction

A **vector** is an ordered list of numbers, e.g. `v = [2, 1]`. Picture it as an arrow from the
origin to the point (2, 1). In this course a vector usually stands for *a token* (Chapter 3)
or *a position in the residual stream* (Chapter 6), and it has hundreds of entries, not two.

- **Length (norm)**: `‖v‖ = sqrt(2² + 1²) = sqrt(5) ≈ 2.24`. Read this as: how long the arrow is.
- **Adding** two vectors adds entry by entry: `[2, 1] + [1, 3] = [3, 4]`. Picture: put the second
  arrow's tail on the first arrow's head.
- **Scaling** multiplies every entry: `2 · [2, 1] = [4, 2]`. Same direction, twice as long.

### The dot product — the single most important operation in this course

```
a · b = a₁b₁ + a₂b₂ + … + aₙbₙ
```

Read this as: multiply matching entries and add them up. Geometrically:

```
a · b = ‖a‖ ‖b‖ cos(angle between a and b)
```

So the dot product is large and positive when two arrows point the same way, zero when they are
perpendicular, and negative when they point in opposite directions. **Attention scores (Chapter 5)
are dot products**: "how much does this query point in the same direction as that key?"

**Cosine similarity** removes the lengths: `cos = (a · b) / (‖a‖ ‖b‖)`, a number between −1 and 1.
Use it when you care about direction only (comparing embeddings, Chapter 3).

## A.2 Matrices: tables of numbers that transform vectors

A **matrix** is a grid of numbers with `rows × columns` entries. A `3 × 2` matrix has 3 rows and
2 columns. Multiplying a matrix by a vector produces a new vector: each output entry is the dot
product of one *row* with the input vector.

```
W = [[1, 0],        x = [2, 1]        W x = [1·2 + 0·1,  0·2 + 1·1,  1·2 + 1·1] = [2, 1, 3]
     [0, 1],
     [1, 1]]
```

Read this as: a matrix is a machine that turns a 2-entry vector into a 3-entry vector, and the
numbers inside decide *how*. **Every linear layer in a neural network is exactly this**
(`nn.Linear` in PyTorch): output = W · input (+ bias). Training changes the numbers in W.

**Matrix × matrix** (`A @ B` in Python) does this for many vectors at once: the columns of `B`
are inputs, the columns of the result are outputs. Shapes must agree: `(n × k) @ (k × m) = (n × m)`.
Reading shapes is a skill: in `(B, T, d) @ (d, V)` we turn `d`-dimensional token vectors into
`V`-dimensional score vectors, for every token in every batch row.

**Transpose** `Aᵀ` flips rows and columns. `Q @ Kᵀ` in attention lines up every query with every key.

## A.3 Functions you will meet

| Function | Formula | Picture in words | Where it appears |
|---|---|---|---|
| exp | `e^x` | grows fast, always positive | softmax |
| log (natural) | `ln x` | undoes exp; turns products into sums | loss, log-probabilities |
| sigmoid | `σ(x) = 1 / (1 + e^(−x))` | squashes any number into (0, 1) | Bradley–Terry, DPO |
| tanh | `(e^x − e^(−x)) / (e^x + e^(−x))` | squashes into (−1, 1) | micro-autograd demo |
| ReLU | `max(0, x)` | zero for negatives, identity for positives | classic MLPs |
| SiLU | `x · σ(x)` | smooth ReLU | SwiGLU MLP (Chapter 6) |
| softmax | `e^(xᵢ) / Σⱼ e^(xⱼ)` | turns scores into probabilities that sum to 1 | attention, output layer |

**Why logs everywhere?** Probabilities of sequences are products of many small numbers
(Chapter 1): `0.1 × 0.2 × 0.05 × …` underflows to zero in a computer. `log` turns the product into
a sum of manageable negative numbers: `log 0.1 + log 0.2 + …`. Cross-entropy loss is just
`−log(probability the model gave the correct token)`.

## A.4 Probability in five lines

- A **probability distribution** over a finite set assigns each outcome a number in [0, 1], all summing to 1.
  A language model's output is a distribution over the vocabulary (Chapter 1).
- **Conditional probability** `P(next | context)`: the distribution *given* what came before.
- **Chain rule**: `P(w₁ w₂ w₃) = P(w₁) · P(w₂ | w₁) · P(w₃ | w₁ w₂)`. Read this as: a sentence's
  probability is the product of each word's probability given everything before it. That is
  literally what a language model computes.
- **Expectation** `E[X] = Σ x · P(x)`: the average value you would see over many samples. RL objectives
  (Chapter 18) are expectations of reward.
- **Entropy** `H = −Σ P(x) log P(x)`: how spread out a distribution is. Zero when one outcome is
  certain; maximal (`log V`) when all `V` outcomes are equally likely. Perplexity is `e^H` — "the
  effective number of equally likely choices".

**KL divergence** `KL(P ‖ Q) = Σ P(x) log(P(x) / Q(x))` measures how different Q is from P (it is
zero only when they are equal, and it is not symmetric). In RL for LLMs (Chapters 18–20) it keeps
the trained policy from drifting too far from a reference model.

## A.5 Derivatives: slopes, and the chain rule

The **derivative** of a function at a point is its slope there: `f'(x) = lim (f(x + h) − f(x)) / h`.
Read this as: if I nudge `x` a tiny bit, how much does `f` change, per unit of nudge?

- `d/dx (x²) = 2x`, `d/dx (eˣ) = eˣ`, `d/dx (ln x) = 1/x`, `d/dx (c · x) = c`.
- **Chain rule**: if `y = f(g(x))` then `dy/dx = f'(g(x)) · g'(x)`. Read this as: slopes multiply along
  a chain. Backpropagation (Chapter 4) is the chain rule applied along every path in a computation
  graph, adding up contributions when paths merge.
- **Gradient**: for a function of many inputs, the vector of all partial derivatives
  `∇f = [∂f/∂x₁, …, ∂f/∂xₙ]`. It points *uphill*; training steps go `−∇f` (downhill).

A **partial derivative** `∂f/∂x₁` is the ordinary derivative with respect to `x₁` while holding the
other inputs fixed.

## A.6 Notation cheat-sheet used in this course

| Symbol | Meaning |
|---|---|
| `B, T, d, h, V` | batch size, sequence length, model width, number of heads, vocabulary size |
| `N, D, C` | parameters, training tokens, compute in FLOPs (`C ≈ 6ND`) |
| `x` | the residual stream, shape `(B, T, d)` |
| `θ` | all the model's parameters together |
| `π_θ(y | x)` | the policy: probability the model assigns to response `y` given prompt `x` |
| `π_ref` | a frozen reference copy of the model |
| `L` | a loss (lower is better); `J` an objective (higher is better) |
| `E[·]` | expectation (average over samples) |
| `∇_θ` | gradient with respect to the parameters |
| `σ` | sigmoid; `β` a temperature-like constant in DPO/RL; `ε` a small clipping threshold |
| `‖v‖` | the length of vector `v` |

## A.7 Three worked calculations

**Softmax with temperature.** Scores `[2, 1, 0]`, temperature 1: `e² = 7.39, e¹ = 2.72, e⁰ = 1`,
sum 11.11, probabilities `[0.665, 0.245, 0.090]`. Temperature 0.5 divides scores first: `[4, 2, 0]`
→ `[0.867, 0.117, 0.016]` — sharper. Temperature 2: `[1, 0.5, 0]` → `[0.506, 0.307, 0.186]` — flatter.

**Cross-entropy of one prediction.** The model gives the correct next token probability 0.25.
Loss = `−ln 0.25 = 1.386`. If it gave 0.9, loss = 0.105. Averaged over many tokens, `e^(mean loss)`
is the perplexity.

**One gradient step.** Loss `L(w) = (w − 3)²`, start at `w = 0`, learning rate 0.1.
`dL/dw = 2(w − 3) = −6`. Step: `w ← 0 − 0.1 · (−6) = 0.6`. Repeat: the gradient shrinks as `w`
approaches 3. This is Chapter 4 in one line.
