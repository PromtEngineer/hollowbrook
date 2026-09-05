"""Lab 11: data-parallel training on a CPU with torch.distributed.

Two worker processes each hold a *full copy* of a nano TinyLM. Every step, both draw
the same global batch from a shared random generator, keep only their own rows
(rank r takes rows r::2), compute gradients on those rows, and then average the
gradients with an all-reduce. Because the average of two half-batch gradients is
exactly the full-batch gradient, the two-process run must end with the same weights
as one process training on the whole batch. We check that numerically.

Run:  python3 labs/lab11_data_parallel.py            (quick, < 1 min)
      python3 labs/lab11_data_parallel.py --full     (a few minutes)
"""
from _common import setup, check, banner, section, savefig, done, plt

import os
import sys
import time
import tempfile
import platform

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llm.config import preset
from llm.model import TinyLM
from llm.train import get_batch
from llm.pipeline import get_tokenizer, get_tokens

args = setup("Lab 11: data-parallel training with torch.distributed")

WORLD = 2                                   # number of workers ("GPUs")
STEPS = 8 if args.quick else 40
B_LOCAL = 8 if args.quick else 16           # rows per worker; the global batch is WORLD * B_LOCAL
T = 64 if args.quick else 128
LR = 1e-3
BYTES_PER_GRAD = 4                          # float32 gradients on CPU


# ------------------------------------------------------------------ the loop
def make_model(vocab_size: int, seed: int) -> TinyLM:
    """Every worker must start from *identical* weights, so seed before construction."""
    torch.manual_seed(seed)
    return TinyLM(preset("nano", vocab_size=vocab_size))


def train_shard(model, tokens, rank, world, steps, b_local, seq_len, seed, lr, allreduce_fn=None):
    """The data-parallel training loop for one worker.

    allreduce_fn(flat_tensor) must SUM the tensor across all workers in place.
    With allreduce_fn=None this is ordinary single-process training on this rank's rows.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)            # the SAME generator on every rank
    params = [p for p in model.parameters()]
    losses, comm_seconds = [], 0.0
    for step in range(steps):
        x, y = get_batch(tokens, b_local * world, seq_len, g)   # everyone draws the same global batch
        x, y = x[rank::world], y[rank::world]                   # (b_local, T): this rank's rows
        _, loss = model(x, y)                                   # loss = mean over this rank's rows
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if allreduce_fn is not None:
            # Real DDP flattens gradients into buckets so each all-reduce is one big message.
            flat = torch.cat([p.grad.reshape(-1) for p in params])   # (n_params,)
            t0 = time.perf_counter()
            allreduce_fn(flat)                                  # sum over workers
            comm_seconds += time.perf_counter() - t0
            flat.div_(world)                                    # sum / world = mean over the global batch
            offset = 0
            for p in params:
                n = p.numel()
                p.grad.copy_(flat[offset:offset + n].view_as(p.grad))
                offset += n
        torch.nn.utils.clip_grad_norm_(params, 1.0)             # same clip on the same (averaged) grads
        opt.step()
        losses.append(loss.item())
    return losses, comm_seconds


def dp_worker(rank, world, init_file, out_dir, vocab_size, tokens, steps, b_local, seq_len, seed, lr):
    """Entry point of each worker process."""
    torch.set_num_threads(max(1, torch.get_num_threads() // world))   # share the cores fairly
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    model = make_model(vocab_size, seed)

    def allreduce(t):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    t0 = time.perf_counter()
    losses, comm = train_shard(model, tokens, rank, world, steps, b_local, seq_len, seed, lr, allreduce)
    wall = time.perf_counter() - t0
    torch.save({"state_dict": model.state_dict(), "losses": losses, "wall": wall, "comm": comm},
               os.path.join(out_dir, f"rank{rank}.pt"))
    dist.barrier()
    dist.destroy_process_group()


def simulate_dp(world, vocab_size, tokens, steps, b_local, seq_len, seed, lr):
    """Fallback: the same algorithm with `world` model copies living in ONE process.
    The 'all-reduce' is a plain sum over the copies' gradients."""
    models = [make_model(vocab_size, seed) for _ in range(world)]
    opts = [torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0) for m in models]
    gens = [torch.Generator().manual_seed(seed) for _ in range(world)]
    losses = [[] for _ in range(world)]
    t0 = time.perf_counter()
    for step in range(steps):
        for r, m in enumerate(models):
            x, y = get_batch(tokens, b_local * world, seq_len, gens[r])
            x, y = x[r::world], y[r::world]
            _, loss = m(x, y)
            opts[r].zero_grad(set_to_none=True)
            loss.backward()
            losses[r].append(loss.item())
        for grads in zip(*[list(p.grad for p in m.parameters()) for m in models]):
            total = sum(grads) / world                          # the "all-reduce"
            for g_ in grads:
                g_.copy_(total)
        for m, o in zip(models, opts):
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            o.step()
    wall = time.perf_counter() - t0
    return [{"state_dict": m.state_dict(), "losses": losses[r], "wall": wall, "comm": 0.0}
            for r, m in enumerate(models)]


def run_distributed(world, vocab_size, tokens, steps, b_local, seq_len, seed, lr, timeout=900):
    """Launch `world` processes; return their results or None if anything fails."""
    tmp = tempfile.mkdtemp(prefix="lab11_")
    init_file = os.path.join(tmp, "store")
    try:
        ctx = mp.get_context("spawn")                           # fresh interpreters: safest everywhere
        procs = [ctx.Process(target=dp_worker,
                             args=(r, world, init_file, tmp, vocab_size, tokens, steps, b_local, seq_len, seed, lr))
                 for r in range(world)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout)
        if any(p.is_alive() or p.exitcode != 0 for p in procs):
            for p in procs:
                if p.is_alive():
                    p.terminate()
            print("   worker processes did not finish cleanly:", [p.exitcode for p in procs])
            return None
        return [torch.load(os.path.join(tmp, f"rank{r}.pt")) for r in range(world)]
    except Exception as e:  # noqa: BLE001
        print(f"   torch.distributed launch failed ({type(e).__name__}: {e})")
        return None


def max_param_diff(sd_a, sd_b) -> float:
    return max((sd_a[k].float() - sd_b[k].float()).abs().max().item() for k in sd_a)


# ------------------------------------------------------------------ memory arithmetic
def zero_memory_gb(n_params: float, world: int, stage: int) -> float:
    """Model + optimizer memory per GPU for mixed-precision AdamW (the ZeRO paper's accounting).

    bf16 weights (2 B) + bf16 grads (2 B) + fp32 master weights, Adam m and v (12 B) = 16 B / param.
    stage 0: everything replicated.        stage 1: optimizer states (12 B) sharded.
    stage 2: + gradients sharded.          stage 3: + weights sharded (FSDP).
    """
    if stage == 0:
        per = 16.0
    elif stage == 1:
        per = 4.0 + 12.0 / world
    elif stage == 2:
        per = 2.0 + 14.0 / world
    else:
        per = 16.0 / world
    return n_params * per / 1e9


if __name__ == "__main__":
    tok = get_tokenizer()
    train_tokens, _ = get_tokens(tok)
    n_params = make_model(tok.vocab_size, args.seed).num_params(non_embedding=False)
    grad_mb = n_params * BYTES_PER_GRAD / 1e6
    print(f"model: nano TinyLM, {n_params:,} params -> {grad_mb:.2f} MB of float32 gradients")
    print(f"world={WORLD} workers, {STEPS} steps, global batch {WORLD * B_LOCAL} x {T} tokens "
          f"({B_LOCAL} rows per worker)")

    # ---------------------------------------------------------------- 1. two workers
    section(f"1. {WORLD} processes, gradients averaged with all-reduce (gloo backend)")
    print(f"   platform {platform.system()}, torch {torch.__version__}, "
          f"gloo available: {dist.is_gloo_available()}")
    t0 = time.perf_counter()
    results = run_distributed(WORLD, tok.vocab_size, train_tokens, STEPS, B_LOCAL, T, args.seed, LR)
    launch_wall = time.perf_counter() - t0
    mode = "torch.distributed"
    if results is None:
        print("   -> falling back to a simulated 2-worker loop in one process (same maths, no sockets)")
        mode = "simulated"
        results = simulate_dp(WORLD, tok.vocab_size, train_tokens, STEPS, B_LOCAL, T, args.seed, LR)
        launch_wall = results[0]["wall"]
    for r, res in enumerate(results):
        print(f"   rank {r}: first loss {res['losses'][0]:.4f} -> last loss {res['losses'][-1]:.4f} "
              f"| train wall {res['wall']:.1f}s | time inside all-reduce {res['comm']:.2f}s")
    print(f"   whole launch incl. process start-up: {launch_wall:.1f}s  ({mode})")
    rank_diff = max_param_diff(results[0]["state_dict"], results[1]["state_dict"])
    check(rank_diff < 1e-6, f"both workers hold identical weights after training (max |diff| = {rank_diff:.2e})")

    # ---------------------------------------------------------------- 2. one process, whole batch
    section("2. reference: ONE process, the whole global batch every step")
    torch.set_num_threads(torch.get_num_threads())
    ref = make_model(tok.vocab_size, args.seed)
    t0 = time.perf_counter()
    ref_losses, _ = train_shard(ref, train_tokens, rank=0, world=1, steps=STEPS,
                                b_local=WORLD * B_LOCAL, seq_len=T, seed=args.seed, lr=LR)
    ref_wall = time.perf_counter() - t0
    print(f"   first loss {ref_losses[0]:.4f} -> last loss {ref_losses[-1]:.4f} | wall {ref_wall:.1f}s")

    # the DP loss at a step is the mean of the two ranks' local losses
    dp_mean = [(a + b) / 2 for a, b in zip(results[0]["losses"], results[1]["losses"])]
    loss_gap = max(abs(a - b) for a, b in zip(dp_mean, ref_losses))
    print(f"   step | rank0 loss | rank1 loss | mean(ranks) | single-process")
    for s in range(0, STEPS, max(1, STEPS // 4)):
        print(f"   {s:4d} | {results[0]['losses'][s]:10.4f} | {results[1]['losses'][s]:10.4f} | "
              f"{dp_mean[s]:11.4f} | {ref_losses[s]:14.4f}")
    check(loss_gap < 1e-3, f"mean of the ranks' losses == single-process loss at every step (max gap {loss_gap:.1e})")

    dp_diff = max_param_diff(results[0]["state_dict"], ref.state_dict())
    print(f"   max |w_dp - w_single| = {dp_diff:.2e}")

    # ---------------------------------------------------------------- 3. control: one worker, no averaging
    section("3. control: rank 0 alone on its half of the data, NO all-reduce")
    solo = make_model(tok.vocab_size, args.seed)
    solo_losses, _ = train_shard(solo, train_tokens, rank=0, world=WORLD, steps=STEPS,
                                 b_local=B_LOCAL, seq_len=T, seed=args.seed, lr=LR)
    solo_diff = max_param_diff(solo.state_dict(), ref.state_dict())
    print(f"   max |w_solo - w_single| = {solo_diff:.2e}  (this is how different a *different* run looks)")
    check(dp_diff < 1e-4, f"data-parallel weights match the single-process weights ({dp_diff:.1e} < 1e-4)")
    check(dp_diff < solo_diff / 100, f"...and are >100x closer than the no-all-reduce control ({solo_diff:.1e})")

    # ---------------------------------------------------------------- 4. communication volume
    section("4. what went over the wire")
    ring_per_device = 2 * (WORLD - 1) / WORLD * grad_mb
    print(f"   gradient vector: {n_params:,} floats = {grad_mb:.2f} MB")
    print(f"   ring all-reduce sends+receives 2(N-1)/N x that per device per step = {ring_per_device:.2f} MB (N={WORLD})")
    print(f"   over {STEPS} steps: {ring_per_device * STEPS:.1f} MB per device")
    for n_big, name in [(7e9, "7B"), (70e9, "70B"), (1.6e12, "1.6T (DeepSeek-V4-class)")]:
        print(f"   {name:>26}: {n_big * 2 / 1e9:8.0f} GB of bf16 gradients per step -> "
              f"{2 * 7 / 8 * n_big * 2 / 1e9:6.0f} GB per device on an 8-GPU ring")
    comm_frac = results[0]["comm"] / max(results[0]["wall"], 1e-9)
    print(f"   this run: {100 * comm_frac:.1f}% of worker wall-clock was spent inside all-reduce")
    speedup = ref_wall / max(results[0]["wall"], 1e-9)
    print(f"   speed: single-process {ref_wall:.1f}s vs data-parallel {results[0]['wall']:.1f}s -> {speedup:.2f}x "
          f"(ideal {WORLD}x; CPUs share memory bandwidth, so expect less)")

    # ---------------------------------------------------------------- 5. ZeRO table
    section("5. memory per GPU for a 70B model under AdamW mixed precision (weights+grads+optimizer only)")
    N70 = 70e9
    print(f"   {'GPUs':>5} | {'ZeRO-0 (DDP)':>13} | {'ZeRO-1':>9} | {'ZeRO-2':>9} | {'ZeRO-3/FSDP':>12}")
    for w in [1, 8, 64, 512]:
        row = [zero_memory_gb(N70, w, s) for s in range(4)]
        print(f"   {w:>5} | {row[0]:11.0f} GB | {row[1]:6.0f} GB | {row[2]:6.0f} GB | {row[3]:9.1f} GB")
    print("   (an H100 has 80 GB; activations come on top of these numbers)")
    check(zero_memory_gb(N70, 1, 0) == 70e9 * 16 / 1e9, "16 bytes/param: 70B params need 1120 GB without sharding")

    # ---------------------------------------------------------------- figure
    fig, axes = plt().subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(results[0]["losses"], "o-", ms=3, label="rank 0 (its half)")
    ax.plot(results[1]["losses"], "s-", ms=3, label="rank 1 (its half)")
    ax.plot(dp_mean, "-", lw=2, label="mean of ranks")
    ax.plot(ref_losses, "k--", lw=1.5, label="single process, full batch")
    ax.set_xlabel("step"); ax.set_ylabel("train loss"); ax.set_title(f"data parallel ({mode})"); ax.legend()
    ax = axes[1]
    worlds = [1, 8, 64, 512]
    for s, lab in enumerate(["ZeRO-0", "ZeRO-1", "ZeRO-2", "ZeRO-3 / FSDP"]):
        ax.plot(worlds, [zero_memory_gb(N70, w, s) for w in worlds], "o-", label=lab)
    ax.axhline(80, color="red", ls=":", label="one H100 (80 GB)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("number of GPUs"); ax.set_ylabel("GB per GPU (70B, AdamW)"); ax.set_title("ZeRO sharding"); ax.legend()
    fig.tight_layout()
    savefig(fig, "lab11_data_parallel.png")
    done()
