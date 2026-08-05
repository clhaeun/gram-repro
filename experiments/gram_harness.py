"""Shared plumbing for the GRAM checkpoint verification experiments (e1-e4).

GRAM's latent transition is stochastic, so K independent forward passes on the
*same* input already give K distinct answers -- unlike TRM, which is deterministic
and needs test-time augmentation for sample diversity.  Every rollout here is
therefore plain repetition: no symmetry transform, no inverse mapping.

Provides:
  load_model()      checkpoint -> eager ACTLossHead (all_config.yaml + ema=False)
  load_puzzles()    n test puzzles sampled from the raw .npy arrays
  rollout()         K forward passes -> preds / v_logits / token accuracy
  metrics()         pass@j, majority@j, hit distribution, no-hit / majority-miss
  capture_kl(), capture_sigma()   monkeypatch probes (diag_sigma.py pattern)
  dist_*()          optional torchrun sharding across GPUs
"""
import json
import os
from collections import Counter
from contextlib import contextmanager
from math import comb

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

import pretrain as P

PAD = 0                 # dataset.json: pad_id = ignore_label_id = 0
SEQ_LEN = 81            # sudoku default only; _batch/rollout now take seq_len from inputs.shape[-1]


# --------------------------------------------------------------------------- #
# distributed (optional -- scripts also run fine as plain `python foo.py`)
# --------------------------------------------------------------------------- #
def dist_setup():
    """Returns (rank, world_size).  Only used to shard puzzles; no DDP wrapping."""
    if "LOCAL_RANK" not in os.environ:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1
    dist.init_process_group(backend="gloo")     # CPU group: we only gather results
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return dist.get_rank(), dist.get_world_size()


def dist_gather(obj, rank, world_size):
    """all_gather a picklable per-rank object; returns the list on every rank."""
    if world_size == 1:
        return [obj]
    out = [None] * world_size
    dist.all_gather_object(out, obj)
    return out


def dist_teardown():
    if dist.is_initialized():
        dist.destroy_process_group()


def shard(n, rank, world_size):
    """Contiguous slice of range(n) owned by this rank."""
    return np.array_split(np.arange(n), world_size)[rank]


# --------------------------------------------------------------------------- #
# checkpoint / data loading
# --------------------------------------------------------------------------- #
def load_model(ckpt_dir, step):
    """eval_checkpoint.py pattern: all_config.yaml + load_checkpoint, ema=False.

    Returns the EAGER ACTLossHead (unwrapping torch.compile via ._orig_mod) so the
    monkeypatches and the sigma_scale attribute take effect without retracing.
    """
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "all_config.yaml"))
    cfg.load_checkpoint = os.path.join(ckpt_dir, step)
    cfg.ema = False                  # checkpoint is already the EMA copy
    cfg.eval_save_outputs = []
    cfg.evaluators = []
    # harness 는 추론만 한다 -- optimizer param_group 은 의미가 없다.  끄지 않으면
    # pretrain.py 의 load_checkpoint 가 `load_state_dict(..., assign=True)` 로
    # Parameter 객체를 통째로 교체하면서 fp_transformer 가 붙인 `_no_weight_decay`
    # 속성이 사라지고, _adamw_param_arg 가 "마킹 0개" 로 보고 ValueError 로 즉사한다.
    # (학습은 load_checkpoint=None 이라 이 경로를 안 타서 마킹 2개가 살아 있다.)
    cfg.nodecay_marked = False
    config = P.PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))

    _, train_metadata = P.create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1)
    ts = P.init_train_state(config, train_metadata, rank=0, world_size=1)

    net = getattr(ts.model, "_orig_mod", ts.model)   # eager ACTLossHead
    net.eval()
    return net, config


def load_puzzles(config, n, seed=0):
    """n test puzzles, sampled without replacement.  Returns (inputs, labels) int64."""
    test_dir = os.path.join(config.data_paths[0], "test")
    inputs = np.load(os.path.join(test_dir, "all__inputs.npy"))
    labels = np.load(os.path.join(test_dir, "all__labels.npy"))
    idx = np.random.default_rng(seed).choice(len(inputs), size=min(n, len(inputs)), replace=False)
    return inputs[idx].astype(np.int64), labels[idx].astype(np.int64)


def _batch(inputs, labels, lo, hi, batch_size):
    """Fixed-size padded batch on GPU (tail batch padded so shapes stay static)."""
    seq_len = inputs.shape[-1]          # was hardcoded SEQ_LEN=81 (sudoku); breaks N-Queens (64)
    inp = np.zeros((batch_size, seq_len), dtype=np.int32)
    lab = np.zeros((batch_size, seq_len), dtype=np.int32)
    inp[: hi - lo] = inputs[lo:hi]
    lab[: hi - lo] = labels[lo:hi]
    return {
        "inputs": torch.from_numpy(inp).cuda(),
        "labels": torch.from_numpy(lab).cuda(),
        # test set has a single blank identifier (all__puzzle_identifiers is all-zero)
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.int32).cuda(),
    }


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
@contextmanager
def sigma_scale(net, scale):
    inner = net.model.inner
    prev = inner.sigma_scale
    inner.sigma_scale = float(scale)
    try:
        yield
    finally:
        inner.sigma_scale = prev


@contextmanager
def mu_scale(net, scale):
    """Scale the guide (mu) before sampling.  scale=0 -> "stochasticity only"
    (eps = sigma * noise, guide removed).  scale=1 (default) is a no-op wrapper
    that returns exactly what reparameterize returned before, so the normal path
    stays bit-identical.

    Paired with sigma_scale(net, 0) this gives the three ablation modes of the
    paper's Table 3b without touching models/recursive_reasoning/gram.py.
    """
    if float(scale) == 1.0:                 # no-op: do not even wrap
        yield
        return
    inner = net.model.inner
    orig = inner.reparameterize

    def patched(mu, logvar):
        return orig(mu * float(scale), logvar)

    inner.reparameterize = patched
    try:
        yield
    finally:
        inner.reparameterize = orig


@contextmanager
def capture_kl(net, sink):
    """Append per-sample KL (summed over L and D) for every kl_loss call."""
    inner = net.model.inner
    orig = inner.kl_loss

    def patched(mu_q, logvar_q, mu_p, logvar_p):
        kl = orig(mu_q, logvar_q, mu_p, logvar_p)
        sink.append(kl.detach().float().sum(-1).sum(-1).cpu().numpy())
        return kl

    inner.kl_loss = patched
    try:
        yield
    finally:
        inner.kl_loss = orig


@contextmanager
def capture_sigma(net, sink):
    """Append mean/median/p99 of the sampled std for every reparameterize call.

    Calls arrive in transition order (H_cycles per forward), so sink index % H_cycles
    identifies the transition depth.
    """
    inner = net.model.inner
    orig = inner.reparameterize

    def patched(mu, logvar):
        std = (0.5 * logvar.detach().float()).exp() * inner.sigma_scale
        flat = std.flatten()
        q = torch.quantile(flat[torch.randperm(flat.numel(), device=flat.device)[:100000]],
                           torch.tensor([0.5, 0.99], device=flat.device))
        sink.append(dict(mean=flat.mean().item(), std=flat.std().item(),
                         median=q[0].item(), p99=q[1].item()))
        return orig(mu, logvar)

    inner.reparameterize = patched
    try:
        yield
    finally:
        inner.reparameterize = orig


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def rollout(net, inputs, labels, K, sigma_scale_=1.0, mode="prior",
            batch_size=512, seed=0, verbose=True, mu_scale_=1.0):
    """K independent forward passes over the same puzzles.

    mode="prior"     : eps ~ p(.|u)     (no labels used)
    mode="posterior" : eps ~ q(.|u,y)   (y injected, sees labels)

    Selected via the explicit use_posterior kwarg on gram.py's forward, so the
    module stays in eval mode throughout: ACT halting remains deterministic
    (full halt_max_steps) and only the latent path differs between the two.

    Returns dict with
      preds     [K, N, 81] int64 argmax tokens
      v         [K, N]     float32, LPRM head at the final transition
      tok_acc   [K, N]     float32 per-sample token accuracy
    """
    assert mode in ("prior", "posterior")
    n = len(inputs)
    seq_len = inputs.shape[-1]          # was hardcoded SEQ_LEN=81 (sudoku); breaks N-Queens (64)
    preds = np.zeros((K, n, seq_len), dtype=np.int64)
    v = np.zeros((K, n), dtype=np.float32)

    with sigma_scale(net, sigma_scale_), mu_scale(net, mu_scale_):
        for k in range(K):
            torch.random.manual_seed(seed * 100003 + k)
            for lo in range(0, n, batch_size):
                hi = min(lo + batch_size, n)
                batch = _batch(inputs, labels, lo, hi, batch_size)
                with torch.device("cuda"):
                    carry = net.initial_carry(batch)

                p_final = torch.zeros(batch_size, seq_len, dtype=torch.long, device="cuda")
                v_final = torch.zeros(batch_size, dtype=torch.float32, device="cuda")
                done = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
                while True:
                    carry, _, _, out, all_finish = net(
                        carry=carry, batch=batch, return_keys=["preds", "lrpm_logits"],
                        use_posterior=(mode == "posterior"))
                    newly = carry.halted & (~done)
                    if newly.any():
                        p_final[newly] = out["preds"][newly].long()
                        v_final[newly] = out["lrpm_logits"][newly][:, -1].float()
                    done |= carry.halted
                    if all_finish:
                        break

                preds[k, lo:hi] = p_final[: hi - lo].cpu().numpy()
                v[k, lo:hi] = v_final[: hi - lo].cpu().numpy()
            if verbose:
                print(f"  rollout {mode} sigma={sigma_scale_} {k + 1}/{K}", flush=True)

    mask = labels != PAD
    tok_acc = ((preds == labels[None]) & mask[None]).sum(-1) / np.maximum(mask.sum(-1), 1)[None]
    return dict(preds=preds, v=v, tok_acc=tok_acc.astype(np.float32))


def sharded_rollout(net, inputs, labels, K, rank, world_size, **kw):
    """rollout() split puzzle-wise across ranks, reassembled in the original order.

    Every rank returns the full [K, N, ...] arrays.  With world_size == 1 this is
    exactly rollout().
    """
    idx = shard(len(inputs), rank, world_size)
    part = rollout(net, inputs[idx], labels[idx], K, **kw)
    if world_size == 1:
        return part

    out = {k: np.zeros((K, len(inputs)) + part[k].shape[2:], dtype=part[k].dtype)
           for k in ("preds", "v", "tok_acc")}
    for r_idx, r_part in dist_gather((idx, part), rank, world_size):
        for k in out:
            out[k][:, r_idx] = r_part[k]
    return out


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def pass_at_j(num_correct, k, j):
    """Unbiased pass@j given num_correct hits out of k iid samples."""
    if k - num_correct < j:
        return 1.0
    return 1.0 - comb(k - num_correct, j) / comb(k, j)


def majority_vote(preds_k):
    """preds_k [K, 81] -> (grid, votes) of the most common full grid; ties -> first seen."""
    counts = Counter(tuple(g) for g in preds_k)
    grid, votes = counts.most_common(1)[0]
    return np.array(grid), votes, counts


def metrics(preds, labels, js=(1, 2, 4, 8, 16)):
    """preds [K, N, 81], labels [N, 81] -> the e1 metric bundle."""
    K, n = preds.shape[0], preds.shape[1]
    correct = (preds == labels[None]).all(-1).T            # [N, K]
    num_correct = correct.sum(1)
    any_hit = num_correct > 0

    maj_correct = np.zeros(n, dtype=bool)
    for p in range(n):
        grid, _, _ = majority_vote(preds[:, p])
        maj_correct[p] = np.array_equal(grid, labels[p])

    rows = {}
    # always include the full budget K, so a non-default --k still gets its own row
    for j in sorted({j for j in js if j <= K} | {K}):
        pj = float(np.mean([pass_at_j(int(c), K, j) for c in num_correct]))
        mj = np.mean([np.array_equal(majority_vote(preds[:j, p])[0], labels[p]) for p in range(n)])
        rows[j] = dict(pass_at_j=pj, majority_at_j=float(mj))

    mask = labels != PAD
    return dict(
        n=n, K=K,
        single_exact=float(correct[:, 0].mean()),
        mean_sample_exact=float(correct.mean()),
        majority_exact=float(maj_correct.mean()),
        pass_at_K=float(any_hit.mean()),
        no_hit_rate=float((~any_hit).mean()),
        majority_miss_rate=float((any_hit & ~maj_correct).mean()),
        voting_headroom=float(any_hit.mean() - maj_correct.mean()),
        mean_token_acc=float((((preds == labels[None]) & mask[None]).sum(-1)
                              / np.maximum(mask.sum(-1), 1)[None]).mean()),
        by_j=rows,
        hits_histogram={str(c): int(v) for c, v in sorted(Counter(num_correct.tolist()).items())},
        _correct=correct,             # kept for e3; stripped before JSON
        _maj_correct=maj_correct,
    )


def auroc(scores, labels_bool):
    """Rank-based AUROC (sklearn is not installed in this env). Ties get mean rank."""
    y = np.asarray(labels_bool, dtype=bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(np.asarray(scores, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = np.asarray(scores, dtype=np.float64)
    for v in np.unique(s):                       # average ranks within tie groups
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_hits_histogram(hist, n, K):
    print(f"\n  hit distribution (correct samples out of {K})")
    for c, cnt in sorted(hist.items(), key=lambda kv: int(kv[0])):
        print(f"  {int(c):>3}/{K}: {cnt:>5}  {'#' * max(1, round(60 * cnt / n))}")


def save_json(obj, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    clean = {k: v for k, v in obj.items() if not k.startswith("_")}
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=default)
    print(f"\nsaved -> {path}")
    return path


def add_common_args(ap):
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", default="step_65100")
    ap.add_argument("--num-puzzles", type=int, default=1000,
                    help="name avoids torchrun's --n* abbreviation clash")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None, help="default: <ckpt-dir>/experiments")
    return ap


def out_dir_for(args):
    return args.out_dir or os.path.join(args.ckpt_dir, "experiments")
