"""Failure analysis for a TRM Sudoku checkpoint: no-hit rate and pass@k vs majority voting.

TRM inference is deterministic (argmax + ACT halting), so repeated runs on the same
input give the same answer. Sample diversity therefore comes from test-time
augmentation over the Sudoku symmetry group (digit relabeling, band/stack and
within-band row/col permutations, transpose) -- the same group the model was
trained on via shuffle_sudoku(). Predictions are mapped back to the original
frame before voting.

Reported:
  * no-hit rate: puzzles where none of the k samples is exactly correct
  * pass@k (unbiased estimator) vs majority-vote accuracy
  * "majority miss": a correct answer is among the samples but voting picks a wrong grid

Usage (1 GPU):
  python analyze_sudoku_tta.py \
    --checkpoint-dir checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku \
    --checkpoint step_65100 --num-puzzles 1000 --k 16
"""
import argparse
import json
import os
from collections import Counter
from math import comb

import numpy as np
import torch
from omegaconf import OmegaConf

import pretrain as P

# Data tokens are digit+1: 0 = PAD, 1 = blank ("0"), 2..10 = digits 1..9.
PAD = 0


def sample_transform(rng: np.random.Generator) -> dict:
    """Random element of the Sudoku symmetry group, matching shuffle_sudoku()."""
    digit_perm = np.pad(rng.permutation(np.arange(1, 10)), (1, 0))  # digit space, 0 (blank) fixed
    transpose = bool(rng.random() < 0.5)
    bands = rng.permutation(3)
    row_perm = np.concatenate([b * 3 + rng.permutation(3) for b in bands])
    stacks = rng.permutation(3)
    col_perm = np.concatenate([s * 3 + rng.permutation(3) for s in stacks])
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    # Lift the digit permutation into token space (token = digit + 1), PAD -> PAD.
    tok_perm = np.zeros(11, dtype=np.int64)
    tok_perm[0] = PAD
    for d in range(10):
        tok_perm[d + 1] = digit_perm[d] + 1
    inv_tok_perm = np.zeros(11, dtype=np.int64)
    inv_tok_perm[tok_perm] = np.arange(11)

    return dict(mapping=mapping, transpose=transpose, tok_perm=tok_perm, inv_tok_perm=inv_tok_perm)


def apply_transform(grid81: np.ndarray, t: dict) -> np.ndarray:
    """Original frame -> augmented frame. grid81 is a length-81 token array."""
    x = grid81.reshape(9, 9)
    if t["transpose"]:
        x = x.T
    x = x.flatten()[t["mapping"]]
    return t["tok_perm"][x]


def invert_transform(grid81: np.ndarray, t: dict) -> np.ndarray:
    """Augmented frame -> original frame (exact inverse of apply_transform)."""
    x = t["inv_tok_perm"][grid81]
    out = np.empty(81, dtype=x.dtype)
    out[t["mapping"]] = x  # forward was out_flat = in_flat[mapping]
    x = out.reshape(9, 9)
    if t["transpose"]:
        x = x.T
    return x.flatten()


def pass_at_j(num_correct: int, k: int, j: int) -> float:
    """Unbiased pass@j given num_correct hits out of k iid samples."""
    if k - num_correct < j:
        return 1.0
    return 1.0 - comb(k - num_correct, j) / comb(k, j)


@torch.inference_mode()
def predict(model, inputs: np.ndarray, labels: np.ndarray, batch_size: int) -> np.ndarray:
    """Run the ACT loop to completion; return argmax token predictions (N, 81)."""
    n = len(inputs)
    out = np.zeros((n, 81), dtype=np.int64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bs = end - start
        # Pad the tail batch to a fixed size so torch.compile doesn't retrace.
        inp = np.zeros((batch_size, 81), dtype=np.int32)
        lab = np.zeros((batch_size, 81), dtype=np.int32)
        inp[:bs] = inputs[start:end]
        lab[:bs] = labels[start:end]

        batch = {
            "inputs": torch.from_numpy(inp).cuda(),
            "labels": torch.from_numpy(lab).cuda(),
            "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.int32).cuda(),
        }
        with torch.device("cuda"):
            carry = model.initial_carry(batch)

        preds_final = torch.zeros(batch_size, 81, dtype=torch.long, device="cuda")
        done = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
        while True:
            carry, _, _, preds, all_finish = model(carry=carry, batch=batch, return_keys=["preds"])
            newly = carry.halted & (~done)
            if newly.any():
                preds_final[newly] = preds["preds"][newly].long()
            done |= carry.halted
            if all_finish:
                break

        out[start:end] = preds_final[:bs].cpu().numpy()
        print(f"  predicted {end}/{n}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--checkpoint", default="step_65100")
    ap.add_argument("--num-puzzles", type=int, default=1000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    cfg = OmegaConf.load(os.path.join(args.checkpoint_dir, "all_config.yaml"))
    cfg.load_checkpoint = os.path.join(args.checkpoint_dir, args.checkpoint)
    cfg.ema = False  # checkpoint is already the EMA copy
    cfg.eval_save_outputs = []
    cfg.evaluators = []
    config = P.PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))
    torch.random.manual_seed(config.seed)

    test_dir = os.path.join(config.data_paths[0], "test")
    all_inputs = np.load(os.path.join(test_dir, "all__inputs.npy"))
    all_labels = np.load(os.path.join(test_dir, "all__labels.npy"))
    total = len(all_inputs)

    idx = rng.choice(total, size=min(args.num_puzzles, total), replace=False)
    inputs, labels = all_inputs[idx].astype(np.int64), all_labels[idx].astype(np.int64)
    n = len(inputs)
    print(f"test puzzles: {total:,} | analyzing {n:,} | k={args.k}\n")

    # Build model (create_model loads the checkpoint on rank 0).
    _, train_metadata = P.create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1)
    train_state = P.init_train_state(config, train_metadata, rank=0, world_size=1)
    model = train_state.model
    model.eval()

    # ---- identity (no augmentation) baseline: should reproduce the reported eval ----
    print("[1/2] identity pass (unaugmented baseline)")
    ident_pred = predict(model, inputs, labels, args.batch_size)
    ident_correct = (ident_pred == labels).all(axis=1)

    # ---- k iid random augmentations per puzzle ----
    print(f"\n[2/2] {args.k} augmented samples per puzzle")
    transforms = [[sample_transform(rng) for _ in range(args.k)] for _ in range(n)]

    # Sanity check: invert(apply(x)) == x on the solutions.
    for p in range(min(n, 32)):
        for t in transforms[p]:
            assert np.array_equal(invert_transform(apply_transform(labels[p], t), t), labels[p])

    aug_inputs = np.stack([apply_transform(inputs[p], transforms[p][j])
                           for p in range(n) for j in range(args.k)])
    aug_labels = np.stack([apply_transform(labels[p], transforms[p][j])
                           for p in range(n) for j in range(args.k)])
    aug_pred = predict(model, aug_inputs, aug_labels, args.batch_size)

    # Map predictions back to the original frame.
    back = np.stack([invert_transform(aug_pred[p * args.k + j], transforms[p][j])
                     for p in range(n) for j in range(args.k)]).reshape(n, args.k, 81)

    correct = (back == labels[:, None, :]).all(axis=2)          # (n, k)
    num_correct = correct.sum(axis=1)                            # hits per puzzle
    any_hit = num_correct > 0

    # Majority vote over full grids, in the original frame.
    maj_correct = np.zeros(n, dtype=bool)
    maj_votes, correct_votes = np.zeros(n, int), np.zeros(n, int)
    for p in range(n):
        counts = Counter(tuple(g) for g in back[p])
        grid, votes = counts.most_common(1)[0]               # ties -> first seen
        maj_correct[p] = np.array_equal(np.array(grid), labels[p])
        maj_votes[p] = votes
        correct_votes[p] = counts.get(tuple(labels[p]), 0)

    no_hit = ~any_hit
    maj_miss = any_hit & ~maj_correct        # answer was reachable, voting threw it away
    maj_save = ~ident_correct & maj_correct  # voting rescued a puzzle the single pass got wrong

    cell_acc_nohit = (back[no_hit] == labels[no_hit][:, None, :]).mean() if no_hit.any() else float("nan")

    print("\n" + "=" * 66)
    print(f"FAILURE ANALYSIS  ckpt={args.checkpoint}  n={n}  k={args.k}")
    print("=" * 66)
    print(f"  identity (1 sample, no aug) exact acc : {ident_correct.mean():.4f}")
    print(f"  mean per-sample exact acc (augmented) : {correct.mean():.4f}")
    print(f"  majority-vote exact acc (k={args.k:2d})        : {maj_correct.mean():.4f}")
    print(f"  pass@{args.k:<2d} (>=1 correct sample)         : {any_hit.mean():.4f}")
    print()
    print(f"  NO-HIT rate (never correct in {args.k} tries) : {no_hit.mean():.4f}  ({no_hit.sum()}/{n})")
    print(f"  majority MISS (hit exists, vote wrong)  : {maj_miss.mean():.4f}  ({maj_miss.sum()}/{n})")
    print(f"  majority RESCUE (identity wrong -> ok)  : {maj_save.mean():.4f}  ({maj_save.sum()}/{n})")
    print()
    print(f"  headroom of voting over pass@k          : {any_hit.mean() - maj_correct.mean():.4f}")
    print(f"  mean cell acc on no-hit puzzles         : {cell_acc_nohit:.4f}")
    if any_hit.any():
        print(f"  mean hits/k on solvable puzzles         : {num_correct[any_hit].mean() / args.k:.4f}")

    print("\n  pass@j (unbiased) vs majority@j (prefix of samples)")
    print(f"  {'j':>3}  {'pass@j':>8}  {'majority@j':>11}")
    rows = {}
    for j in sorted({j for j in (1, 2, 4, 8, args.k) if j <= args.k}):
        pj = float(np.mean([pass_at_j(int(c), args.k, j) for c in num_correct]))
        mj = np.zeros(n, dtype=bool)
        for p in range(n):
            counts = Counter(tuple(g) for g in back[p][:j])
            mj[p] = np.array_equal(np.array(counts.most_common(1)[0][0]), labels[p])
        rows[j] = (pj, float(mj.mean()))
        print(f"  {j:>3}  {pj:>8.4f}  {mj.mean():>11.4f}")

    print("\n  distribution of correct samples per puzzle (hits out of k)")
    hist = Counter(num_correct.tolist())
    for c in sorted(hist):
        bar = "#" * max(1, round(60 * hist[c] / n))
        print(f"  {c:>3}/{args.k}: {hist[c]:>5}  {bar}")

    out = {
        "checkpoint": args.checkpoint, "n_puzzles": n, "k": args.k, "seed": args.seed,
        "identity_exact_acc": float(ident_correct.mean()),
        "mean_sample_exact_acc": float(correct.mean()),
        "majority_exact_acc": float(maj_correct.mean()),
        "pass_at_k": float(any_hit.mean()),
        "no_hit_rate": float(no_hit.mean()),
        "majority_miss_rate": float(maj_miss.mean()),
        "majority_rescue_rate": float(maj_save.mean()),
        "voting_headroom": float(any_hit.mean() - maj_correct.mean()),
        "mean_cell_acc_on_no_hit": float(cell_acc_nohit),
        "pass_and_majority_at_j": {str(j): {"pass": v[0], "majority": v[1]} for j, v in rows.items()},
        "hits_histogram": {str(c): int(v) for c, v in sorted(hist.items())},
    }
    path = os.path.join(args.checkpoint_dir, f"tta_analysis_{args.checkpoint}_k{args.k}_n{n}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
