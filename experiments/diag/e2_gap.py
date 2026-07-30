"""E2 -- prior/posterior gap: how much does the model gain from seeing y?

Same puzzles, two rollout paths:
  prior     eps ~ p(.|u)     -- the deployable path (eval)
  posterior eps ~ q(.|u,y)   -- inner forced to train mode, so y is injected

A large gap means the latent carries answer information the prior cannot
reconstruct at test time -- i.e. the posterior is smuggling y (see diag_sigma.py).
A near-zero gap means the latent is not being used to carry the answer at all.
The final-transition KL distribution is captured alongside to price that gap.

Usage (4 GPUs):
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node 4 --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29501 --nnodes=1 experiments/e2_gap.py \
    --ckpt-dir checkpoints/.../gram_sudoku_lsum_exp --step step_65100
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# experiments/diag/ 로 내려왔다: repo root(pretrain), experiments/(gram_harness), 그리고 여기
sys.path[:0] = [os.path.dirname(os.path.dirname(_HERE)), os.path.dirname(_HERE), _HERE]
import gram_harness as H  # noqa: E402


def summarize(roll, labels, tag):
    exact = (roll["preds"] == labels[None]).all(-1)
    return {
        f"{tag}_token_acc": float(roll["tok_acc"].mean()),
        f"{tag}_exact_acc": float(exact.mean()),
        f"{tag}_pass_at_K": float(exact.any(0).mean()),
    }


def main():
    ap = H.add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    rank, world = H.dist_setup()

    net, config = H.load_model(args.ckpt_dir, args.step)
    inputs, labels = H.load_puzzles(config, args.num_puzzles, args.seed)
    if rank == 0:
        print(f"E2  ckpt={args.step}  n={len(inputs)}  K={args.k}  world={world}\n")

    res = {}
    prior = H.sharded_rollout(net, inputs, labels, args.k, rank, world,
                              mode="prior", batch_size=args.batch_size,
                              seed=args.seed, verbose=(rank == 0))
    res.update(summarize(prior, labels, "prior"))

    # posterior path also fires kl_loss; capture the per-sample KL it is charged.
    kl_sink = []
    with H.capture_kl(net, kl_sink):
        post = H.sharded_rollout(net, inputs, labels, args.k, rank, world,
                                 mode="posterior", batch_size=args.batch_size,
                                 seed=args.seed, verbose=(rank == 0))
    res.update(summarize(post, labels, "posterior"))

    # kl_loss fires once per ACT step; the last halt_max_steps entries of the final
    # rollout are the final transition's KL for this rank's shard.
    kl_last = np.concatenate(kl_sink[-1:]) if kl_sink else np.zeros(0, dtype=np.float32)
    kl_all = np.concatenate([k.ravel() for k in H.dist_gather(kl_last, rank, world)])

    if rank == 0:
        res["gap_token_acc"] = res["posterior_token_acc"] - res["prior_token_acc"]
        res["gap_exact_acc"] = res["posterior_exact_acc"] - res["prior_exact_acc"]
        qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        res["kl_last_transition"] = {
            "mean": float(kl_all.mean()) if kl_all.size else float("nan"),
            "std": float(kl_all.std()) if kl_all.size else float("nan"),
            "quantiles": {str(q): float(np.percentile(kl_all, q)) for q in qs} if kl_all.size else {},
            "n_samples": int(kl_all.size),
        }

        print("\n" + "=" * 68)
        print(f"PRIOR / POSTERIOR GAP  ckpt={args.step}  n={args.num_puzzles}  K={args.k}")
        print("=" * 68)
        print(f"  {'path':12s} {'token acc':>10} {'exact acc':>10} {'pass@K':>10}")
        for tag in ("prior", "posterior"):
            print(f"  {tag:12s} {res[f'{tag}_token_acc']:>10.4f} "
                  f"{res[f'{tag}_exact_acc']:>10.4f} {res[f'{tag}_pass_at_K']:>10.4f}")
        print(f"  {'GAP':12s} {res['gap_token_acc']:>+10.4f} {res['gap_exact_acc']:>+10.4f}")
        print("\n  final-transition KL per sample (summed over L, D)")
        k = res["kl_last_transition"]
        print(f"    mean={k['mean']:.4f}  std={k['std']:.4f}  n={k['n_samples']}")
        for q, v in k["quantiles"].items():
            print(f"    p{q:>3}: {v:12.4f}")

        res.update(checkpoint=args.step, n=args.num_puzzles, K=args.k, seed=args.seed)
        H.save_json(res, H.out_dir_for(args), f"e2_gap_{args.step}_k{args.k}_n{args.num_puzzles}.json")

    H.dist_teardown()


if __name__ == "__main__":
    main()
