"""E3 -- is the LPRM head a usable verifier?

E1 shows how much headroom majority voting leaves on the table.  If the LPRM
value head can rank a correct sample above the wrong ones, that headroom is
recoverable by best-of-K selection instead of voting.

Measures:
  * AUROC of v over (sample, is_correct) pairs -- does v separate right from wrong?
  * three selection curves vs K: best-of-K by v / majority@K / pass@K (oracle)
  * rescue rate: among majority-miss puzzles, how often does v rank a correct
    sample first?  That is exactly the slice best-of-K would win over voting.

Usage (4 GPUs):
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node 4 --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29501 --nnodes=1 experiments/e3_lprm.py \
    --ckpt-dir checkpoints/.../gram_sudoku_lsum_exp --step step_65100
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# experiments/legacy/diag/ 로 내려왔다: repo root(pretrain), experiments/(gram_harness), 그리고 여기
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.dirname(_HERE))), os.path.dirname(os.path.dirname(_HERE)), _HERE]
import gram_harness as H  # noqa: E402


def main():
    ap = H.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--sigma-scale", type=float, default=1.0,
                    help="latent std multiplier; e4 finds the argmax is ~0.5, not the trained 1.0")
    args = ap.parse_args()
    rank, world = H.dist_setup()

    net, config = H.load_model(args.ckpt_dir, args.step)
    inputs, labels = H.load_puzzles(config, args.num_puzzles, args.seed)
    if rank == 0:
        print(f"E3  ckpt={args.step}  n={len(inputs)}  K={args.k}  "
              f"sigma_scale={args.sigma_scale}  world={world}\n")

    roll = H.sharded_rollout(net, inputs, labels, args.k, rank, world,
                             mode="prior", sigma_scale_=args.sigma_scale,
                             batch_size=args.batch_size,
                             seed=args.seed, verbose=(rank == 0))
    if rank != 0:
        H.dist_teardown()
        return

    m = H.metrics(roll["preds"], labels)
    correct = m["_correct"]                       # [N, K]
    v = roll["v"].T                               # [N, K]
    n = len(labels)

    res = dict(checkpoint=args.step, n=n, K=args.k, seed=args.seed,
               sigma_scale=args.sigma_scale,
               auroc_sample_level=H.auroc(v.ravel(), correct.ravel()),
               v_mean_correct=float(v[correct].mean()) if correct.any() else float("nan"),
               v_mean_wrong=float(v[~correct].mean()) if (~correct).any() else float("nan"))

    # selection curves over prefixes of the K samples
    curves = {}
    for j in sorted({j for j in (1, 2, 4, 8, args.k) if j <= args.k}):
        pick = v[:, :j].argmax(1)
        best_of = correct[np.arange(n), pick]
        maj = np.array([np.array_equal(H.majority_vote(roll["preds"][:j, p])[0], labels[p])
                        for p in range(n)])
        curves[j] = dict(best_of_j_by_v=float(best_of.mean()),
                         majority_at_j=float(maj.mean()),
                         pass_at_j=float(correct[:, :j].any(1).mean()))
    res["curves"] = curves

    # the slice that matters: voting failed but a correct sample was available
    maj_miss = correct.any(1) & ~m["_maj_correct"]
    top1 = correct[np.arange(n), v.argmax(1)]
    res["majority_miss_count"] = int(maj_miss.sum())
    res["v_rescues_majority_miss"] = float(top1[maj_miss].mean()) if maj_miss.any() else float("nan")
    res["v_breaks_majority_hit"] = float((~top1[m["_maj_correct"]]).mean()) if m["_maj_correct"].any() else float("nan")

    print("\n" + "=" * 68)
    print(f"LPRM VERIFIER  ckpt={args.step}  n={n}  K={args.k}  sigma_scale={args.sigma_scale}")
    print("=" * 68)
    print(f"  AUROC (sample level, v vs is_correct) : {res['auroc_sample_level']:.4f}")
    print(f"  mean v | correct                      : {res['v_mean_correct']:+.4f}")
    print(f"  mean v | wrong                        : {res['v_mean_wrong']:+.4f}")
    print(f"\n  {'j':>3}  {'best-of-j by v':>15}  {'majority@j':>11}  {'pass@j (oracle)':>16}")
    for j, c in curves.items():
        print(f"  {j:>3}  {c['best_of_j_by_v']:>15.4f}  {c['majority_at_j']:>11.4f}  {c['pass_at_j']:>16.4f}")
    print(f"\n  majority-miss puzzles                 : {res['majority_miss_count']}")
    print(f"  ...of which v ranks a correct sample 1st: {res['v_rescues_majority_miss']:.4f}")
    print(f"  majority-hit puzzles v would break     : {res['v_breaks_majority_hit']:.4f}")

    H.save_json(res, H.out_dir_for(args),
                f"e3_lprm_{args.step}_k{args.k}_n{n}_sig{args.sigma_scale}.json")
    H.dist_teardown()


if __name__ == "__main__":
    main()
