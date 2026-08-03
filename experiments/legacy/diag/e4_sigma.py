"""E4 -- sigma diagnosis and sweep: is the learned noise helping or hurting?

(a) Learned sigma per transition depth.  reparameterize() fires H_cycles times per
    forward, so call index % H_cycles gives the transition; this shows whether the
    model anneals noise with depth or keeps it flat.

(b) sigma_scale sweep.  gram.py multiplies the sampled std by inner.sigma_scale
    (default 1.0 == unchanged).  sigma_scale=0 makes the latent deterministic --
    a direct "GRAM without stochasticity" comparison point against TRM.  If single
    accuracy rises at 0 while pass@K collapses, the noise is buying diversity, not
    capability; if both rise, the noise is simply hurting.

Usage (4 GPUs):
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node 4 --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29501 --nnodes=1 experiments/e4_sigma.py \
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
    ap.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--sigma-probe-n", type=int, default=512,
                    help="puzzles used for the per-transition sigma statistics")
    args = ap.parse_args()
    rank, world = H.dist_setup()

    net, config = H.load_model(args.ckpt_dir, args.step)
    inputs, labels = H.load_puzzles(config, args.num_puzzles, args.seed)
    H_cycles = net.model.config.H_cycles
    if rank == 0:
        print(f"E4  ckpt={args.step}  n={len(inputs)}  K={args.k}  "
              f"H_cycles={H_cycles}  scales={args.scales}  world={world}\n")

    res = dict(checkpoint=args.step, n=args.num_puzzles, K=args.k, seed=args.seed,
               H_cycles=H_cycles, scales=args.scales)

    # ---- (a) learned sigma per transition depth (prior path, sigma_scale=1) ----
    if rank == 0:
        probe_n = min(args.sigma_probe_n, len(inputs))
        sink = []
        with H.capture_sigma(net, sink):
            H.rollout(net, inputs[:probe_n], labels[:probe_n], 1,
                      batch_size=args.batch_size, seed=args.seed, verbose=False)
        by_t = {}
        for t in range(H_cycles):
            rec = sink[t::H_cycles]
            by_t[t + 1] = {k: float(np.mean([r[k] for r in rec])) for k in ("mean", "std", "median", "p99")}
            by_t[t + 1]["n_calls"] = len(rec)
        res["sigma_p_by_transition"] = by_t

        print("=" * 68)
        print(f"(a) LEARNED sigma_p BY TRANSITION  (prior path, n={probe_n})")
        print("=" * 68)
        print(f"  {'transition':>11} {'mean':>10} {'std':>10} {'median':>10} {'p99':>10} {'calls':>7}")
        for t, s in by_t.items():
            print(f"  {t:>11} {s['mean']:>10.4f} {s['std']:>10.4f} "
                  f"{s['median']:>10.4f} {s['p99']:>10.4f} {s['n_calls']:>7}")

    # ---- (b) sigma_scale sweep ----
    sweep = {}
    for sc in args.scales:
        roll = H.sharded_rollout(net, inputs, labels, args.k, rank, world,
                                 sigma_scale_=sc, mode="prior",
                                 batch_size=args.batch_size, seed=args.seed,
                                 verbose=(rank == 0))
        if rank == 0:
            m = H.metrics(roll["preds"], labels, js=(1, args.k))
            sweep[str(sc)] = dict(single_exact=m["single_exact"],
                                  mean_sample_exact=m["mean_sample_exact"],
                                  mean_token_acc=m["mean_token_acc"],
                                  pass_at_K=m["pass_at_K"],
                                  majority_exact=m["majority_exact"],
                                  no_hit_rate=m["no_hit_rate"],
                                  n_distinct_mean=float(np.mean(
                                      [len({tuple(g) for g in roll["preds"][:, p]})
                                       for p in range(len(labels))])))

    if rank == 0:
        res["sweep"] = sweep
        print("\n" + "=" * 68)
        print(f"(b) sigma_scale SWEEP  n={args.num_puzzles}  K={args.k}")
        print("=" * 68)
        print(f"  {'scale':>6} {'single':>9} {'token':>9} {'pass@K':>9} "
              f"{'major@K':>9} {'no-hit':>9} {'distinct':>9}")
        for sc, s in sweep.items():
            print(f"  {sc:>6} {s['single_exact']:>9.4f} {s['mean_token_acc']:>9.4f} "
                  f"{s['pass_at_K']:>9.4f} {s['majority_exact']:>9.4f} "
                  f"{s['no_hit_rate']:>9.4f} {s['n_distinct_mean']:>9.2f}")
        print("\n  scale=0.0 is the deterministic (TRM-like) comparison point.")
        H.save_json(res, H.out_dir_for(args), f"e4_sigma_{args.step}_k{args.k}_n{args.num_puzzles}.json")

    H.dist_teardown()


if __name__ == "__main__":
    main()
