"""E1 -- failure decomposition: where does GRAM's accuracy actually go?

K prior rollouts per puzzle (GRAM is stochastic, so no augmentation is needed).
Splits failures into the two kinds that call for different fixes:
  * NO-HIT      -- the model never produces the right grid in K tries  -> capability gap
  * MAJORITY MISS -- a correct sample exists but voting discards it    -> selection gap

Printed next to the TRM numbers on the same benchmark for reference.

Usage (4 GPUs):
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node 4 --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29501 --nnodes=1 experiments/e1_failure_decomp.py \
    --ckpt-dir checkpoints/.../gram_sudoku_lsum_exp --step step_65100
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# experiments/legacy/diag/ 로 내려왔다: repo root(pretrain), experiments/(gram_harness), 그리고 여기
# experiments/legacy/ 를 추가 -- gram_harness_legacy 를 여기서 찾는다.
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.dirname(_HERE))), os.path.dirname(os.path.dirname(_HERE)), os.path.dirname(_HERE), _HERE]
import gram_harness_legacy as H  # noqa: E402  (현행 gram_harness 는 축소됨)

# TRM on the same 10k Sudoku-Extreme test subsample, k=16 (analyze_sudoku_tta.py).
TRM_REF = dict(no_hit_rate=0.015, majority_miss_rate=0.008, pass_at_K=0.985)


def main():
    ap = H.add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    rank, world = H.dist_setup()

    net, config = H.load_model(args.ckpt_dir, args.step)
    inputs, labels = H.load_puzzles(config, args.num_puzzles, args.seed)
    if rank == 0:
        print(f"E1  ckpt={args.step}  n={len(inputs)}  K={args.k}  world={world}\n")

    roll = H.sharded_rollout(net, inputs, labels, args.k, rank, world,
                             mode="prior", batch_size=args.batch_size,
                             seed=args.seed, verbose=(rank == 0))
    m = H.metrics(roll["preds"], labels)

    if rank == 0:
        print("\n" + "=" * 68)
        print(f"FAILURE DECOMPOSITION  ckpt={args.step}  n={m['n']}  K={m['K']}")
        print("=" * 68)
        print(f"  single-sample exact acc            : {m['single_exact']:.4f}")
        print(f"  mean per-sample exact acc          : {m['mean_sample_exact']:.4f}")
        print(f"  mean per-sample token acc          : {m['mean_token_acc']:.4f}")
        print(f"  majority@{m['K']} exact acc               : {m['majority_exact']:.4f}")
        print(f"  pass@{m['K']} (>=1 correct sample)        : {m['pass_at_K']:.4f}")

        print("\n  failure split                        GRAM       TRM      delta")
        for key, label in (("no_hit_rate", "NO-HIT (never right in K)"),
                           ("majority_miss_rate", "MAJORITY MISS (hit thrown away)"),
                           ("pass_at_K", "pass@K")):
            g, t = m[key], TRM_REF[key]
            print(f"  {label:34s} {g:.4f}    {t:.4f}   {g - t:+.4f}")
        print(f"\n  voting headroom (pass@K - majority@K): {m['voting_headroom']:.4f}")

        print(f"\n  {'j':>3}  {'pass@j':>8}  {'majority@j':>11}")
        for j, r in m["by_j"].items():
            print(f"  {j:>3}  {r['pass_at_j']:>8.4f}  {r['majority_at_j']:>11.4f}")
        H.print_hits_histogram(m["hits_histogram"], m["n"], m["K"])

        m.update(checkpoint=args.step, seed=args.seed, trm_reference=TRM_REF)
        H.save_json(m, H.out_dir_for(args), f"e1_failure_{args.step}_k{args.k}_n{m['n']}.json")

    H.dist_teardown()


if __name__ == "__main__":
    main()
