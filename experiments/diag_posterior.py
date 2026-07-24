"""Posterior vs prior 경로 대조 진단.

핵심 질문: 백본이 유효 보드를 만들 줄 아는가?
  - posterior(y 주입)로 test에서 valid가 높으면 -> 백본 정상, prior만 실패 (KL/sigma 층 문제)
  - posterior로도 0%면 -> 백본이 제약을 못 배움 (train 1.0은 순수 암기)

사용법:
    python experiments/diag_posterior.py \
        --ckpt-dir checkpoints/Nqueens-8-ACT-torch/gram_nq8_mean_nh512 --step step_3417
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.gram_harness import load_model, rollout  # noqa: E402
from evaluators.nqueens import verify_boards  # noqa: E402


def report(tag, preds, inputs, n):
    r = verify_boards(preds, inputs, n)
    qc = r["queen_count"]
    print(f"[{tag}] valid={r['valid'].mean():.4f}  consistent={r['consistent'].mean():.4f}  "
          f"strict={r['strict'].mean():.4f}  queen_count mean={qc.mean():.2f} "
          f"(min={qc.min()} max={qc.max()})")
    return r


def show_boards(preds, inputs, n, k=3):
    for i in range(k):
        iq = (inputs[i].reshape(n, n) == 2).astype(int)
        pq = (preds[i].reshape(n, n) == 2).astype(int)
        kept = bool(((inputs[i] != 2) | (preds[i] == 2)).all())
        print(f"--- board {i}  (input queens={iq.sum()}, pred queens={pq.sum()}, input 보존={kept})")
        for r_in, r_pr in zip(iq, pq):
            print("".join("Q" if v else "." for v in r_in), "   ",
                  "".join("Q" if v else "." for v in r_pr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", default="data/nqueens-8/test")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--num", type=int, default=256, help="검사할 puzzle 수")
    args = ap.parse_args()

    net, _ = load_model(args.ckpt_dir, args.step)

    inputs = np.load(f"{args.data}/all__inputs.npy")[: args.num]
    labels = np.load(f"{args.data}/all__labels.npy")[: args.num]

    print(f"n={args.num} puzzles, checkpoint={args.ckpt_dir}/{args.step}\n")

    out_post = rollout(net, inputs, labels, K=1, mode="posterior", verbose=False)
    report("posterior (y 주입)", out_post["preds"][0], inputs, args.n)

    out_prior = rollout(net, inputs, labels, K=1, mode="prior", verbose=False)
    report("prior     (실전)  ", out_prior["preds"][0], inputs, args.n)

    print("\n== posterior 출력 보드 (좌: input, 우: pred) ==")
    show_boards(out_post["preds"][0], inputs, args.n)
    print("\n== prior 출력 보드 ==")
    show_boards(out_prior["preds"][0], inputs, args.n)

    print("\n판독:")
    print("  posterior valid 높음 + prior 0  -> 백본 정상, prior가 못 따라옴 (KL/sigma 층)")
    print("  posterior valid도 0            -> 백본 미학습, train 1.0은 암기 (과적합 층)")


if __name__ == "__main__":
    main()