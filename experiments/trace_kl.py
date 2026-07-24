"""Transition별 KL 계측.

GRAM Eq 14는 supervision step당 마지막 transition에만 KL을 건다.
앞선 H_cycles-1개 transition은 posterior를 쓰면서도 페널티를 받지 않으므로,
y가 그 경로로 새어 들어갈 수 있다 (train exact 1.0 / eval 0.0 의 후보 원인).

이 스크립트는 학습된 checkpoint에서 transition별 KL을 직접 측정한다.
gram.py의 forward에 trace_kl 계측 코드가 들어 있어야 한다.

사용법:
    DISABLE_COMPILE=1 python experiments/trace_kl.py \
        --ckpt-dir checkpoints/Nqueens-8-ACT-torch/gram_nq8_mean1 \
        --step step_17085
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.gram_harness import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True, help="예: step_17085")
    ap.add_argument("--data", default="data/nqueens-8/test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--sup-steps", type=int, default=3, help="관찰할 supervision step 수")
    args = ap.parse_args()

    net, config = load_model(args.ckpt_dir, args.step)
    model = getattr(net, "model", net)  # ACTLossHead 래퍼 대응
    inner = model.inner
    model.eval()

    if not hasattr(inner, "trace_kl") and not hasattr(inner, "_kl_trace"):
        print("경고: gram.py에 trace_kl 계측 코드가 없는 것 같다. 계속 진행한다.")
    inner.trace_kl = True

    B = args.batch_size
    batch = {
        "inputs": torch.from_numpy(np.load(f"{args.data}/all__inputs.npy")[:B]).cuda(),
        "labels": torch.from_numpy(np.load(f"{args.data}/all__labels.npy")[:B]).cuda(),
        "puzzle_identifiers": torch.zeros(B, dtype=torch.int32).cuda(),
    }

    with torch.device("cuda"):
        carry = model.initial_carry(batch)

    print(f"checkpoint : {args.ckpt_dir}/{args.step}")
    print(f"batch      : {B} puzzles")
    print(f"H_cycles   : {model.config.H_cycles}  (transition/supervision step)")
    print("KL 정의    : D축 합, batch·seq 평균  (= train 로그의 kl_loss, diag/kl_raw와 동일)")
    print("-" * 62)

    with torch.no_grad():
        for n in range(args.sup_steps):
            carry, _ = model(carry=carry, batch=batch, use_posterior=True)
            tr = list(getattr(inner, "_kl_trace", []))
            if not tr:
                print("계측값이 비었다. gram.py의 trace_kl 코드를 확인해라.")
                break
            body = "  ".join(f"t{i+1}={v:9.4f}" for i, v in enumerate(tr))
            flag = ""
            if len(tr) >= 2 and tr[-1] > 0 and max(tr[:-1]) > 10 * tr[-1]:
                flag = "   <-- 앞선 transition이 마지막보다 훨씬 큼 (누출 의심)"
            print(f"sup step {n+1}: {body}{flag}")

    print("-" * 62)
    print("판독:")
    print("  [큰값, 큰값, ~0.002] -> 무페널티 transition으로 y 누출 확정")
    print("  [~0.002 ...]        -> 가설 기각, 다른 경로로 y가 흐른다")
    print("  마지막 원소는 train 로그의 kl_loss(~0.0018)와 대조 가능해야 한다")


if __name__ == "__main__":
    main()