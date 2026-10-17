"""Sudoku harness -- 논문 지표.

  ① exact accuracy (조건부)          Table 1.  본 스크립트의 headline.
  ② N x iteration 격자               Figure 4(left).  --N 과 --halt-max-steps 를
                                     바꿔가며 이 스크립트를 반복 호출해 만든다.
                                     논문 대조점: N=20, halt_max_steps=16 -> 97.0%
  ④ LPRM 선택                        ②의 N>1 선택기 (verifier@N)

  ③ unconditional generation validity (Figure 5, 99.05%) 는 --unconditional 로
    설계만 넣어 두었다.  §설계 확인 참조.

★ coverage 는 넣지 않는다.  Sudoku-Extreme 은 퍼즐당 해가 유일하고 sidecar 도
  없다.  N-Queens/GC 의 coverage@20 과 대응하는 축이 존재하지 않는다.

★ exp/analyze_sudoku_tta.py 를 쓰지 않는다.  그쪽은 TRM 용이라 test-time
  augmentation 으로 다양성을 만드는데, GRAM 은 latent transition 이 stochastic
  이라 같은 입력에 K 번 forward 하면 이미 K 개의 서로 다른 답이 나온다
  (experiments/gram_harness.py 헤더).  TTA 를 얹으면 축이 어긋난다.

selftest 는 CPU 이고 모델이 필요 없다.  본 측정 전에 반드시 통과해야 한다.

용법:
    python experiments/sudoku_harness.py --selftest
    python experiments/sudoku_harness.py \
        --ckpt-dir checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch/sudoku_paper_34 \
        --step step_65104 --data data/sudoku-extreme-1k-aug-1000/test \
        --N 20 --halt-max-steps 16 --out reports/sudoku/final_N20_hms16.json
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluators.sudoku import BLANK, SEQ_LEN, exact_match, verify_boards   # noqa: E402


def majority_row(rows: np.ndarray) -> np.ndarray:
    """N개 보드 중 최빈 보드 1개, 동률은 첫 등장.  sweep_harness 와 같은 규칙."""
    return np.array(Counter(tuple(r) for r in rows).most_common(1)[0][0], dtype=np.int64)


def rate_block(preds: np.ndarray, inputs: np.ndarray, labels: np.ndarray) -> dict:
    """[B, 81] 보드 한 벌에 대한 지표 묶음."""
    r = verify_boards(preds, inputs)
    return {
        "exact": float(exact_match(preds, labels).mean()),
        "solved": float(r["solved"].mean()),
        "valid": float(r["valid"].mean()),
        "consistent": float(r["consistent"].mean()),
        "digits_ok": float(r["digits_ok"].mean()),
        "violations_mean": float(r["violations"].mean()),
        "clue_breaks_mean": float(r["clue_breaks"].mean()),
    }


def flat_stats(cand: np.ndarray, inputs: np.ndarray, labels: np.ndarray) -> dict:
    """전 샘플(N x B) 평균.  N=1 일 때 이 값이 곧 단일시도 정확도다."""
    N, B, L = cand.shape
    return rate_block(cand.reshape(N * B, L),
                      np.tile(inputs, (N, 1)), np.tile(labels, (N, 1)))


def pass_at_N(cand: np.ndarray, inputs: np.ndarray, labels: np.ndarray) -> dict:
    """N개 중 하나라도 맞은 퍼즐 비율 (선택기 없는 상한)."""
    N, B, L = cand.shape
    ok_e = exact_match(cand.reshape(N * B, L), np.tile(labels, (N, 1))).reshape(N, B)
    ok_s = verify_boards(cand.reshape(N * B, L),
                         np.tile(inputs, (N, 1)))["solved"].reshape(N, B)
    return {"exact": float(ok_e.any(0).mean()), "solved": float(ok_s.any(0).mean())}


def selftest(data: str) -> None:
    inputs = np.load(os.path.join(data, "all__inputs.npy")).astype(np.int64)[:256]
    labels = np.load(os.path.join(data, "all__labels.npy")).astype(np.int64)[:256]
    B = len(inputs)

    r = verify_boards(labels, inputs)
    a = rate_block(labels, inputs, labels)
    assert a["exact"] == 1.0, a
    assert a["valid"] == 1.0 and a["solved"] == 1.0, a
    assert a["consistent"] == 1.0, a
    assert r["violations"].sum() == 0 and r["clue_breaks"].sum() == 0
    print(f"[selftest a] 라벨      exact=1.0 valid=1.0 consistent=1.0 "
          f"violations={int(r['violations'].sum())} clue_breaks={int(r['clue_breaks'].sum())}")

    rng = np.random.default_rng(0)
    rnd = rng.integers(2, 11, size=(B, SEQ_LEN))
    rb = verify_boards(rnd, inputs)
    b = rate_block(rnd, inputs, labels)
    assert b["exact"] == 0.0 and b["valid"] == 0.0, b
    assert rb["violations"].min() > 0, rb["violations"].min()
    print(f"[selftest b] 무작위    exact=0.0 valid=0.0 "
          f"violations_mean={rb['violations'].mean():.2f}/27 (min {int(rb['violations'].min())})")

    bad = labels.copy()
    cell = rng.integers(0, SEQ_LEN, size=B)
    cur = bad[np.arange(B), cell]
    bad[np.arange(B), cell] = np.where(cur == 10, 2, cur + 1)   # 다른 숫자로
    cb = verify_boards(bad, inputs)
    c = rate_block(bad, inputs, labels)
    assert c["exact"] == 0.0, c
    assert cb["violations"].min() > 0, cb["violations"].min()
    print(f"[selftest c] 1칸 변조  exact=0.0 valid={c['valid']:.4f} "
          f"violations_mean={cb['violations'].mean():.2f} (min {int(cb['violations'].min())})")

    clue_bad = labels.copy()
    for i in range(B):
        pos = np.nonzero(inputs[i] != BLANK)[0]
        j = pos[rng.integers(len(pos))]
        clue_bad[i, j] = 2 + (clue_bad[i, j] - 2 + 1) % 9      # 다른 숫자로
    d1b = verify_boards(clue_bad, inputs)
    d1 = rate_block(clue_bad, inputs, labels)
    assert d1["consistent"] == 0.0, d1                          # 단서 보존 전부 실패
    assert (d1b["clue_breaks"] == 1).all(), d1b["clue_breaks"]
    assert d1["solved"] == 0.0 and d1["exact"] == 0.0, d1
    print(f"[selftest d] 단서1칸   consistent=0.0 clue_breaks=1 (전건) "
          f"solved=0.0 exact=0.0 violations_mean={d1b['violations'].mean():.2f}")

    relabel = labels.copy()
    relabel = 2 + (relabel - 2 + 1) % 9
    d2b = verify_boards(relabel, inputs)
    d2 = rate_block(relabel, inputs, labels)
    assert d2["valid"] == 1.0, d2            # 제약은 여전히 전부 통과
    assert d2["consistent"] == 0.0, d2       # 단서 보존은 전부 실패
    assert d2["solved"] == 0.0, d2           # -> solved 가 걸러낸다
    assert d2["exact"] == 0.0, d2
    assert d2b["violations"].sum() == 0
    print(f"[selftest d2] 숫자재라벨 valid=1.0 violations=0 consistent=0.0 solved=0.0 "
          f"clue_breaks_mean={d2b['clue_breaks'].mean():.2f}  "
          f"<- ★valid 만 봤으면 100% 통과했을 보드★")

    print(f"[selftest] PASSED  (n={B}, a 라벨=exact1 / b 무작위=exact0 위반다수 / "
          f"c 1칸변조=exact0 위반검출 / d 단서1칸=consistent0 검출 / "
          f"d2 재라벨=valid1 이지만 consistent0)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--step")
    ap.add_argument("--out")
    ap.add_argument("--data", default="data/sudoku-extreme-1k-aug-1000/test")
    ap.add_argument("--N", type=int, default=20, help="병렬 샘플 수 (Figure 4 의 곡선축)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-scale", type=float, default=1.0)
    ap.add_argument("--mu-scale", type=float, default=1.0)
    ap.add_argument("--halt-max-steps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--unconditional", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.data)
    assert args.ckpt_dir and args.step and args.out, "--ckpt-dir/--step/--out 필수"

    from experiments.gram_harness import load_model, rollout   # GPU 경로에서만 import

    net, _ = load_model(args.ckpt_dir, args.step)
    if args.halt_max_steps is not None:
        net.model.config.halt_max_steps = args.halt_max_steps

    inputs = np.load(os.path.join(args.data, "all__inputs.npy")).astype(np.int64)
    labels = np.load(os.path.join(args.data, "all__labels.npy")).astype(np.int64)
    n_total = len(inputs)
    if args.limit is not None:
        inputs, labels = inputs[: args.limit], labels[: args.limit]
    assert 0 <= args.shard_index < args.shard_count, "shard-index 범위 오류"
    bounds = np.linspace(0, len(inputs), args.shard_count + 1).astype(int)
    lo, hi = int(bounds[args.shard_index]), int(bounds[args.shard_index + 1])
    shard_range = [lo, hi]
    inputs, labels = inputs[lo:hi], labels[lo:hi]
    B = len(inputs)

    if args.unconditional:
        inputs = np.full((B, SEQ_LEN), BLANK, dtype=np.int64)
        labels = np.zeros((B, SEQ_LEN), dtype=np.int64)

    pri = rollout(net, inputs, labels, K=args.N, sigma_scale_=args.sigma_scale,
                  mode="prior", seed=args.seed, verbose=False, mu_scale_=args.mu_scale)
    cand, v = pri["preds"], pri["v"]                            # [N, B, 81], [N, B]

    res = {
        "checkpoint": os.path.join(args.ckpt_dir, args.step),
        "data": args.data,
        "n": B, "n_total_in_split": n_total, "limit": args.limit,
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "shard_range": shard_range,
        "N": args.N, "seed": args.seed,
        "sigma_scale": args.sigma_scale, "mu_scale": args.mu_scale,
        "halt_max_steps": net.model.config.halt_max_steps,
        "unconditional": bool(args.unconditional),
    }

    res["prior_all_samples"] = flat_stats(cand, inputs, labels)
    res["pass_at_N"] = pass_at_N(cand, inputs, labels)
    maj = np.stack([majority_row(cand[:, b, :]) for b in range(B)])
    sel = np.stack([cand[int(np.argmax(v[:, b])), b, :] for b in range(B)])
    res["majority_at_N"] = rate_block(maj, inputs, labels)
    res["verifier_selected_at_N"] = rate_block(sel, inputs, labels)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    p, m, s = res["prior_all_samples"], res["majority_at_N"], res["verifier_selected_at_N"]
    print(f"{res['checkpoint']}  hms={res['halt_max_steps']} N={args.N} n={B}"
          f"{' UNCOND' if args.unconditional else ''}\n"
          f"  prior  exact={p['exact']:.4f} solved={p['solved']:.4f} "
          f"valid={p['valid']:.4f} consistent={p['consistent']:.4f}\n"
          f"  pass@N exact={res['pass_at_N']['exact']:.4f} solved={res['pass_at_N']['solved']:.4f}\n"
          f"  maj@N  exact={m['exact']:.4f} solved={m['solved']:.4f}\n"
          f"  vsel@N exact={s['exact']:.4f} solved={s['solved']:.4f}\n"
          f"  [out] {args.out}")


if __name__ == "__main__":
    main()
