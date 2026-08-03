"""Graph-Coloring 용 harness: prior N샘플 + posterior 1샷, conflict/coverage 집계.

sweep_harness.py 를 분기하지 않고 별도 스크립트로 둔다 -- 그쪽 --n 은 보드 크기(n*n=seq),
여기 --n 은 노드 수(n(n-1)/2=seq)라 한 파일에 섞으면 의미가 엉킨다.

논문 GC 지표 (Table 1, 8-vertex; 번호는 sweep_harness.py docstring 의 ①~⑤ 체계):
  ② Conflict ↓ (accuracy 대신 보고) = (1/(N·B)) Σ_x Σ_k conflicts(ŷ_x^k)
              conflicts = 양 끝 노드가 같은 색인 edge 의 개수. 논문 2.7±2.1.
              -> flat_stats() 의 conflicts_mean (std 병기).
  ③ coverage  = 정답해 집합을 몇 % 발견했는가. 논문 85.8±0.5.
              found(x) = N 샘플 중 valid 인 보드를 **first-appearance 정규형으로
              canonical 화한 뒤** unique 개수.  <- 정규화 필수. 빠뜨리면 같은 해의
              색 치환(최대 3!=6개)이 서로 다른 해로 잡혀 coverage 가 부풀려진다.
              total(x) = sidecar_gc_8.json 의 per-graph canonical 해 개수.
              빌더가 first-appearance 정규형으로 열거하므로 같은 규칙이어야
              found ⊆ 열거집합이 보장된다 (n_found_gt_total=0 이 그 검산).

  ④ majority@20 (§2.3 방법 1) = canonical 정규화 후 최빈 1개를 valid 채점
              -> majority_at_N_valid_canonical.  정규화 없는 원본 토큰 최빈은
              색 치환 때문에 사실상 무작위 선택이라 진단 병기용이다.
  ⑤ verifier@20 = LPRM best-of-N (§2.3 방법 2) -> verifier_selected_at_N_valid.

N-Queens 와 달리 consistent 개념이 없다: 입력이 그래프 자체라 보존할 힌트가 없고,
valid(전 edge 양끝 다른 색)가 곧 "x 의 해"다.

진단 지표(valid/pass@k/distinct/prior_single_k0/posterior/num_colors_used/
비정규화 majority)는 2026-08-03 minimal 정리에서 삭제 --
필요하면 legacy/gc_harness_legacy.py.

사용법:
    python experiments/gc_harness.py --ckpt-dir <dir> --step step_N.live \
        --seed 0 --out experiments/results/gc_<run>_<step>.json
    python experiments/gc_harness.py --selftest        # CPU, 모델 불필요
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators.graph_coloring import verify_colorings, n_from_seq_len  # noqa: E402

COLOR0 = 3


# ---------------------------------------------------------------------------
# 논문 지표 (전부 순수 numpy -- selftest 가 모델 없이 검증한다)
# ---------------------------------------------------------------------------

def majority_row(rows):
    """④ 의 selection: 최빈 보드, 동률은 첫 등장 (sweep_harness 와 동일).

    GC 논문 지표로는 canonicalize() 를 거친 보드에 적용한다 (main() 의 majc 경로).
    """
    return np.array(Counter(tuple(r) for r in rows).most_common(1)[0][0], dtype=np.int64)


def canonicalize(colors):
    """[..., n] 색(0..2 기준 아님, 토큰 3..5) -> first-appearance 정규형 (0..2).

    빌더(dataset/build_graph_coloring_dataset.py)와 같은 규칙: 처음 나온 색이 0,
    다음 새 색이 1, 그 다음이 2.  색 토큰이 3..5 를 벗어나면 호출 전에 걸러야 한다.
    """
    c = colors - COLOR0
    out = np.empty_like(c)
    for i in range(c.shape[0]):
        remap, nxt = {}, 0
        for j, v in enumerate(c[i]):
            if v not in remap:
                remap[v] = nxt
                nxt += 1
            out[i, j] = remap[v]
    return out


def graph_hash_rows(inputs):
    """[B, P] 토큰(1/2) -> 빌더와 동일한 sha1[:16] 해시 리스트."""
    return [hashlib.sha1(row.astype(np.uint8).tobytes()).hexdigest()[:16] for row in inputs]


def flat_stats(preds, inputs, n):
    """preds [K, B, L] -> ② Conflict = conflicts_mean (±std, 논문 2.7±2.1 형식)."""
    K, B, L = preds.shape
    r = verify_colorings(preds.reshape(K * B, L), np.tile(inputs, (K, 1)), n)
    return dict(conflicts_mean=float(r["conflicts"].mean()),
                conflicts_std=float(r["conflicts"].std()))


def coverage_at_N(cand, inputs, n, totals):
    """③ coverage.  cand [N, B, L].  found = valid 샘플의 canonical unique 수, total = sidecar 해 수."""
    N, B, L = cand.shape
    r = verify_colorings(cand.reshape(N * B, L), np.tile(inputs, (N, 1)), n)
    ok = r["valid"].reshape(N, B)

    found = np.zeros(B, dtype=np.int64)
    for b in range(B):
        good = [cand[k, b, :n] for k in range(N) if ok[k, b]]
        if good:
            canon = canonicalize(np.stack(good))
            found[b] = len({tuple(c) for c in canon})

    have = totals > 0
    cov = np.zeros(B, dtype=np.float64)
    cov[have] = found[have] / totals[have]

    res = dict(
        coverage=float(cov[have].mean()) if have.any() else None,
        n_puzzles_scored=int(have.sum()),
        n_puzzles_no_total=int((~have).sum()),
        found_mean=float(found[have].mean()) if have.any() else None,
        total_mean=float(totals[have].mean()) if have.any() else None,
        # canonical 화가 맞으면 found 는 열거집합의 부분집합이라 절대 total 을 못 넘는다.
        n_found_gt_total=int((found[have] > totals[have]).sum()),
    )
    # 논문 Fig 15 대응 버킷 분해
    buckets = [(1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"), (21, 10**9, "21+")]
    bk = {}
    for lo, hi, name in buckets:
        m = have & (totals >= lo) & (totals <= hi)
        if m.any():
            bk[name] = dict(n=int(m.sum()), coverage=float(cov[m].mean()),
                            found_mean=float(found[m].mean()), total_mean=float(totals[m].mean()))
    res["by_solution_bucket"] = bk
    return res


METRIC_DEFS = {
    "prior_all_samples.conflicts_mean/std": "② 논문 GC 헤드라인(8-vertex 2.7±2.1, 낮을수록 좋음). "
                                            "N샘플 전체의 위반 edge 수 분포.",
    "majority_at_N_valid_canonical": "④ 논문 §2.3 selection 방법 1. 색 치환 동치류를 canonical 화로 "
                                     "합친 뒤의 최빈 1개를 valid 채점 (비정규화 최빈은 색 치환 때문에 사실상 무작위 선택).",
    "verifier_selected_at_N_valid": "⑤ 논문 §2.3 방법 2 = LPRM best-of-N. 모델 lrpm 점수 최고 샘플 1개를 valid 채점. 정답 미사용.",
    "coverage_at_20": "③ 논문 coverage(85.8±0.5). found = valid 샘플의 first-appearance canonical unique 수, "
                      "total = sidecar per-graph canonical 해 수. N-Queens 와 달리 consistent 조건 없음(입력에 힌트가 없다).",
}


# ---------------------------------------------------------------------------
# selftest (CPU, 모델 불필요)
# ---------------------------------------------------------------------------

def selftest(data_dir, sidecar_path, n):
    inputs = np.load(os.path.join(data_dir, "all__inputs.npy")).astype(np.int64)
    labels = np.load(os.path.join(data_dir, "all__labels.npy")).astype(np.int64)
    B, L = inputs.shape
    side = json.load(open(sidecar_path))["graphs"]
    hashes = graph_hash_rows(inputs)
    miss = sum(1 for h in hashes if h not in side)
    totals = np.array([len(side[h]["colorings"]) if h in side else 0 for h in hashes])
    print(f"[selftest] puzzles={B} seq={L} n={n}  sidecar 매칭실패={miss}  total 평균={totals[totals>0].mean():.2f}")
    assert miss == 0, "graph hash 가 sidecar 와 안 맞는다"

    # (a) 라벨 주입 -> conflict 0
    lab = labels[None]                              # [1, B, L]
    st = flat_stats(lab, inputs, n)
    print(f"[a] 라벨 주입      : conflicts={st['conflicts_mean']:.6f}  (기대 0.0)")
    assert st["conflicts_mean"] == 0.0

    # (b) 무작위 색 -> conflict ~= edges/3
    rng = np.random.default_rng(0)
    rnd = np.zeros((20, B, L), dtype=np.int64)
    rnd[:, :, :n] = rng.integers(3, 6, size=(20, B, n))
    st = flat_stats(rnd, inputs, n)
    exp = float((inputs == 2).sum(1).mean()) / 3.0
    print(f"[b] 무작위 색      : conflicts={st['conflicts_mean']:.3f}±{st['conflicts_std']:.3f}  (기대 edges/3 ≈ {exp:.3f})")
    assert abs(st["conflicts_mean"] - exp) < 0.15

    # (c) sidecar 전체 해 주입 -> coverage 1.0, found>total 0
    K = int(totals.max())
    cand = np.zeros((K, B, L), dtype=np.int64)
    for b, h in enumerate(hashes):
        sols = np.array(side[h]["colorings"], dtype=np.int64) + COLOR0   # [S, n]
        for k in range(K):
            cand[k, b, :n] = sols[min(k, len(sols) - 1)]
    cov = coverage_at_N(cand, inputs, n, totals)
    print(f"[c] 전체 해 주입   : coverage={cov['coverage']:.6f} found_gt_total={cov['n_found_gt_total']}  (기대 1.0 / 0)")
    assert cov["coverage"] == 1.0 and cov["n_found_gt_total"] == 0

    # (d) 색 치환 중복 주입 -> canonical 화로 found 1 이어야
    one = labels[:1]                                # 아무 퍼즐 1개
    perms = []
    base = one[0, :n] - COLOR0
    import itertools
    for p in itertools.permutations(range(3)):
        perms.append(np.array(p)[base] + COLOR0)
    cand = np.zeros((6, 1, L), dtype=np.int64)
    for k in range(6):
        cand[k, 0, :n] = perms[k]
    cov = coverage_at_N(cand, inputs[:1], n, np.array([len(side[hashes[0]]["colorings"])]))
    print(f"[d] 색치환 6개 주입: found_mean={cov['found_mean']:.1f}  (기대 1.0 -- 정규화가 중복 제거)")
    assert cov["found_mean"] == 1.0
    print("[selftest] PASSED")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--step")
    ap.add_argument("--data", default="data/graph-coloring-8/test")
    ap.add_argument("--n", type=int, default=8, help="노드 수 (보드 크기 아님)")
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-scale", type=float, default=1.0)
    ap.add_argument("--sidecar", default="data/graph-coloring-8/sidecar_gc_8.json")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.data, args.sidecar, args.n)
        return
    assert args.ckpt_dir and args.step and args.out, "--ckpt-dir/--step/--out 필수"

    from experiments.gram_harness import load_model, rollout   # GPU 경로에서만 import

    net, _ = load_model(args.ckpt_dir, args.step)
    inputs = np.load(os.path.join(args.data, "all__inputs.npy")).astype(np.int64)
    labels = np.load(os.path.join(args.data, "all__labels.npy")).astype(np.int64)
    B = len(inputs)

    side = json.load(open(args.sidecar))["graphs"]
    hashes = graph_hash_rows(inputs)
    totals = np.array([len(side[h]["colorings"]) if h in side else 0 for h in hashes])

    pri = rollout(net, inputs, labels, K=args.N, sigma_scale_=args.sigma_scale,
                  mode="prior", seed=args.seed, verbose=False)

    cand, v = pri["preds"], pri["v"]
    res = {"checkpoint": os.path.join(args.ckpt_dir, args.step), "n": B, "N": args.N,
           "seed": args.seed, "sigma_scale": args.sigma_scale, "tag": args.tag}

    res["prior_all_samples"] = flat_stats(cand, inputs, args.n)     # ② Conflict

    # ④ canonical majority / ⑤ verifier (LPRM best-of-N)
    sel, majc = [], []
    for b in range(B):
        rows = cand[:, b, :]
        sel.append(rows[int(np.argmax(v[:, b]))])       # ⑤ v = lrpm_head 점수 [N, B]
        # ④ canonical majority: 색 치환된 같은 해를 한 후보로 합친 뒤 최빈을 뽑는다.
        # 원본 토큰 최빈은 색 치환 때문에 사실상 무작위 1개라 논문 지표가 아니다.
        # canonical 화는 색만 재라벨하므로 validity 를 보존한다 -> canonical 보드를 그대로 채점.
        cols = rows[:, :args.n]
        okc = ((cols >= COLOR0) & (cols <= COLOR0 + 2)).all(-1)
        if okc.any():
            canon = canonicalize(cols[okc]) + COLOR0            # [M, n]
            win = majority_row(canon)                           # 최빈 canonical
            row = np.zeros(rows.shape[1], dtype=np.int64)
            row[:args.n] = win
            majc.append(row)
        else:
            majc.append(rows[0])                                # 색토큰이 하나도 없으면 아무거나(invalid)
    sel, majc = np.stack(sel), np.stack(majc)
    res["majority_at_N_valid_canonical"] = float(verify_colorings(majc, inputs, args.n)["valid"].mean())
    res["verifier_selected_at_N_valid"] = float(verify_colorings(sel, inputs, args.n)["valid"].mean())

    res["coverage_at_20"] = coverage_at_N(cand, inputs, args.n, totals)     # ③
    res["metric_defs"] = METRIC_DEFS

    p, c = res["prior_all_samples"], res["coverage_at_20"]
    print(f"{res['checkpoint']}")
    print(f"  prior  conflict={p['conflicts_mean']:.3f}±{p['conflicts_std']:.3f}")
    print(f"  coverage@{args.N}={c['coverage']:.4f} (found {c['found_mean']:.2f}/{c['total_mean']:.2f}, "
          f"gt_total={c['n_found_gt_total']})")
    print(f"  maj_canon@{args.N}={res['majority_at_N_valid_canonical']:.4f} "
          f"vsel@{args.N}={res['verifier_selected_at_N_valid']:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  [out] {args.out}")


if __name__ == "__main__":
    main()
