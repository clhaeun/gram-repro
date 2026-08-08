"""GC conflict 축 단위 규명 -- 논문 baseline 범위(2.7 ~ 199)에 어떤 정의가 들어맞는지.

논문 baseline: Direct Pred 179.3 / Looped TF 136.0 / TRM 109.3 / GRAM 2.7±2.1 (8-vertex).
그런데 gc8 그래프의 전체 edge 는 28 x 0.5455 ~= 15.3 개다.  15개 중 179 위반은
불가능하므로, 우리 harness 의 "위반 edge 수(1 샘플 평균)" 와 논문 축은 다르다.

아래 후보들을 같은 보드에 대해 전부 계산해 값으로 낸다 (판단은 쓰지 않는다):
  (a) 실제 edge 수                     mean/min/max/std          <- 기준선
  (b) 위반 edge 수, 20샘플 평균         <- 현행 harness 축
  (c) 위반 edge 수, 20샘플 합계         (= b x 20)
  (d) 위반률 %                         (위반 edge / 전체 edge x 100)
  (e) 위반에 관여한 노드 수 평균         (충돌 edge 의 양 끝점 unique)
  (f) 잘못 칠해진 노드 수               (전체 proper coloring 중 최소 hamming)
  (g) 무작위 보드에서의 (a)~(f)          <- 상한 감각

용법:
  python experiments/gc_conflict_units.py --ckpt-dir <dir> --step <step> \
      --data data/graph-coloring-8/test --sidecar data/graph-coloring-8/sidecar_gc_8.json \
      --n 8 --N 20 --seed 0 --out reports/gc/units_gc8_ema.json
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.gc_harness import COLOR0, graph_hash_rows   # noqa: E402


def stat(a):
    a = np.asarray(a, dtype=np.float64)
    return dict(mean=float(a.mean()), std=float(a.std()),
                min=float(a.min()), max=float(a.max()))


def edge_pairs(n):
    iu, iv = np.triu_indices(n, 1)
    return iu, iv


def per_sample_metrics(cand, inputs, n, all_colorings):
    """cand [K, B, L] -> (b)(d)(e)(f) 원자료.  반환은 전부 [K, B] 배열."""
    K, B, L = cand.shape
    iu, iv = edge_pairs(n)
    edge = inputs == 2                                  # [B, P]
    n_edges = edge.sum(1)                               # [B]

    viol = np.zeros((K, B), dtype=np.int64)
    vnodes = np.zeros((K, B), dtype=np.int64)
    wrong = np.zeros((K, B), dtype=np.int64)

    for b in range(B):
        e = np.nonzero(edge[b])[0]
        u, v = iu[e], iv[e]
        cols = cand[:, b, :n]                           # [K, n]
        clash = cols[:, u] == cols[:, v]                # [K, E]
        viol[:, b] = clash.sum(1)
        for k in range(K):
            nd = np.unique(np.concatenate([u[clash[k]], v[clash[k]]])) if clash[k].any() else []
            vnodes[k, b] = len(nd)
        # (f) 전체 proper coloring (canonical x 6 perm) 중 최소 hamming
        props = all_colorings[b]                        # [M6, n] 토큰 공간
        d = (cols[:, None, :] != props[None, :, :]).sum(-1)   # [K, M6]
        wrong[:, b] = d.min(1)

    return dict(viol=viol, vnodes=vnodes, wrong=wrong, n_edges=n_edges)


def expand_colorings(side, hashes, n):
    """각 퍼즐의 전체 proper coloring 을 토큰 공간 [M*6, n] 으로."""
    perms = list(itertools.permutations(range(3)))
    out = []
    for h in hashes:
        c = np.array(side[h]["colorings"], dtype=np.int64)          # [M, n] 0..2
        stack = np.concatenate([np.array(p)[c] for p in perms], 0)  # [M*6, n]
        out.append(stack + COLOR0)
    return out


def block(name, m, n_edges, K):
    viol, vnodes, wrong = m["viol"], m["vnodes"], m["wrong"]
    tot = np.tile(n_edges, (K, 1))
    return {
        "b_violated_edges_per_sample": stat(viol),
        "c_violated_edges_sum_over_N": stat(viol.sum(0)),
        "d_violation_rate_pct": stat(100.0 * viol / np.maximum(tot, 1)),
        "e_nodes_in_conflict": stat(vnodes),
        "f_wrongly_colored_nodes": stat(wrong),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", default="data/graph-coloring-8/test")
    ap.add_argument("--sidecar", default="data/graph-coloring-8/sidecar_gc_8.json")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inputs = np.load(os.path.join(args.data, "all__inputs.npy")).astype(np.int64)
    labels = np.load(os.path.join(args.data, "all__labels.npy")).astype(np.int64)
    B, n, K = len(inputs), args.n, args.N
    side = json.load(open(args.sidecar))["graphs"]
    hashes = graph_hash_rows(inputs)
    props = expand_colorings(side, hashes, n)

    n_edges = (inputs == 2).sum(1)
    res = {"checkpoint": os.path.join(args.ckpt_dir, args.step), "n": B, "N": K,
           "seed": args.seed, "nodes": n,
           "a_edges_per_graph": stat(n_edges)}

    from experiments.gram_harness import load_model, rollout   # GPU 경로에서만 import
    net, _ = load_model(args.ckpt_dir, args.step)
    pri = rollout(net, inputs, labels, K=K, mode="prior", seed=args.seed, verbose=False)
    cand = pri["preds"]
    res["model"] = block("model", per_sample_metrics(cand, inputs, n, props), n_edges, K)

    # (g) 무작위 보드 -- gc_harness selftest (b) 와 같은 생성 방식
    rnd = np.zeros_like(cand)
    rnd[:, :, :n] = np.random.default_rng(0).integers(3, 6, size=(K, B, n))
    res["random"] = block("random", per_sample_metrics(rnd, inputs, n, props), n_edges, K)

    # 참고: 라벨(정답) 보드
    lab = np.tile(labels[None], (K, 1, 1))
    res["label"] = block("label", per_sample_metrics(lab, inputs, n, props), n_edges, K)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    a, mo, rn = res["a_edges_per_graph"], res["model"], res["random"]
    print(f"{res['checkpoint']} n={B} nodes={n} N={K}\n"
          f"  (a) edges/graph      {a['mean']:.3f} [{a['min']:.0f}, {a['max']:.0f}] std {a['std']:.3f}\n"
          f"  (b) viol edges/샘플   model {mo['b_violated_edges_per_sample']['mean']:.4f}  "
          f"random {rn['b_violated_edges_per_sample']['mean']:.4f}\n"
          f"  (c) viol edges sum20 model {mo['c_violated_edges_sum_over_N']['mean']:.4f}  "
          f"random {rn['c_violated_edges_sum_over_N']['mean']:.4f}\n"
          f"  (d) viol rate %      model {mo['d_violation_rate_pct']['mean']:.4f}  "
          f"random {rn['d_violation_rate_pct']['mean']:.4f}\n"
          f"  (e) nodes in conflict model {mo['e_nodes_in_conflict']['mean']:.4f}  "
          f"random {rn['e_nodes_in_conflict']['mean']:.4f}\n"
          f"  (f) wrong nodes      model {mo['f_wrongly_colored_nodes']['mean']:.4f}  "
          f"random {rn['f_wrongly_colored_nodes']['mean']:.4f}\n"
          f"  [out] {args.out}")


if __name__ == "__main__":
    main()
