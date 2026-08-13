"""Graph-Coloring harness. --ckpt-dir <dir> --step step_N --out <json>"""
import argparse, hashlib, itertools, json, os, sys
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluators.graph_coloring import verify_colorings   # noqa: E402

COLOR0 = 3

def majority_row(rows):
    return np.array(Counter(tuple(r) for r in rows).most_common(1)[0][0], dtype=np.int64)

def canonicalize(colors):
    c = colors - COLOR0
    out = np.empty_like(c)
    for i, r in enumerate(c):
        remap = {}
        for j, v in enumerate(r): out[i, j] = remap.setdefault(v, len(remap))
    return out

def graph_hash_rows(inputs):
    return [hashlib.sha1(row.astype(np.uint8).tobytes()).hexdigest()[:16] for row in inputs]

def conflict_stats(preds, inputs, n):
    M = preds.shape[0]
    r = verify_colorings(preds.reshape(-1, preds.shape[-1]), np.tile(inputs, (M, 1)), n)
    c, ok = r["conflicts"], r["colors_ok"]
    return dict(
        conflicts_mean=float(c.mean()), conflicts_std=float(c.std()),
        conflicts_std_across_puzzles=float(c.reshape(M, -1).mean(0).std()),
        colors_ok_rate=float(ok.mean()),
        conflicts_mean_colors_ok=float(c[ok].mean()) if ok.any() else None,
        conflicts_std_colors_ok=float(c[ok].std()) if ok.any() else None,
        valid_rate=float(r["valid"].mean()),
        num_colors_used_mean=float(r["num_colors_used"].mean()),
        n_samples=int(M), n_scored=int(c.size))

def coverage_at_N(cand, inputs, n, totals):
    N, B, L = cand.shape
    ok = verify_colorings(cand.reshape(N * B, L), np.tile(inputs, (N, 1)), n)["valid"].reshape(N, B)
    found = np.array([len({tuple(c) for c in canonicalize(np.stack(g))})
                      if (g := [cand[k, b, :n] for k in range(N) if ok[k, b]]) else 0
                      for b in range(B)], dtype=np.int64)
    have = totals > 0; cov = np.zeros(B, dtype=np.float64); cov[have] = found[have] / totals[have]
    res = dict(coverage=float(cov[have].mean()) if have.any() else None,
               n_puzzles_scored=int(have.sum()), n_puzzles_no_total=int((~have).sum()),
               found_mean=float(found[have].mean()) if have.any() else None,
               total_mean=float(totals[have].mean()) if have.any() else None,
               n_found_gt_total=int((found[have] > totals[have]).sum()))
    res["by_solution_bucket"] = {                        # 논문 Fig 15 대응 (JSON 전용)
        name: dict(n=int(m.sum()), coverage=float(cov[m].mean()),
                   found_mean=float(found[m].mean()), total_mean=float(totals[m].mean()))
        for lo, hi, name in [(1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"), (21, 10**9, "21+")]
        for m in [have & (totals >= lo) & (totals <= hi)] if m.any()}
    return res

def selftest(data_dir, sidecar_path, n):
    inputs = np.load(os.path.join(data_dir, "all__inputs.npy")).astype(np.int64); B, L = inputs.shape
    labels = np.load(os.path.join(data_dir, "all__labels.npy")).astype(np.int64)
    side = json.load(open(sidecar_path))["graphs"]; hashes = graph_hash_rows(inputs)
    assert not any(h not in side for h in hashes), "graph hash 가 sidecar 와 안 맞는다"
    totals = np.array([len(side[h]["colorings"]) for h in hashes])
    lab = conflict_stats(labels[None], inputs, n)
    assert lab["conflicts_mean"] == 0.0 and lab["colors_ok_rate"] == 1.0 and lab["valid_rate"] == 1.0
    assert lab["conflicts_std_across_puzzles"] == 0.0
    rnd = np.zeros((20, B, L), dtype=np.int64)
    rnd[:, :, :n] = np.random.default_rng(0).integers(3, 6, size=(20, B, n))
    assert abs(conflict_stats(rnd, inputs, n)["conflicts_mean"] - float((inputs == 2).sum(1).mean()) / 3.0) < 0.15
    K = int(totals.max()); cand = np.zeros((K, B, L), dtype=np.int64)
    for b, h in enumerate(hashes):
        sols = np.array(side[h]["colorings"], dtype=np.int64) + COLOR0
        cand[:, b, :n] = sols[np.minimum(np.arange(K), len(sols) - 1)]
    cov = coverage_at_N(cand, inputs, n, totals); assert cov["coverage"] == 1.0 and cov["n_found_gt_total"] == 0
    cand = np.zeros((6, 1, L), dtype=np.int64)
    for k, p in enumerate(itertools.permutations(range(3))):
        cand[k, 0, :n] = np.array(p)[labels[0, :n] - COLOR0] + COLOR0
    assert coverage_at_N(cand, inputs[:1], n, totals[:1])["found_mean"] == 1.0
    bad = labels.copy(); bad[:, 0] = 0
    e = conflict_stats(bad[None], inputs, n)
    assert e["colors_ok_rate"] == 0.0 and e["valid_rate"] == 0.0
    assert e["conflicts_mean"] == 0.0 and e["conflicts_mean_colors_ok"] is None
    print("[selftest] PASSED (a 라벨=conflict0, b 무작위=edges/3, c 전해=cov1.0, d 색치환6=found1, "
          "e 불법토큰=conflict0 이지만 colors_ok0)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--ckpt-dir")
    ap.add_argument("--step"); ap.add_argument("--out")
    ap.add_argument("--data", default="data/graph-coloring-8/test")
    ap.add_argument("--sidecar", default="data/graph-coloring-8/sidecar_gc_8.json")
    ap.add_argument("--n", type=int, default=8, help="노드 수"); ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-scale", type=float, default=1.0)
    args = ap.parse_args()
    if args.selftest: return selftest(args.data, args.sidecar, args.n)
    assert args.ckpt_dir and args.step and args.out, "--ckpt-dir/--step/--out 필수"
    from experiments.gram_harness import load_model, rollout
    net, _ = load_model(args.ckpt_dir, args.step)
    inputs = np.load(os.path.join(args.data, "all__inputs.npy")).astype(np.int64); B = len(inputs)
    labels = np.load(os.path.join(args.data, "all__labels.npy")).astype(np.int64)
    side = json.load(open(args.sidecar))["graphs"]
    totals = np.array([len(side[h]["colorings"]) if h in side else 0 for h in graph_hash_rows(inputs)])
    pri = rollout(net, inputs, labels, K=args.N, sigma_scale_=args.sigma_scale,
                  mode="prior", seed=args.seed, verbose=False)
    cand, v = pri["preds"], pri["v"]
    res = {"checkpoint": os.path.join(args.ckpt_dir, args.step), "n": B, "N": args.N,
           "n_puzzles": B, "n_nodes": args.n,
           "seed": args.seed, "sigma_scale": args.sigma_scale}
    sel, majc = [], []
    for b in range(B):
        rows = cand[:, b, :]
        sel.append(rows[int(np.argmax(v[:, b]))])                                 # ⑤ LPRM best-of-N
        cols = rows[:, :args.n]                                                   # ④ canonical majority
        okc = ((cols >= COLOR0) & (cols <= COLOR0 + 2)).all(-1)
        row = np.zeros(rows.shape[1], dtype=np.int64)
        if okc.any(): row[:args.n] = majority_row(canonicalize(cols[okc]) + COLOR0)
        majc.append(row if okc.any() else rows[0])
    sel, majc = np.stack(sel), np.stack(majc)

    res["conflict"] = {"mean": conflict_stats(cand, inputs, args.n),
                       "maj":  conflict_stats(majc[None], inputs, args.n),
                       "lprm": conflict_stats(sel[None], inputs, args.n)}
    res["prior_all_samples"] = res["conflict"]["mean"]
    res["conflict_mean"] = res["conflict"]["mean"]["conflicts_mean"]
    res["conflict_maj"] = res["conflict"]["maj"]["conflicts_mean"]
    res["conflict_lprm"] = res["conflict"]["lprm"]["conflicts_mean"]
    res["majority_at_N_valid_canonical"] = res["conflict"]["maj"]["valid_rate"]
    res["verifier_selected_at_N_valid"] = res["conflict"]["lprm"]["valid_rate"]
    res["coverage_at_20"] = coverage_at_N(cand, inputs, args.n, totals)           # ③
    c, p = res["coverage_at_20"], res["prior_all_samples"]
    if c["n_found_gt_total"]: print(f"WARNING: n_found_gt_total={c['n_found_gt_total']} (0 이어야 정상)")
    for k in ("mean", "maj", "lprm"):
        if res["conflict"][k]["colors_ok_rate"] < 1.0:
            s = res["conflict"][k]
            print(f"WARNING: conflict[{k}] colors_ok_rate={s['colors_ok_rate']:.4f} < 1 "
                  f"-- conflicts_mean={s['conflicts_mean']:.3f} 은 불법 토큰만큼 과소계상이다 "
                  f"(legal 조건부 {s['conflicts_mean_colors_ok']})")
    print(f"{res['checkpoint']}  conflict mean={p['conflicts_mean']:.3f}±{p['conflicts_std']:.3f} "
          f"maj={res['conflict_maj']:.3f} lprm={res['conflict_lprm']:.3f}  "
          f"cov@{args.N}={c['coverage']:.4f} maj_canon={res['majority_at_N_valid_canonical']:.4f} "
          f"vsel={res['verifier_selected_at_N_valid']:.4f}  [out] {args.out}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f: json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
