"""N-Queens harness. --ckpt-dir <dir> --step step_N --out <json>"""
import argparse, hashlib, json, os, sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.gram_harness import load_model, rollout   # noqa: E402
from evaluators.nqueens import queen_key, verify_boards    # noqa: E402


def flat_stats(preds, inputs, n):
    K, B, L = preds.shape
    r = verify_boards(preds.reshape(K * B, L), np.tile(inputs, (K, 1)), n)
    qc = r["queen_count"]
    return dict(valid=float(r["valid"].mean()),
                queen_count_mean=float(qc.mean()),
                queen_count_min=int(qc.min()), queen_count_max=int(qc.max()))


def majority_row(rows):
    return np.array(Counter(tuple(r) for r in rows).most_common(1)[0][0], dtype=np.int64)


def solution_counts(inputs, sidecar_path):
    tbl = json.load(open(sidecar_path))["inputs"]
    out = np.full(len(inputs), -1, dtype=np.int64)
    for i, row in enumerate(inputs):
        e = tbl.get(hashlib.sha1(row.astype(np.uint8).tobytes()).hexdigest()[:16])
        if e is not None:
            out[i] = len(e["solutions"])
    return out


SOL_BUCKETS = [("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20), ("21+", 21, 10 ** 9)]


def coverage_at_N(cand, inputs, n, totals):
    N, B, L = cand.shape
    r = verify_boards(cand.reshape(N * B, L), np.tile(inputs, (N, 1)), n)
    ok = (r["valid"] & r["consistent"]).reshape(N, B)
    found = np.zeros(B, dtype=np.int64)
    for b in range(B):
        found[b] = len({queen_key(cand[k, b]) for k in range(N) if ok[k, b]})
    have = totals > 0
    cov = np.zeros(B, dtype=np.float64)
    cov[have] = found[have] / totals[have]
    res = dict(
        coverage=float(cov[have].mean()) if have.any() else None,
        n_puzzles_scored=int(have.sum()), n_puzzles_no_total=int((~have).sum()),
        found_mean=float(found[have].mean()) if have.any() else None,
        total_mean=float(totals[have].mean()) if have.any() else None,
        n_found_gt_total=int((found[have] > totals[have]).sum()))
    res["by_solution_bucket"] = [                       # 논문 Fig 15 대응 (JSON 전용)
        dict(bucket=name, n_puzzles=int(m.sum()),
             coverage=(float(cov[m].mean()) if m.any() else None),
             found_mean=(float(found[m].mean()) if m.any() else None),
             total_mean=(float(totals[m].mean()) if m.any() else None))
        for name, lo, hi in SOL_BUCKETS for m in [have & (totals >= lo) & (totals <= hi)]]
    ihash = [hashlib.sha1(v.astype(np.uint8).tobytes()).hexdigest()[:16] for v in inputs]
    res["per_row"] = {
        "ihash": ihash, "totals": totals.tolist(), "found": found.tolist()}
    agg = {}
    for h, t, f in zip(ihash, totals.tolist(), found.tolist()):
        if t > 0:
            a = agg.setdefault(h, [0, 0]); a[0] += f; a[1] += t
    res["coverage_macro_by_input"] = (
        float(np.mean([min(f, t) / t for f, t in agg.values()])) if agg else None)
    res["n_inputs_scored"] = len(agg)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", default="data/nqueens-8/test")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-scale", type=float, default=1.0)  # 0 = guide only
    ap.add_argument("--mu-scale", type=float, default=1.0)     # 0 = stochasticity only
    ap.add_argument("--sidecar", default="data/nqueens-8/sidecar_8.json")
    ap.add_argument("--halt-max-steps", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    net, _ = load_model(args.ckpt_dir, args.step)
    if args.halt_max_steps is not None:
        net.model.config.halt_max_steps = args.halt_max_steps
    inputs = np.load(f"{args.data}/all__inputs.npy").astype(np.int64)
    labels = np.load(f"{args.data}/all__labels.npy").astype(np.int64)
    B = len(inputs)

    pri = rollout(net, inputs, labels, K=args.N, sigma_scale_=args.sigma_scale,
                  mode="prior", seed=args.seed, verbose=False, mu_scale_=args.mu_scale)
    cand, v = pri["preds"], pri["v"]                           # [N, B, L], [N, B]

    res = {"checkpoint": os.path.join(args.ckpt_dir, args.step), "n": B, "N": args.N,
           "seed": args.seed, "sigma_scale": args.sigma_scale, "mu_scale": args.mu_scale,
           "halt_max_steps": net.model.config.halt_max_steps}
    res["prior_all_samples"] = flat_stats(cand, inputs, args.n)                  # ①
    maj = np.stack([majority_row(cand[:, b, :]) for b in range(B)])              # ④
    sel = np.stack([cand[int(np.argmax(v[:, b])), b, :] for b in range(B)])      # ⑤ LPRM best-of-N
    res["majority_at_N_valid"] = float(verify_boards(maj, inputs, args.n)["valid"].mean())
    res["verifier_selected_at_N_valid"] = float(verify_boards(sel, inputs, args.n)["valid"].mean())
    res["coverage_at_20"] = coverage_at_N(cand, inputs, args.n,
                                          solution_counts(inputs, args.sidecar))  # ③

    c = res["coverage_at_20"]
    if c["n_found_gt_total"]:
        print(f"WARNING: n_found_gt_total={c['n_found_gt_total']} (0 이어야 정상)")
    cov_micro = c["found_mean"] / c["total_mean"]
    print(f"{res['checkpoint']}  hms={res['halt_max_steps']} "
          f"acc={res['prior_all_samples']['valid']:.4f} "
          f"cov_row_macro@{args.N}={c['coverage']:.4f} "
          f"cov_micro@{args.N}={cov_micro:.4f} "
          f"cov_puzzle_macro@{args.N}={c['coverage_macro_by_input']:.4f} "
          f"maj={res['majority_at_N_valid']:.4f} "
          f"vsel={res['verifier_selected_at_N_valid']:.4f} "
          f"qc={res['prior_all_samples']['queen_count_mean']:.4f}  [out] {args.out}")
    try:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
    except OSError as e:
        print(f"!!! WRITE FAILED {args.out}: {e}\n"
              f"!!! 아래 RESULT_JSON 한 줄이 전량 산출물이다 (복구용)", flush=True)
        print("RESULT_JSON " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
