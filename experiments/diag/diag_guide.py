"""guide(prior_mu) 생사 진단 -- Task 2(mu/sigma 크기) + Task 3(해 개수 버킷 분해).

Task 1(3모드 harness)은 sweep_harness.py 에 추가한 --mu-scale / 기존 --sigma-scale 로 돈다.
이 스크립트는 GPU 1장, inference 전용, 학습/체크포인트를 건드리지 않는다.

사용:
  python experiments/diag_guide.py --ckpt-dir <dir> --step step_N.live \
      --out experiments/results/guide_<tag>.json [--probe-n 256] [--N 20]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

# experiments/diag/ 로 한 단계 내려왔으므로 repo root 는 부모의 부모의 부모다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.gram_harness import load_model, rollout   # noqa: E402
from evaluators.nqueens import verify_boards               # noqa: E402


def probe_mu_sigma(net, inputs, labels, n_probe, seed=0):
    """prior 경로 forward 를 돌며 reparameterize 호출마다 mu/sigma 크기를 기록.

    호출은 transition 순서로 도착한다(H_cycles per forward) -- gram_harness.capture_sigma
    의 주석과 같은 규약. index % H_cycles 가 transition depth.

    ||mu||   : per-token L2 norm (hidden dim 전체), 배치·토큰 평균
    ||sigma||: per-token L2 norm of the std vector = sqrt(sum_i sigma_i^2)
               = 노이즈 항 sigma*eps 의 기대 노름 스케일. ||mu|| 와 같은 축이라 비교 가능.
    sigma_elem: sigma 의 원소별 평균 (참고용, 축이 다름)
    """
    inner = net.model.inner
    orig = inner.reparameterize
    rec = []

    def patched(mu, logvar):
        m = mu.detach().float()
        std = (0.5 * logvar.detach().float()).exp() * inner.sigma_scale
        rec.append(dict(
            mu_norm=m.norm(dim=-1).mean().item(),
            sigma_norm=std.norm(dim=-1).mean().item(),
            sigma_elem=std.mean().item(),
            mu_elem_abs=m.abs().mean().item(),
        ))
        return orig(mu, logvar)

    inner.reparameterize = patched
    try:
        rollout(net, inputs[:n_probe], labels[:n_probe], K=1, mode="prior",
                seed=seed, verbose=False, batch_size=n_probe)
    finally:
        inner.reparameterize = orig

    H = int(net.model.inner.config.H_cycles)
    by_t = {}
    for i, r in enumerate(rec):
        by_t.setdefault(i % H, []).append(r)
    out = []
    for t in sorted(by_t):
        g = by_t[t]
        mu_n = float(np.mean([x["mu_norm"] for x in g]))
        sg_n = float(np.mean([x["sigma_norm"] for x in g]))
        out.append(dict(transition=t, n_calls=len(g),
                        mu_norm=mu_n, sigma_norm=sg_n,
                        ratio_mu_over_sigma=(mu_n / sg_n if sg_n > 0 else float("inf")),
                        sigma_elem=float(np.mean([x["sigma_elem"] for x in g])),
                        mu_elem_abs=float(np.mean([x["mu_elem_abs"] for x in g]))))
    return dict(H_cycles=H, total_calls=len(rec), per_transition=out)


def solution_counts(inputs, sidecar_path):
    """test input 각각의 해 개수를 sidecar 에서 조회.

    sidecar meta: hash = 'sha1(flat token board uint8 bytes).hexdigest()[:16]'
    """
    sc = json.load(open(sidecar_path))
    tbl = sc["inputs"]
    counts = np.full(len(inputs), -1, dtype=np.int64)
    for i, row in enumerate(inputs):
        h = hashlib.sha1(row.astype(np.uint8).tobytes()).hexdigest()[:16]
        e = tbl.get(h)
        if e is not None:
            counts[i] = len(e["solutions"])
    return counts, sc.get("meta", {})


BUCKETS = [("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20), ("21+", 21, 10 ** 9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", default="data/nqueens-8/test")
    ap.add_argument("--sidecar", default="data/nqueens-8/sidecar_8.json")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-n", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    net, _ = load_model(args.ckpt_dir, args.step)
    inputs = np.load(f"{args.data}/all__inputs.npy").astype(np.int64)
    labels = np.load(f"{args.data}/all__labels.npy").astype(np.int64)
    B = len(inputs)

    res = {"checkpoint": os.path.join(args.ckpt_dir, args.step), "n_eval": B,
           "N": args.N, "seed": args.seed}

    # ---- Task 2: mu / sigma 크기 -------------------------------------------
    res["mu_sigma"] = probe_mu_sigma(net, inputs, labels, args.probe_n, seed=args.seed)
    print(f"[mu/sigma] H_cycles={res['mu_sigma']['H_cycles']} calls={res['mu_sigma']['total_calls']}")
    for r in res["mu_sigma"]["per_transition"]:
        print(f"  t={r['transition']}  ||mu||={r['mu_norm']:.4f}  ||sigma||={r['sigma_norm']:.4f}  "
              f"ratio={r['ratio_mu_over_sigma']:.4f}  (sigma_elem={r['sigma_elem']:.5f})")

    # ---- Task 3: 해 개수 버킷 분해 (정상 모드) ------------------------------
    pri = rollout(net, inputs, labels, K=args.N, mode="prior", seed=args.seed, verbose=False)
    cand = pri["preds"]                                        # [N, B, L]
    vb = verify_boards(cand.reshape(args.N * B, -1), np.tile(inputs, (args.N, 1)), args.n)
    valid = vb["valid"].reshape(args.N, B).T                   # [B, N]

    res["overall_valid"] = float(valid.mean())
    res["overall_pass_at_N"] = float((valid.sum(1) > 0).mean())

    counts, meta = solution_counts(inputs, args.sidecar)
    res["sidecar_meta"] = meta
    res["sidecar_hit_rate"] = float((counts >= 0).mean())
    print(f"[bucket] sidecar 조회 성공률 {res['sidecar_hit_rate']*100:.2f}%  "
          f"(전체 {B}, 실패 {(counts < 0).sum()})")

    buckets = []
    for name, lo, hi in BUCKETS:
        m = (counts >= lo) & (counts <= hi)
        if m.sum() == 0:
            buckets.append(dict(bucket=name, n_puzzles=0)); continue
        buckets.append(dict(bucket=name, n_puzzles=int(m.sum()),
                            valid_rate=float(valid[m].mean()),
                            pass_at_N=float((valid[m].sum(1) > 0).mean())))
    miss = (counts < 0)
    if miss.sum():
        buckets.append(dict(bucket="조회실패", n_puzzles=int(miss.sum()),
                            valid_rate=float(valid[miss].mean()),
                            pass_at_N=float((valid[miss].sum(1) > 0).mean())))
    res["buckets"] = buckets
    for b in buckets:
        if b["n_puzzles"]:
            print(f"  {b['bucket']:>6}: n={b['n_puzzles']:5d}  valid={b['valid_rate']*100:6.2f}%  "
                  f"pass@{args.N}={b['pass_at_N']*100:6.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  [out] {args.out}")


if __name__ == "__main__":
    main()
