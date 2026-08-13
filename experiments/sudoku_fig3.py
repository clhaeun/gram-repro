"""Sudoku Figure 3 대조 셀 harness (--data / --limit / GPU 샤딩).

채점 로직·RNG 스트림은 재구현하지 않고 legacy 의 _make_batch / _forward_once /
SudokuTask / compute_metrics 를 import 해 쓴다.  표본은 --data + --limit 연속
슬라이싱으로 뽑는다 (legacy 쪽 rng.choice 규약이 아니다).

샤딩 주의: RNG 는 candidate k 마다 seed*100003+k 로 재시드되고 배치 순서대로 randn 을
소비하므로 행->노이즈 대응이 샤드 경계와 batch_size 에 의존한다.  레이아웃이 다르면
평균은 표본오차 안에서 일치하지만 비트 동일하지는 않다.

사용:
  # 샤드 (GPU 마다 1개)
  CUDA_VISIBLE_DEVICES=4 python experiments/sudoku_fig3.py --ckpt-dir <dir> \
      --step step_65100 --data data/sudoku-extreme-1k-aug-1000/test \
      --limit 20000 --N 20 --halt-max-steps 16 --seed 0 \
      --shard 0 --num-shards 4 --out results/fig3/cell.json
  # 병합
  python experiments/sudoku_fig3.py --merge results/fig3/cell.json --num-shards 4
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.legacy.eval_harness import (            # noqa: E402
    SudokuTask, _forward_once, _make_batch, compute_metrics, print_report)
from experiments.legacy.gram_harness_legacy import (      # noqa: E402
    load_model, sigma_scale as _sigma_scale_ctx)


def load_block(data_dir, offset, limit):
    """test 디렉토리에서 연속 블록 [offset, offset+limit) 를 읽는다.  난수 없음."""
    inputs = np.load(os.path.join(data_dir, "all__inputs.npy"))
    labels = np.load(os.path.join(data_dir, "all__labels.npy"))
    hi = len(inputs) if limit is None else min(offset + limit, len(inputs))
    inp, lab = inputs[offset:hi], labels[offset:hi]
    # md5 는 raw uint8 바이트 기준 (34번 규약: [0,4000) -> 9bb82c414a70)
    md5 = hashlib.md5(inp.astype(np.uint8).tobytes()).hexdigest()
    return inp.astype(np.int64), lab.astype(np.int64), md5, len(inputs)


def shard_range(n, shard, num_shards):
    """행을 연속 블록으로 등분.  나머지는 앞 샤드에 1행씩 배분."""
    base, rem = divmod(n, num_shards)
    lo = shard * base + min(shard, rem)
    hi = lo + base + (1 if shard < rem else 0)
    return lo, hi


def sample_rows(net, inputs, labels, N, batch_size, seed, sigma):
    """[n, L] -> cand [n, N, L] int64, v [n, N] float32.  legacy sample_dataset 와
    같은 순서 (candidate outer, batch inner, k 마다 재시드)."""
    n, seq_len = inputs.shape
    cand = np.zeros((n, N, seq_len), dtype=np.int64)
    v = np.zeros((n, N), dtype=np.float32)
    import torch
    with _sigma_scale_ctx(net, sigma):
        for k in range(N):
            torch.random.manual_seed(seed * 100003 + k)
            t0 = time.time()
            for lo in range(0, n, batch_size):
                hi = min(lo + batch_size, n)
                batch = _make_batch(inputs, labels, lo, hi, batch_size, seq_len)
                p, vv = _forward_once(net, batch, seq_len)
                cand[lo:hi, k] = p[: hi - lo]
                v[lo:hi, k] = vv[: hi - lo]
            print(f"  cand {k + 1}/{N}  {time.time() - t0:.1f}s", flush=True)
    return cand, v


def shard_path(out, shard):
    return os.path.splitext(out)[0] + f".shard{shard}.npz"


def run_shard(args):
    net, config = load_model(args.ckpt_dir, args.step)
    if args.halt_max_steps is not None:
        net.model.config.halt_max_steps = args.halt_max_steps   # inner 와 공유 객체
    hms = net.model.config.halt_max_steps

    inp_all, lab_all, md5_all, n_total = load_block(args.data, args.offset, args.limit)
    lo, hi = shard_range(len(inp_all), args.shard, args.num_shards)
    inp, lab = inp_all[lo:hi], lab_all[lo:hi]
    print(f"shard {args.shard}/{args.num_shards}  rows [{args.offset + lo}, {args.offset + hi}) "
          f"n={len(inp)}  hms={hms}  block_md5={md5_all[:12]}  test_rows={n_total}", flush=True)

    t0 = time.time()
    cand, v = sample_rows(net, inp, lab, args.N, args.batch_size, args.seed, args.sigma_scale)
    el = time.time() - t0

    p = shard_path(args.out, args.shard)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    np.savez_compressed(
        p, cand=cand.astype(np.uint8), v=v,
        inputs=inp.astype(np.uint8), labels=lab.astype(np.uint8),
        meta=np.array(json.dumps(dict(
            shard=args.shard, num_shards=args.num_shards, lo=int(lo), hi=int(hi),
            offset=args.offset, limit=args.limit, N=args.N, seed=args.seed,
            halt_max_steps=int(hms), sigma_scale=args.sigma_scale,
            batch_size=args.batch_size, data=args.data, block_md5=md5_all,
            ckpt=os.path.join(args.ckpt_dir, args.step), elapsed_s=el))))
    print(f"[shard {args.shard}] {el:.1f}s -> {p}", flush=True)


def run_merge(args):
    parts = []
    for s in range(args.num_shards):
        p = shard_path(args.merge, s)
        if not os.path.exists(p):
            raise SystemExit(f"missing shard: {p}")
        parts.append(np.load(p, allow_pickle=False))
    metas = [json.loads(str(z["meta"])) for z in parts]
    order = np.argsort([m["lo"] for m in metas])
    parts = [parts[i] for i in order]
    metas = [metas[i] for i in order]

    for key in ("N", "seed", "halt_max_steps", "sigma_scale", "offset", "limit",
                "block_md5", "ckpt", "data"):
        vals = {m[key] for m in metas}
        if len(vals) != 1:
            raise SystemExit(f"shard 간 {key} 불일치: {vals}")

    cand = np.concatenate([z["cand"] for z in parts]).astype(np.int64)
    v = np.concatenate([z["v"] for z in parts])
    inputs = np.concatenate([z["inputs"] for z in parts]).astype(np.int64)
    labels = np.concatenate([z["labels"] for z in parts]).astype(np.int64)

    # 샤드가 원 블록을 빠짐없이/겹침없이 덮었는지
    cov = sorted((m["lo"], m["hi"]) for m in metas)
    exp = 0
    for lo, hi in cov:
        if lo != exp:
            raise SystemExit(f"샤드 커버리지 구멍/겹침: {cov}")
        exp = hi
    if exp != len(cand):
        raise SystemExit(f"행 수 불일치: {exp} != {len(cand)}")

    m0 = metas[0]
    ks = tuple(int(x) for x in args.ks.split(","))
    m = compute_metrics(cand, v, inputs, labels, SudokuTask(), ks=ks)
    m.update({k: m0[k] for k in ("N", "seed", "halt_max_steps", "sigma_scale",
                                 "offset", "limit", "block_md5", "ckpt", "data")})
    m["block_md5_12"] = m0["block_md5"][:12]
    m["merged_inputs_md5_12"] = hashlib.md5(
        inputs.astype(np.uint8).tobytes()).hexdigest()[:12]
    m["num_shards"] = args.num_shards
    m["shard_rows"] = [[mm["lo"], mm["hi"]] for mm in metas]
    m["elapsed_s_per_shard"] = [mm["elapsed_s"] for mm in metas]
    # verifier 축 (solved) = SudokuTask.verify: 유효 격자 AND 힌트 일치
    print_report(m)
    with open(args.merge, "w") as f:
        json.dump(m, f, indent=2)
    print(f"\nsaved -> {args.merge}")
    k = str(m0["N"])
    r = m["by_k"][k]
    print(f"\nCELL  ckpt={m0['ckpt']} hms={m0['halt_max_steps']} N={m0['N']} "
          f"seed={m0['seed']} n={m['n']} md5={m['block_md5_12']}\n"
          f"  exact_single={m['single']:.4f} exact_mean={m['mean_sample']:.4f} "
          f"solved_mean={m['verifier_pass_rate']:.4f}\n"
          f"  maj@{k}={r['majority']:.4f} vsel_lprm@{k}={r['lprm_best']:.4f} "
          f"vsel_verif@{k}={r['verifier_selected']:.4f} pass@{k}={r['pass']:.4f} "
          f"passfirst@{k}={r['pass_first_k']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--step", default="step_65100")
    ap.add_argument("--data", default="data/sudoku-extreme-1k-aug-1000/test")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-scale", type=float, default=1.0)
    ap.add_argument("--halt-max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--ks", default="1,5,20")
    ap.add_argument("--out")
    ap.add_argument("--merge", help="샤드 병합 모드: 결과 JSON 경로")
    args = ap.parse_args()

    if args.merge:
        run_merge(args)
        return
    if not (args.ckpt_dir and args.out):
        raise SystemExit("--ckpt-dir / --out 필요 (또는 --merge)")
    run_shard(args)


if __name__ == "__main__":
    main()
