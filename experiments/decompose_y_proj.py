"""y_concat_proj.weight 를 축소 성분과 방향 성분으로 분해한다.

`diag/y_proj_dev` (= ‖W - [I;I]‖_F / ‖[I;I]‖_F) 는 판정에 쓸 수 없다.  AdamW 가
model.parameters() 전체에 weight_decay 를 걸기 때문에(pretrain.py:185, 예외 group 없음)
W 는 gradient 와 무관하게 매 step (1 - lr_t*wd) 배로 줄고, W = a*[I;I] 인 순수 축소만
있어도 dev = |1 - a| 가 된다.  lr=1e-4 / wd=1.0 / 40k step 이면 a -> 0.019 라 dev 가
0.98 까지 올라가는데, 이건 "concat 이 뭔가 배웠다"와 아무 관계가 없다.

그래서 [I;I] 방향으로 사영해 둘을 분리한다:

    alpha    = <W, [I;I]> / ‖[I;I]‖^2 = (tr W[:, :D] + tr W[:, D:]) / (2D)
    residual = ‖W - alpha*[I;I]‖_F / ‖[I;I]‖_F

alpha 는 축소 성분이고 weight decay 예측곡선 prod_t (1 - lr_t*wd) 와 비교하면 된다.
residual 이 "concat 이 add 와 다른 함수를 배웠는가"의 답이다 -- 이것만 판정에 쓴다.
residual 이 평평하면 concat 은 add 의 축소판에 불과하다.

    python experiments/decompose_y_proj.py --run nq8_v2_cat_bm4_b500_s0
    python experiments/decompose_y_proj.py --run nq8_v2_cat_b500_s0 --which ema
    python experiments/decompose_y_proj.py --all --json out.json
"""

import argparse
import glob
import json
import math
import os
import re

import torch

CKPT_ROOT = "checkpoints/Nqueens-8-ACT-torch"


def decay_curve(step, base_lr=1e-4, warmup=500, wd=1.0, lr_min_ratio=1.0):
    """AdamW decoupled decay 누적계수 prod_t (1 - lr_t * wd).

    lr 스케줄은 pretrain.py:211 의 cosine_schedule_with_warmup 이고, 이 실험군은
    lr_min_ratio=1.0 이라 warmup 이후 base_lr 고정이다.
    """
    if lr_min_ratio != 1.0:
        raise NotImplementedError("lr_min_ratio != 1.0 은 cosine 구간을 따로 적분해야 한다")
    logf = 0.0
    for t in range(step):
        lr = base_lr * t / warmup if t < warmup else base_lr
        logf += math.log1p(-lr * wd)
    return math.exp(logf)


def decompose(W):
    """W [D, 2D] -> (alpha, residual, ‖W‖_F, dev).  dev 는 대조용으로만 반환한다."""
    D = W.shape[0]
    ref = math.sqrt(2.0 * D)                      # ‖[I;I]‖_F
    W = W.float()
    alpha = (W[:, :D].diagonal().sum() + W[:, D:].diagonal().sum()).item() / (2.0 * D)

    res = W.clone()
    res[:, :D].diagonal().sub_(alpha)
    res[:, D:].diagonal().sub_(alpha)

    dev = W.clone()
    dev[:, :D].diagonal().sub_(1.0)
    dev[:, D:].diagonal().sub_(1.0)

    return alpha, (res.norm() / ref).item(), W.norm().item(), (dev.norm() / ref).item()


def scan_run(run, which="live"):
    """run 의 저장된 모든 eval checkpoint 에서 분해값을 뽑는다."""
    d = os.path.join(CKPT_ROOT, run)
    suffix = ".live" if which == "live" else ""
    pat = os.path.join(d, "step_*.live") if which == "live" else os.path.join(d, "step_*")
    files = [f for f in glob.glob(pat)
             if (f.endswith(".live") if which == "live" else not f.endswith(".live"))]
    steps = sorted(int(re.search(r"step_(\d+)", os.path.basename(f)).group(1)) for f in files)

    rows = []
    for s in steps:
        sd = torch.load(os.path.join(d, f"step_{s}{suffix}"), map_location="cpu", weights_only=False)
        key = [k for k in sd if k.endswith("y_concat_proj.weight")]
        bias = [v for k, v in sd.items() if k.endswith("prior_logvar_bias")]
        row = {"step": s, "logvar_bias": bias[0].float().item() if bias else None}
        if key:
            a, r, n, dv = decompose(sd[key[0]])
            row.update(alpha=a, residual=r, w_norm=n, y_proj_dev=dv,
                       alpha_pred=decay_curve(s))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--all", action="store_true", help="checkpoints 아래 concat run 전부")
    ap.add_argument("--which", choices=["live", "ema"], default="live",
                    help="live = raw SGD 가중치(기본). ema = 편향보정 EMA 사본")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    runs = list(args.run)
    if args.all:
        for d in sorted(glob.glob(os.path.join(CKPT_ROOT, "*"))):
            cks = glob.glob(os.path.join(d, "step_*"))
            if not cks:
                continue
            sd = torch.load(sorted(cks)[0], map_location="cpu", weights_only=False)
            if any(k.endswith("y_concat_proj.weight") for k in sd):
                runs.append(os.path.basename(d))
    if not runs:
        ap.error("--run 또는 --all 이 필요하다")

    out = {}
    for run in runs:
        rows = scan_run(run, args.which)
        out[run] = rows
        print(f"\n=== {run}  ({args.which}) ===")
        if not rows:
            print("  체크포인트 없음")
            continue
        if "alpha" not in rows[0]:
            print("  y_concat_proj 없음 (add arm)")
            continue
        print(f"{'step':>7s} {'alpha':>9s} {'alpha_pred':>10s} {'alpha/pred':>10s} "
              f"{'residual':>9s} {'‖W‖_F':>9s} {'(dev)':>8s} {'logvar_bias':>11s}")
        for r in rows:
            print(f"{r['step']:7d} {r['alpha']:9.6f} {r['alpha_pred']:10.6f} "
                  f"{r['alpha']/r['alpha_pred']:10.4f} {r['residual']:9.6f} "
                  f"{r['w_norm']:9.4f} {r['y_proj_dev']:8.5f} {r['logvar_bias']:+11.6f}")
        first, last = rows[0], rows[-1]
        print(f"  residual {first['residual']:.6f} -> {last['residual']:.6f}  "
              f"(x{last['residual']/first['residual']:.2f})   "
              f"alpha {first['alpha']:.4f} -> {last['alpha']:.4f}")

    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
        print(f"\nsaved -> {args.json}")


if __name__ == "__main__":
    main()
