"""y_emb vs z_H 스케일 측정 + posterior 입력 불변성 테스트 (N-Queens / Sudoku 공통).

전부 읽기 전용이다. 학습 코드는 수정하지 않고, 런타임에 `_y_embeddings` 를 감싸고
`post_mu` 에 forward hook 을 걸어 `u_y = z_H + y_emb` (gram.py:337 / :363) 시점의
두 항을 분리해 노름을 잰다. 측정이 끝나면 전부 원복하고 assert 로 확인한다.

z_H 복원: hook 이 받는 입력이 u_y 이므로 z_H = u_y - y_emb.

pad 주의: `_y_embeddings`(gram.py:229-230)는 앞쪽 puzzle_emb_len 위치를 0으로 pad 한다.
따라서 전 토큰 평균은 그 0들 때문에 아래로 끌린다. 두 값을 따로 보고한다.

사용법:
    python experiments/diag_yemb.py --ckpt-dir <dir> --step <step> --task nqueens
    python experiments/diag_yemb.py --ckpt-dir <dir> --step <step> --task sudoku
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

# experiments/diag/ 로 한 단계 내려왔으므로 repo root 는 부모의 부모의 부모다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.gram_harness import load_model, rollout   # noqa: E402


def measure_scales(net, inner, batch, sup_steps):
    """transition별 ||z_H||, ||y_emb||, 비율."""
    st = {"y": None}
    o_yemb = inner._y_embeddings

    def yemb(labels):
        st["y"] = o_yemb(labels)
        return st["y"]

    rows = []
    busy = {"v": False}

    def hook(mod, args, out):
        if busy["v"]:
            return
        busy["v"] = True
        try:
            u_y = args[0].float()
            y = st["y"].float()
            z = u_y - y
            zn, yn = z.norm(dim=-1), y.norm(dim=-1)          # [B, L]
            nz = yn > 0                                      # pad 아닌 위치
            rows.append(dict(
                zH_norm_all=float(zn.mean()), y_norm_all=float(yn.mean()),
                zH_norm_nonpad=float(zn[nz].mean()), y_norm_nonpad=float(yn[nz].mean()),
                ratio_all=float(yn.mean() / zn.mean().clamp_min(1e-9)),
                ratio_nonpad=float(yn[nz].mean() / zn[nz].mean().clamp_min(1e-9)),
                n_pad_pos=int((~nz).sum(1).float().mean()), n_pos=int(zn.shape[1]),
            ))
        finally:
            busy["v"] = False

    inner._y_embeddings = yemb
    h = inner.post_mu.register_forward_hook(hook)
    torch.random.manual_seed(0)
    try:
        with torch.inference_mode(), torch.device("cuda"):
            carry = net.initial_carry(batch)
            for _ in range(sup_steps):
                carry, _, _, _, fin = net(carry=carry, batch=batch,
                                          return_keys=[], use_posterior=True)
                if fin:
                    break
    finally:
        h.remove()
        inner._y_embeddings = o_yemb
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", required=True, help="test 디렉토리")
    ap.add_argument("--task", choices=["nqueens", "sudoku"], required=True)
    ap.add_argument("--num", type=int, default=128)
    ap.add_argument("--sup-steps", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    net, _ = load_model(args.ckpt_dir, args.step)
    inner = net.model.inner
    net.eval()
    inputs = np.load(f"{args.data}/all__inputs.npy")[: args.num].astype(np.int64)
    labels = np.load(f"{args.data}/all__labels.npy")[: args.num].astype(np.int64)
    res = {"checkpoint": os.path.join(args.ckpt_dir, args.step), "task": args.task,
           "num": args.num, "embed_scale_sqrtD": float(np.sqrt(512))}

    B = args.num
    batch = {"inputs": torch.from_numpy(inputs.astype(np.int32)).cuda(),
             "labels": torch.from_numpy(labels.astype(np.int32)).cuda(),
             "puzzle_identifiers": torch.zeros(B, dtype=torch.int32).cuda()}

    print(f"== {res['checkpoint']}  task={args.task}  n={args.num}")
    print("\n-- Task 1/2b: ||y_emb|| vs ||z_H|| at u_y = z_H + y_emb --")
    rows = measure_scales(net, inner, batch, args.sup_steps)
    ntr = 3
    print(f"  {'idx':>3} {'sup':>3} {'tr':>2} | {'||z_H||':>9} {'||y_emb||':>10} {'ratio':>7} "
          f"| {'||z_H||np':>9} {'||y||np':>9} {'ratio_np':>8}")
    for i, r in enumerate(rows[:6] + rows[-2:]):
        j = i if i < 6 else len(rows) - (8 - i)
        print(f"  {j:>3} {j//ntr:>3} {j%ntr:>2} | {r['zH_norm_all']:9.4f} {r['y_norm_all']:10.4f} "
              f"{r['ratio_all']:7.4f} | {r['zH_norm_nonpad']:9.4f} {r['y_norm_nonpad']:9.4f} "
              f"{r['ratio_nonpad']:8.4f}")
    print(f"  (pad positions/token-row = {rows[0]['n_pad_pos']} of {rows[0]['n_pos']})")
    res["scales"] = rows
    res["trans0"] = rows[0]
    res["steady_ratio_nonpad"] = float(np.mean([r["ratio_nonpad"] for r in rows[1:]]))
    print(f"  trans0 ratio_nonpad = {rows[0]['ratio_nonpad']:.4f} | "
          f"steady(idx>=1) mean ratio_nonpad = {res['steady_ratio_nonpad']:.4f}")

    print("\n-- Task 2a: posterior 입력 불변성 --")
    def acc(preds, inp, lab, tag):
        d = {"exact_vs_label": float((preds == lab).all(1).mean()),
             "token_vs_label": float((preds == lab).mean())}
        if args.task == "nqueens":
            from evaluators.nqueens import verify_boards
            r = verify_boards(preds, inp, 8)
            d.update(valid=float(r["valid"].mean()), consistent=float(r["consistent"].mean()))
        print(f"  [{tag:<34}] exact_vs_label={d['exact_vs_label']:.4f} "
              f"token={d['token_vs_label']:.4f}" +
              (f" valid={d['valid']:.4f} consistent={d['consistent']:.4f}" if 'valid' in d else ""))
        return d

    base = rollout(net, inputs, labels, K=1, mode="posterior", verbose=False)["preds"][0]
    res["post_real"] = acc(base, inputs, labels, "posterior / real inputs")

    shuf = inputs[::-1].copy()
    p = rollout(net, shuf, labels, K=1, mode="posterior", verbose=False)["preds"][0]
    res["post_shuffled_inputs"] = acc(p, shuf, labels, "posterior / shuffled inputs")
    res["post_shuffled_same_as_real"] = float((p == base).all(1).mean())
    print(f"     -> 원본 출력과 동일한 보드: {res['post_shuffled_same_as_real']*100:.2f}%")

    lab_alt = np.roll(labels, 1, axis=0)
    p2 = rollout(net, inputs, lab_alt, K=1, mode="posterior", verbose=False)["preds"][0]
    res["post_wrong_label_reproduces_it"] = float((p2 == lab_alt).all(1).mean())
    print(f"  [posterior / WRONG label            ] 그 틀린 라벨을 그대로 재현: "
          f"{res['post_wrong_label_reproduces_it']*100:.2f}%")

    pr = rollout(net, inputs, labels, K=1, mode="prior", verbose=False)["preds"][0]
    res["prior_real"] = acc(pr, inputs, labels, "prior / real inputs (참고)")

    assert inner._y_embeddings.__func__ is type(inner)._y_embeddings, "_y_embeddings 원복 실패"
    assert type(inner.post_mu).__name__ == "SwiGLU", "post_mu 원복 실패"
    print("\n[원복 확인] _y_embeddings / post_mu 원상태.")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
