"""Task 2: "KL~=0 인데 posterior valid=100%" 모순 규명.

세 측정을 한 스크립트에서 한다. 학습 코드는 건드리지 않는다 -- 전부 런타임
monkeypatch/hook 이고, 각 측정이 끝나면 원복한다.

  2a  eps=0 ablation : reparameterize 가 mu 만 반환하도록 강제 (노이즈 제거).
                       posterior valid 가 유지되는지 본다.
  2b  transition별 KL: gram.py:343-345 / 368-369 의 trace_kl 훅으로 supervision
                       step x transition 격자의 KL 을 전부 기록한다.
                       (학습 로그의 diag/kl_raw 는 gram.py:382 가 마지막
                        transition 의 kl 하나만 _latent_diag 에 넘기므로
                        이 격자의 마지막 열에만 해당한다.)
  2c  mu_q 의 y 의존성: 같은 z_H 에 다른 라벨 y' 를 넣어 mu_q 가 바뀌는지.

사용법:
    python experiments/diag_contradiction.py \
        --ckpt-dir checkpoints/Nqueens-8-ACT-torch/gram_nq8_fix_b --step step_17085
"""
import argparse
import json
import os
import sys
from contextlib import contextmanager

import numpy as np
import torch

# experiments/diag/ 로 한 단계 내려왔으므로 repo root 는 부모의 부모의 부모다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.gram_harness import load_model, rollout          # noqa: E402
from evaluators.nqueens import verify_boards                       # noqa: E402


def summarize(tag, preds, inputs, n):
    r = verify_boards(preds, inputs, n)
    qc = r["queen_count"]
    out = dict(valid=float(r["valid"].mean()), consistent=float(r["consistent"].mean()),
               strict=float(r["strict"].mean()), queen_count_mean=float(qc.mean()),
               queen_count_min=int(qc.min()), queen_count_max=int(qc.max()))
    print(f"  [{tag:<28}] valid={out['valid']:.4f} consistent={out['consistent']:.4f} "
          f"strict={out['strict']:.4f} queens mean={out['queen_count_mean']:.2f} "
          f"(min={out['queen_count_min']} max={out['queen_count_max']})")
    return out


@contextmanager
def deterministic_latent(inner):
    """reparameterize -> mu (노이즈 항 제거). gram.py:247-250 의 임시 대체."""
    orig = inner.reparameterize
    inner.reparameterize = lambda mu, logvar: mu
    try:
        yield
    finally:
        inner.reparameterize = orig


# --------------------------------------------------------------------------- #
def task2a(net, inner, inputs, labels, n, res):
    print("\n== 2a. eps=0 ablation (reparameterize -> mu) ==")
    res["2a"] = {}
    for mode in ("posterior", "prior"):
        base = rollout(net, inputs, labels, K=1, mode=mode, verbose=False)["preds"][0]
        res["2a"][f"{mode}_baseline"] = summarize(f"{mode} baseline", base, inputs, n)
        with deterministic_latent(inner):
            abl = rollout(net, inputs, labels, K=1, mode=mode, verbose=False)["preds"][0]
        res["2a"][f"{mode}_eps0"] = summarize(f"{mode} eps=0", abl, inputs, n)
    # 원복 확인
    chk = rollout(net, inputs, labels, K=1, mode="posterior", verbose=False)["preds"][0]
    res["2a"]["posterior_restored"] = summarize("posterior restored", chk, inputs, n)


# --------------------------------------------------------------------------- #
def task2b(net, inner, batch, sup_steps, res):
    print("\n== 2b. transition별 KL (posterior 경로, 전체 ACT rollout) ==")
    inner.trace_kl = True
    grid = []
    torch.random.manual_seed(0)
    try:
        with torch.inference_mode(), torch.device("cuda"):
            carry = net.initial_carry(batch)
            for s in range(sup_steps):
                carry, _, _, _, all_finish = net(carry=carry, batch=batch,
                                                 return_keys=[], use_posterior=True)
                grid.append(list(inner._kl_trace))
                if all_finish:
                    break
    finally:
        inner.trace_kl = False

    ncol = max(len(r) for r in grid)
    head = "  sup_step | " + " ".join(f"trans{j}".rjust(12) for j in range(ncol))
    print(head + "      <- 마지막 열만 학습 KL 페널티 대상 (Eq 14)")
    print("  " + "-" * (len(head) - 2))
    for i, row in enumerate(grid):
        print(f"  {i:>8} | " + " ".join(f"{v:12.4f}" for v in row))
    cols = [[r[j] for r in grid if j < len(r)] for j in range(ncol)]
    print("  " + "-" * (len(head) - 2))
    print("  mean     | " + " ".join(f"{np.mean(c):12.4f}" for c in cols))
    res["2b"] = {"grid": grid,
                 "col_mean": [float(np.mean(c)) for c in cols],
                 "n_transitions_per_step": ncol,
                 "note": "마지막 열 == 학습 로그의 diag/kl_raw (gram.py:382)"}


# --------------------------------------------------------------------------- #
def task2c(net, inner, batch, sup_steps, res):
    print("\n== 2c. mu_q 의 y 의존성 (같은 z_H, 다른 라벨) ==")
    # y_emb 는 gram.py:311 이 labels 로부터 만든다. 다른 라벨은 배치를 한 칸 굴려서.
    y_true = inner._y_embeddings(batch["labels"])
    y_alt = inner._y_embeddings(torch.roll(batch["labels"], 1, dims=0))

    caught = []
    busy = {"v": False}

    def hook(mod, args, out):
        if busy["v"]:                      # hook 안에서 mod 를 다시 부르므로 재진입 차단
            return
        busy["v"] = True
        try:
            _probe(mod, args, out)
        finally:
            busy["v"] = False

    def _probe(mod, args, out):
        u_y = args[0]                      # gram.py:337/363 의 u_y = z_H + y_emb
        z_H = u_y - y_true                 # y_emb 를 빼서 z_H 복원
        mu_q_true = out
        mu_q_alt = mod(z_H + y_alt)
        mu_p = inner.prior_mu(z_H)
        f = lambda t: t.float()
        d_y = (f(mu_q_true) - f(mu_q_alt)).norm(dim=-1)
        d_p = (f(mu_q_true) - f(mu_p)).norm(dim=-1)
        nrm = f(mu_q_true).norm(dim=-1)
        caught.append(dict(
            rel_change_from_y=float((d_y / nrm.clamp_min(1e-9)).mean()),
            rel_gap_q_vs_p=float((d_p / nrm.clamp_min(1e-9)).mean()),
            mu_q_norm=float(nrm.mean()),
            zH_norm=float(f(z_H).norm(dim=-1).mean()),
            y_emb_norm=float(f(y_true).norm(dim=-1).mean()),
        ))

    h = inner.post_mu.register_forward_hook(hook)
    torch.random.manual_seed(0)
    try:
        with torch.inference_mode(), torch.device("cuda"):
            carry = net.initial_carry(batch)
            for _ in range(sup_steps):
                carry, _, _, _, all_finish = net(carry=carry, batch=batch,
                                                 return_keys=[], use_posterior=True)
                if all_finish:
                    break
    finally:
        h.remove()

    ncol = res.get("2b", {}).get("n_transitions_per_step") or 3
    print(f"  rel_change_from_y = ||mu_q(y) - mu_q(y')|| / ||mu_q(y)||   (y 를 바꿨을 때 mu_q 변화)")
    print(f"  rel_gap_q_vs_p    = ||mu_q      - mu_p   || / ||mu_q||     (posterior vs prior 평균 차이)")
    print(f"\n  {'idx':>4} {'sup':>4} {'trans':>6} {'rel_change_from_y':>18} {'rel_gap_q_vs_p':>15} "
          f"{'||mu_q||':>10} {'||z_H||':>10} {'||y_emb||':>10}")
    for i, c in enumerate(caught):
        print(f"  {i:>4} {i // ncol:>4} {i % ncol:>6} {c['rel_change_from_y']:>18.5f} "
              f"{c['rel_gap_q_vs_p']:>15.5f} {c['mu_q_norm']:>10.4f} {c['zH_norm']:>10.4f} "
              f"{c['y_emb_norm']:>10.4f}")
    res["2c"] = {"per_transition": caught, "n_transitions_per_step": ncol}


# --------------------------------------------------------------------------- #
@contextmanager
def y_gate(inner, allow):
    """transition 단위로 posterior(y) 사용 여부를 켜고 끈다.

    allow 에 없는 transition 에서는 q 를 p 로 **정확히** 치환한다:
        mu_q     := prior_mu(z_H)
        logvar_q := prior_logvar(z_H) + prior_logvar_bias - post_logvar_bias
    (gram.py:339 가 호출 결과에 post_logvar_bias 를 더하므로 미리 빼둔다.
     gram.py:334 의 prior 쪽과 정확히 같은 값이 되게 하기 위함.)
    z_H 는 u_y - y_emb 로 복원한다 (gram.py:337/363).
    """
    st = {"i": 0, "y": None}
    o_yemb, o_mu, o_lv = inner._y_embeddings, inner.post_mu, inner.post_logvar

    def yemb(labels):
        st["y"] = o_yemb(labels)
        return st["y"]

    def mu(u_y):
        return o_mu(u_y) if st["i"] in allow else inner.prior_mu(u_y - st["y"])

    def lv(u_y):
        i = st["i"]; st["i"] += 1
        if i in allow:
            return o_lv(u_y)
        return (inner.prior_logvar(u_y - st["y"])
                + inner.prior_logvar_bias - inner.post_logvar_bias)

    # post_mu/post_logvar 는 등록된 submodule 이라 함수를 그대로 대입할 수 없다.
    # nn.Module 로 감싸서 자리만 바꿔 끼우고, 끝나면 원래 모듈을 되돌린다.
    class _Fn(torch.nn.Module):
        def __init__(self, f):
            super().__init__()
            self.f = f

        def forward(self, x):
            return self.f(x)

    inner._y_embeddings = yemb
    inner.post_mu, inner.post_logvar = _Fn(mu), _Fn(lv)
    try:
        yield st
    finally:
        inner._y_embeddings = o_yemb
        inner.post_mu, inner.post_logvar = o_mu, o_lv


def task2d(net, inner, inputs, labels, n, res, ntrans=3):
    print("\n== 2d. transition 단위 y 게이팅 (q:=p 로 치환) ==")
    print("   allow = y(posterior) 를 실제로 쓰는 global transition 인덱스 집합")
    all_idx = set(range(16 * ntrans))
    cases = [
        ("allow=ALL (baseline)",      all_idx),
        ("allow={} (y 전면 차단)",     set()),
        ("allow={0} (첫 transition만)", {0}),
        ("allow=ALL-{0} (첫것만 차단)", all_idx - {0}),
        ("allow=sup0 전체 {0,1,2}",    {0, 1, 2}),
    ]
    res["2d"] = {}
    for tag, allow in cases:
        with y_gate(inner, allow):
            p = rollout(net, inputs, labels, K=1, mode="posterior", verbose=False)["preds"][0]
        res["2d"][tag] = summarize(tag, p, inputs, n)
    chk = rollout(net, inputs, labels, K=1, mode="posterior", verbose=False)["preds"][0]
    res["2d"]["restored"] = summarize("restored", chk, inputs, n)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", default="data/nqueens-8/test")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--num", type=int, default=256)
    ap.add_argument("--kl-batch", type=int, default=64)
    ap.add_argument("--sup-steps", type=int, default=16)
    ap.add_argument("--skip", default="", help="쉼표구분: 2a,2b,2c")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    net, _ = load_model(args.ckpt_dir, args.step)
    inner = net.model.inner
    net.eval()

    inputs = np.load(f"{args.data}/all__inputs.npy")[: args.num]
    labels = np.load(f"{args.data}/all__labels.npy")[: args.num]
    res = {"checkpoint": os.path.join(args.ckpt_dir, args.step), "num_puzzles": args.num}
    print(f"checkpoint={res['checkpoint']}  num={args.num}  kl_batch={args.kl_batch}")

    B = args.kl_batch
    batch = {
        "inputs": torch.from_numpy(inputs[:B].astype(np.int32)).cuda(),
        "labels": torch.from_numpy(labels[:B].astype(np.int32)).cuda(),
        "puzzle_identifiers": torch.zeros(B, dtype=torch.int32).cuda(),
    }

    if "2a" not in skip:
        task2a(net, inner, inputs, labels, args.n, res)
    if "2b" not in skip:
        task2b(net, inner, batch, args.sup_steps, res)
    if "2c" not in skip:
        task2c(net, inner, batch, args.sup_steps, res)
    if "2d" not in skip:
        task2d(net, inner, inputs, labels, args.n, res,
               ntrans=res.get("2b", {}).get("n_transitions_per_step", 3))

    # 임시 변경이 남아있지 않은지 최종 확인
    assert inner.reparameterize.__func__ is type(inner).reparameterize, "reparameterize 원복 실패"
    assert getattr(inner, "trace_kl", False) is False, "trace_kl 원복 실패"
    assert type(inner.post_mu).__name__ == "SwiGLU", f"post_mu 원복 실패: {type(inner.post_mu)}"
    assert type(inner.post_logvar).__name__ == "SwiGLU", "post_logvar 원복 실패"
    assert inner._y_embeddings.__func__ is type(inner)._y_embeddings, "_y_embeddings 원복 실패"
    print("\n[원복 확인] reparameterize / trace_kl / post_mu / post_logvar / _y_embeddings 전부 원상태.")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
