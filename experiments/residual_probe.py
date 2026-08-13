"""[50] residual probe + PCA 궤적 (GRAM).  추론 전용, models/ 무수정.

★ 한 셀 = forward 1회.  잔차 통계(앞 256행)와 궤적(같은 퍼즐 N복제)을 같은 패스에서 뽑는다.

u_t 포착
--------
gram.py 의 각 transition 은

    z_H = self.L_level(z_H, z_L, **seq_info)   # <- u_t  (f_H 출력, eps 주입 전)
    mu_p = self.prior_mu(z_H)                  # <- 이 호출의 입력이 곧 u_t
    ...
    z_H = z_H + self.config.eps_scale * eps    # <- h_t

no-grad 루프(H_cycles-1 회, gram.py:468-500)와 마지막 grad transition(gram.py:508-550)이
같은 패턴이라 `inner.prior_mu` 의 forward-pre-hook 하나로 3/3 transition 을 다 잡는다.
eps 는 `inner.reparameterize` 를 갈아끼워 잡는다 (experiments/eps_propagation_probe.py
Recorder 선례).  둘 다 컨텍스트 진입/이탈 시 붙였다 떼므로 학습 경로는 건드리지 않는다.

h_t 는 u_t + eps_scale*eps_t 로 재구성한다.  모델이 하는 연산과 같은 dtype·같은 순서라
비트 단위로 같아야 하며, 마지막 transition 을 carry.inner_carry.z_H 와 대조해 assert 한다.

컬럼 (results/figdata/residual_{key}.csv)
-----------------------------------------
    t, a, d, n, r, cos_eps_du, is_sup_boundary
      a = ||h_t||                            activation norm
      d = ||u_t - h_(t-1)|| / ||h_(t-1)||    결정론적 고정점 잔차  ★a 와 다른 것을 잰다
      n = ||eps_scale*eps_t|| / ||h_(t-1)||
      r = ||h_t - h_(t-1)|| / ||h_(t-1)||
      cos_eps_du = cos(eps_scale*eps_t, u_t - h_(t-1))   토큰별 cos -> 배치·토큰 평균
    전부 토큰별(dim=-1) L2 를 먼저 잡고 비율을 만든 뒤 배치·토큰 평균 (float64 누산).
    본 파일은 전 위치(puzzle_emb 접두 포함), token_content/ 하위 사본은 접두 제외.
    h_0 = H_init (reset_carry 초기값).  is_sup_boundary = supervision step 의 마지막 transition.

dtype
-----
forward_dtype=float32 고정.  bf16 은 상대오차 ~3e-3 이라 d 의 바닥을 반올림이 만든다
(experiments/eps_propagation_probe.py, experiments/contraction_probe.py 와 같은 판단).

사용:
  python experiments/residual_probe.py --key nq10_sig1 \
      --ckpt-dir checkpoints/Nqueens-10-ACT-torch/nq10_repro_s0 --step step_138534 \
      --data data/nqueens-10/test --sigma 1.0 --n-traj 50
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eps_propagation_probe import load_model  # noqa: E402
from models.losses import IGNORE_LABEL_ID, softmax_cross_entropy, stablemax_cross_entropy  # noqa: E402

OUT_DIR = "results/figdata"


# --------------------------------------------------------------------------- #
# recorder
# --------------------------------------------------------------------------- #
class ResidualRecorder:
    """prior_mu 입력(u_t) + reparameterize 출력(eps_t) 을 잡아 transition 통계를 만든다.

    앞 n_probe 행 -> a/d/n/r 통계.  뒤 행 전부 -> 궤적 h 와 decoder CE loss.
    상태를 통째로 쌓지 않고 transition 마다 즉시 축약한다 (256+50 행 x 48 transition).
    """

    def __init__(self, inner, seed, n_probe, labels_traj, keep_traj=True):
        self.inner = inner
        self.seed = seed
        self.n_probe = n_probe
        self.labels_traj = labels_traj          # (n_traj, seq_len) -100 = ignore
        self.keep_traj = keep_traj
        self.eps_scale = float(getattr(inner.config, "eps_scale", 1.0) or 1.0)
        self.pel = inner.puzzle_emb_len
        self.gen = torch.Generator(device="cuda")
        self._u = None
        self.h_prev = None                      # (B, S, D) float32
        self.reset()

    def reset(self):
        self.gen.manual_seed(self.seed)
        self.rows_all = []          # 전 위치 통계
        self.rows_tok = []          # puzzle_emb 접두 제외 통계
        self.h_traj = []            # [(n_traj, S*D) float16]
        self.h_mean = []            # [(n_traj, D) float32]  토큰평균(구 규약 호환)
        self.loss = []              # [(n_traj,)] softmax CE
        self.loss_sm = []           # [(n_traj,)] stablemax CE (학습 손실과 같은 정의)
        self.h_last = None

    # -- 훅 ------------------------------------------------------------------ #
    def _pre_hook(self, module, args):
        self._u = args[0]
        return None

    def _reparameterize(self, mu, logvar):
        # gram.py:326-330 과 같은 식.  난수만 전용 generator 에서 뽑는다.
        std = torch.exp(0.5 * logvar) * self.inner.sigma_scale
        noise = torch.randn(std.shape, generator=self.gen, device=std.device,
                            dtype=torch.float32).to(std.dtype)
        eps = mu + noise * std
        self._record(self._u, eps)
        return eps

    def __enter__(self):
        self._orig_rep = self.inner.reparameterize
        self.inner.reparameterize = self._reparameterize
        self._hook = self.inner.prior_mu.register_forward_pre_hook(self._pre_hook)
        return self

    def __exit__(self, *exc):
        self.inner.reparameterize = self._orig_rep
        self._hook.remove()
        return False

    # -- transition 별 축약 --------------------------------------------------- #
    @staticmethod
    def _stats(u, eps_term, h, hp, sl_tok):
        """토큰별 L2 -> 비율 -> 배치·토큰 평균 (float64).

        cos_eps_du 는 ★크기에서 역산하지 않고 직접 잰 각도★ 다.
        h_t = u_t + eps_scale*eps_t 이므로 h_t - h_(t-1) = (u_t - h_(t-1)) + eps_term 이고
            r^2 = d^2 + n^2 + 2*d*n*cos
        가 된다.  sigma=0 에서 r 이 d, n 보다 작게 나오는 것이 실제 상쇄(cos<0)인지
        확인하려면 이 값을 봐야 한다.  cos 은 토큰별로 구해 배치·토큰 평균한다.
        """
        u, eps_term, h, hp = (x[:, sl_tok] for x in (u, eps_term, h, hp))
        nh_prev = hp.norm(dim=-1)
        a = h.norm(dim=-1)
        du = u - hp
        d = du.norm(dim=-1) / nh_prev
        n = eps_term.norm(dim=-1) / nh_prev
        r = (h - hp).norm(dim=-1) / nh_prev
        cos = F.cosine_similarity(eps_term, du, dim=-1, eps=1e-12)
        return dict(a=float(a.double().mean()), d=float(d.double().mean()),
                    n=float(n.double().mean()), r=float(r.double().mean()),
                    cos_eps_du=float(cos.double().mean()))

    def _record(self, u, eps):
        with torch.no_grad():
            u = u.detach().float()
            eps_term = self.eps_scale * eps.detach().float()
            h = u + eps_term
            hp = self.h_prev
            p = slice(0, self.n_probe)
            self.rows_all.append(self._stats(u[p], eps_term[p], h[p], hp[p], slice(None)))
            self.rows_tok.append(self._stats(u[p], eps_term[p], h[p], hp[p], slice(self.pel, None)))

            if self.keep_traj:
                ht = h[self.n_probe:]
                # losses.py 의 CE 가 .view() 를 쓰므로 슬라이스를 연속으로 만든다.
                logits = self.inner.lm_head(ht)[:, self.pel:].contiguous()
                lab = self.labels_traj
                mask = lab != IGNORE_LABEL_ID
                div = mask.sum(-1).clamp_min(1)
                ce = softmax_cross_entropy(logits, lab, ignore_index=IGNORE_LABEL_ID)
                ce_sm = stablemax_cross_entropy(logits, lab, ignore_index=IGNORE_LABEL_ID,
                                                valid_mask=mask)
                self.loss.append((torch.where(mask, ce, 0).sum(-1) / div).cpu().numpy())
                self.loss_sm.append((torch.where(mask, ce_sm.float(), 0).sum(-1) / div).cpu().numpy())
                self.h_traj.append(ht.reshape(ht.shape[0], -1).half().cpu().numpy())
                self.h_mean.append(ht[:, self.pel:].mean(dim=1).cpu().numpy())

            self.h_prev = h
            self.h_last = h


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def run_cell(args):
    net, config = load_model(args.ckpt_dir, args.step, forward_dtype="float32")
    inner = net.model.inner
    cfg = inner.config
    H, S = int(cfg.H_cycles), int(cfg.halt_max_steps)
    T = H * S

    inputs = np.load(os.path.join(args.data, "all__inputs.npy"))
    labels = np.load(os.path.join(args.data, "all__labels.npy"))
    pid = np.load(os.path.join(args.data, "all__puzzle_identifiers.npy"))
    n_probe = min(args.n_probe, inputs.shape[0])
    i = args.traj_index

    inp = np.concatenate([inputs[:n_probe], np.repeat(inputs[i:i + 1], args.n_traj, 0)])
    lab = np.concatenate([labels[:n_probe], np.repeat(labels[i:i + 1], args.n_traj, 0)])
    pids = np.concatenate([pid[:n_probe], np.repeat(pid[i:i + 1], args.n_traj, 0)])
    batch = {
        "inputs": torch.from_numpy(inp.astype(np.int64)).cuda(),
        "labels": torch.from_numpy(lab.astype(np.int64)).cuda(),
        "puzzle_identifiers": torch.from_numpy(pids.astype(np.int64)).cuda(),
    }
    # puzzle_dataset.py:152-153 과 같은 매핑.  harness 의 _batch 는 이 변환을 안 한다.
    ignore_src = 0                                   # dataset.json: ignore_label_id = 0
    batch["labels"] = torch.where(batch["labels"] == ignore_src, IGNORE_LABEL_ID, batch["labels"])

    B = batch["inputs"].shape[0]
    seq_len = batch["inputs"].shape[1]
    D = int(cfg.hidden_size)
    pel = inner.puzzle_emb_len

    rec = ResidualRecorder(inner, seed=args.seed, n_probe=n_probe,
                           labels_traj=batch["labels"][n_probe:], keep_traj=True)
    # h_0 = reset_carry 의 초기값 (gram.py:430-434)
    rec.h_prev = torch.zeros(B, seq_len + pel, D, device="cuda", dtype=torch.float32) + inner.H_init.float()

    prev_sigma = inner.sigma_scale
    inner.sigma_scale = float(args.sigma)
    try:
        with rec:
            with torch.device("cuda"):
                carry = net.model.initial_carry(batch)
            for _ in range(S):
                carry, _out = net.model(carry=carry, batch=batch)
    finally:
        inner.sigma_scale = prev_sigma

    assert len(rec.rows_all) == T, f"transition 수 {len(rec.rows_all)} != {T}"
    # 재구성 검증: 마지막 transition 의 h 가 실제 carry 상태와 같아야 한다.
    z_ref = carry.inner_carry.z_H.float()
    max_dev = float((rec.h_last - z_ref).abs().max())
    assert max_dev == 0.0, f"h 재구성 불일치: max|Δ| = {max_dev}"

    # ---- 산출 A: residual CSV ---------------------------------------------- #
    os.makedirs(os.path.join(OUT_DIR, "token_content"), exist_ok=True)
    paths = {}
    for name, rows in (("", rec.rows_all), ("token_content", rec.rows_tok)):
        p = os.path.join(OUT_DIR, name, f"residual_{args.key}.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "a", "d", "n", "r", "cos_eps_du", "is_sup_boundary"])
            for t, row in enumerate(rows, start=1):
                w.writerow([t, f"{row['a']:.8g}", f"{row['d']:.8g}",
                            f"{row['n']:.8g}", f"{row['r']:.8g}",
                            f"{row['cos_eps_du']:.8g}", int(t % H == 0)])
        paths["csv" if name == "" else "csv_token_content"] = p

    # ---- 산출 B: 궤적 npz --------------------------------------------------- #
    h = np.stack(rec.h_traj, axis=1)                       # (N, T, S*D) float16
    loss = np.stack(rec.loss, axis=1).astype(np.float32)   # (N, T)
    loss_sm = np.stack(rec.loss_sm, axis=1).astype(np.float32)
    h_mean = np.stack(rec.h_mean, axis=1).astype(np.float32)
    traj_path = os.path.join(OUT_DIR, f"traj_{args.key}.npz")
    np.savez(traj_path, h=h, loss=loss, loss_stablemax=loss_sm, h_mean=h_mean,
             t_index=np.arange(1, T + 1, dtype=np.int32))
    paths["npz"] = traj_path

    meta = {
        "key": args.key,
        "model": "GRAM",
        "ckpt_dir": args.ckpt_dir,
        "step": args.step,
        "weight": "EMA (bare step file; .live 가 live weight)",
        "data": args.data,
        "n_probe": n_probe,
        "n_rows_in_split": int(inputs.shape[0]),
        "traj_index": i,
        "n_traj": args.n_traj,
        "sigma_scale": float(args.sigma),
        "mu_scale": 1.0,
        "path": "prior (use_posterior=False, 라벨 미사용)",
        "seed": args.seed,
        "forward_dtype": "float32",
        "T": T,
        "T_source": f"H_cycles x halt_max_steps = {H} x {S}",
        "D": D,
        "D_hidden": D,
        "seq_len": int(seq_len),
        "puzzle_emb_len": int(pel),
        "h_shape": list(h.shape),
        "h_layout": f"(N, T, seq_all*D) = ({h.shape[0]}, {T}, {seq_len + pel}*{D}) float16, "
                    f"전 위치 flatten (puzzle_emb 접두 {pel} 포함)",
        "h_mean_layout": f"(N, T, {D}) float32, puzzle_emb 접두 제외 토큰평균 (구 traj_dump 규약)",
        "loss": "softmax CE, lm_head(h_t)[:, pel:] vs labels, 유효토큰 평균 (per-sample)",
        "loss_stablemax": "학습 손실과 같은 stablemax CE, 같은 정규화",
        "csv_columns": "t,a,d,n,r,cos_eps_du,is_sup_boundary",
        "csv_reduction": "토큰별 L2 -> 비율 -> 배치·토큰 평균 (float64 누산)",
        "csv_token_set": "본 파일=전 위치, token_content/ 사본=puzzle_emb 접두 제외",
        "h_prev_t1": "H_init (reset_carry 초기값)",
        "is_sup_boundary": f"t %% {H} == 0 (supervision step 의 마지막 transition)",
        "u_capture": "inner.prior_mu forward-pre-hook 의 입력 = f_H 출력 직후·eps 주입 전 z_H",
        "reconstruction_check": f"마지막 transition h 와 carry.z_H 의 max|Δ| = {max_dev}",
        "arch": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in {
            "H_cycles": H, "L_cycles": int(cfg.L_cycles), "halt_max_steps": S,
            "hidden_size": D, "puzzle_emb_len": int(pel),
            "eps_scale": float(getattr(cfg, "eps_scale", 1.0) or 1.0)}.items()},
    }
    for p in (paths["csv"], traj_path):
        with open(p.rsplit(".", 1)[0] + ".meta.json", "w") as f:
            json.dump({**meta, "artifacts": paths}, f, ensure_ascii=False, indent=2)

    d_col = [r["d"] for r in rec.rows_all]
    a_col = [r["a"] for r in rec.rows_all]
    print(f"[{args.key}] T={T} a=[{min(a_col):.4f},{max(a_col):.4f}] "
          f"d[1:10]={' '.join(f'{x:.4f}' for x in d_col[:10])} "
          f"d[-5:]={' '.join(f'{x:.4f}' for x in d_col[-5:])}", flush=True)
    print(f"[out] {paths['csv']}  {traj_path} ({os.path.getsize(traj_path):,} B)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--n-probe", type=int, default=256)
    ap.add_argument("--n-traj", type=int, default=50)
    ap.add_argument("--traj-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    run_cell(ap.parse_args())


if __name__ == "__main__":
    main()
