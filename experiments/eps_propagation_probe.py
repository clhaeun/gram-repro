"""GRAM ε 전파 probe — 추론 전용, 재학습 없음, models/ 무변경.

두 가지를 한 드라이버로 잰다.  둘 다 supervision 축(n=1..halt_max_steps)과
transition 축(t=1..H_cycles)을 함께 분해한다.  기존
experiments/contraction_probe.py 는 z_H 를 얼린 inner L-loop 만 재므로 여기서
재는 것과 다른 양이다.

  --mode sigma   σ(n,t) — prior(추론 경로) 와 posterior(측정 전용) 둘 다.
                 기존 diag/eps_std_*_t{t} 는 n 축을 합쳐 버려서 annealing 을
                 볼 수 없다.  여기서는 n 을 살린다.

  --mode lambda  두 궤적 거리 d_t 의 감쇠율.  잔차가 아니라 궤적 거리를 쓰므로
                 f 의 정의가 필요 없다.
                   궤적 A: 정상 실행
                   궤적 B: transition #1 의 ε 에만 δ 를 더하고 ★이후 난수 동일★
                   d_t = ‖h_t − h'_t‖₂   (position 별 L2 뒤 batch·position 평균)
                 log d_t 의 기울기가 log λ 다.

★ 난수 공유 방식
  reparameterize 를 전용 CUDA Generator 로 갈아끼운다.  두 궤적이 같은 시드로
  같은 호출 순서를 타므로 noise 가 완전히 동일하다.  torch.randn_like(전역 RNG)
  를 그대로 두면 δ 전파와 새 난수를 구분할 수 없다.
  eval 경로에는 다른 RNG 소비자가 없다 (halt_exploration 은 self.training 전용,
  dropout 없음) — gram.py:621, 632 참조.

★ 궤적 포착 방식
  inner.prior_mu 에 forward_pre_hook 을 건다.  그 입력이 곧 ε 을 더하기 직전의
  z_H 다 (gram.py:480, 514).  hook 은 값을 읽기만 하므로 궤적에 영향이 없다.
    h_pre[i]  = prior_mu 입력            (transition i 의 ε 주입 직전 z_H)
    h_post[i] = h_pre[i] + eps_scale·ε_i (다음 transition 으로 넘어가는 z_H)

★ dtype
  δ=0.01·‖ε‖ 은 노름 ~22 의 상태에 대해 1e-2 수준이라 bf16(가수 8bit)에서는
  반올림 잡음에 묻힌다.  기본값을 float32 로 두고, --dtype bfloat16 으로
  학습 시 정밀도를 재현할 수 있게 남겨 둔다.

사용:
  python experiments/eps_propagation_probe.py --mode sigma \
      --ckpt-dir checkpoints/Nqueens-10-ACT-torch/nq10_repro_s0 --step step_138534 \
      --data data/nqueens-10/test --n-puzzles 256 \
      --out results/eps_probe/nq10_sigma.json

  python experiments/eps_propagation_probe.py --mode lambda \
      --ckpt-dir checkpoints/Nqueens-10-ACT-torch/nq10_repro_s0 --step step_138534 \
      --data data/nqueens-10/test --n-puzzles 256 --ratios 0.01,0.1,1.0 \
      --out results/eps_probe/nq10_lambda.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pretrain as P  # noqa: E402
from experiments.gram_harness import _batch  # noqa: E402


# --------------------------------------------------------------------------- #
# loader (gram_harness.load_model + forward_dtype override)
# --------------------------------------------------------------------------- #
def promote_rms_norm_to_float64():
    """측정 전용: gram.py 가 쓰는 rms_norm 을 float64 로 승격한다.

    `models/layers.py:163-169` 의 rms_norm 은 입력 dtype 과 무관하게 내부 계산을
    float32 로 내린다. 그대로 두면 모델을 float64 로 올려도 매 transition 마다
    float32 상대오차(1.2e-07)가 다시 들어와 d_t 의 바닥이 float32 때와 같아진다
    (‖h‖≈32 → 약 4e-06). 바닥이 측정값을 만드는지 가리려면 이 한 곳을 바꿔야 한다.

    gram.py 는 `from models.layers import rms_norm` 으로 이름을 이미 바인딩했으므로
    models.layers 쪽이 아니라 gram 모듈의 이름을 갈아끼운다.
    """
    import models.recursive_reasoning.gram as G

    def rms_norm64(hidden_states, variance_epsilon):
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float64)
        variance = h.square().mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + variance_epsilon)
        return h.to(input_dtype)

    G.rms_norm = rms_norm64
    return G


def load_model(ckpt_dir, step, forward_dtype=None):
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "all_config.yaml"))
    cfg.load_checkpoint = os.path.join(ckpt_dir, step)
    cfg.ema = False                  # checkpoint is already the EMA copy
    cfg.eval_save_outputs = []
    cfg.evaluators = []
    cfg.nodecay_marked = False
    if forward_dtype is not None:
        cfg.arch.forward_dtype = forward_dtype
    config = P.PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))

    _, train_metadata = P.create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1)
    ts = P.init_train_state(config, train_metadata, rank=0, world_size=1)

    net = getattr(ts.model, "_orig_mod", ts.model)   # eager ACTLossHead
    net.eval()
    if forward_dtype == "float32":
        net.float()
    elif forward_dtype == "float64":
        net.double()
    return net, config


# --------------------------------------------------------------------------- #
# deterministic reparameterize + trajectory capture
# --------------------------------------------------------------------------- #
class Recorder:
    """reparameterize 대체 + prior_mu 입력 포착.

    delta 는 호출 인덱스 delta_at 에서 한 번만 더한다.  noise 는 전용 generator
    에서 뽑으므로 두 궤적이 비트 단위로 같은 난수를 쓴다.
    """

    def __init__(self, inner, seed, delta=None, delta_at=0, keep_states=True):
        self.inner = inner
        self.seed = seed
        self.delta = delta
        self.delta_at = delta_at
        self.keep_states = keep_states
        self.gen = torch.Generator(device="cuda")
        # 포착/거리 계산 dtype.  모델이 float64 면 여기도 float64 라야
        # float32 바닥(‖h‖≈32 기준 약 4e-06)이 d_t 를 대신 만들지 않는다.
        acc = next(inner.parameters()).dtype
        self._acc_dtype = acc if acc in (torch.float32, torch.float64) else torch.float32
        self._orig_rep = None
        self._hook = None
        self.reset()

    def reset(self):
        self.gen.manual_seed(self.seed)
        self.idx = 0
        self.h_pre = []      # prior_mu 입력 (ε 주입 직전 z_H)
        self.h_post = []     # ε 주입 직후 z_H
        self.eps = []        # 주입된 ε (δ 포함)
        self.sigma_mean = []      # element mean of σ
        self.sigma_posnorm = []   # per-position L2 of σ, 평균
        self.mu_posnorm = []
        self.eps_posnorm = []

    # -- reparameterize 대체 ------------------------------------------------ #
    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) * self.inner.sigma_scale
        noise = torch.randn(std.shape, generator=self.gen, device=std.device,
                            dtype=torch.float32).to(std.dtype)
        eps = mu + noise * std

        if self.delta is not None and self.idx == self.delta_at:
            eps = eps + self.delta.to(eps.dtype)

        with torch.no_grad():
            s32 = std.detach().to(self._acc_dtype)
            self.sigma_mean.append(float(s32.mean()))
            self.sigma_posnorm.append(float(s32.norm(dim=-1).mean()))
            self.mu_posnorm.append(float(mu.detach().to(self._acc_dtype).norm(dim=-1).mean()))
            e32 = eps.detach().to(self._acc_dtype)
            self.eps_posnorm.append(float(e32.norm(dim=-1).mean()))
            if self.keep_states:
                self.eps.append(e32.clone())
                self.h_post.append(self.h_pre[-1] + float(self.inner.config.eps_scale) * e32)

        self.idx += 1
        return eps

    # -- prior_mu 입력 포착 -------------------------------------------------- #
    def _pre_hook(self, module, args):
        if self.keep_states:
            self.h_pre.append(args[0].detach().to(self._acc_dtype).clone())
        return None

    def __enter__(self):
        self._orig_rep = self.inner.reparameterize
        self.inner.reparameterize = self._reparameterize
        self._hook = self.inner.prior_mu.register_forward_pre_hook(self._pre_hook)
        return self

    def __exit__(self, *exc):
        self.inner.reparameterize = self._orig_rep
        self._hook.remove()
        return False


@torch.inference_mode()
def run_trajectory(net, batch, halt_max_steps, rec):
    """halt_max_steps 번의 supervision step 을 그대로 돌린다.

    eval 모드에서는 halted = is_last_step 뿐이라 (gram.py:616-621) 매 샘플이
    정확히 halt_max_steps 번 돈다 — 조기 halt 가 없어 궤적 길이가 고정된다.
    """
    rec.reset()
    with torch.device("cuda"):
        carry = net.model.initial_carry(batch)
    for _ in range(halt_max_steps):
        carry, _out = net.model(carry=carry, batch=batch)
    return carry


# --------------------------------------------------------------------------- #
# mode: sigma
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def measure_sigma(net, batch, H, N, rec, want_posterior):
    run_trajectory(net, batch, N, rec)
    inner = net.model.inner
    out = {
        "prior": {
            "sigma_mean": rec.sigma_mean,
            "sigma_posnorm": rec.sigma_posnorm,
            "mu_posnorm": rec.mu_posnorm,
            "eps_posnorm": rec.eps_posnorm,
        }
    }
    if want_posterior:
        # 측정 전용 posterior: 포착해 둔 h_pre 로 post head 만 다시 돌린다.
        # 궤적에는 쓰지 않는다 (gram.py:_record_diag_kl 와 같은 규약).
        y_emb = inner._y_embeddings(batch["labels"])
        s_mean, s_norm, m_norm = [], [], []
        for h in rec.h_pre:
            u_y = h.to(inner.forward_dtype) + y_emb
            mu_q = inner.post_mu(u_y)
            logvar_q = inner.post_logvar(u_y) + inner.post_logvar_bias
            std_q = torch.exp(0.5 * logvar_q.float())
            s_mean.append(float(std_q.mean()))
            s_norm.append(float(std_q.norm(dim=-1).mean()))
            m_norm.append(float(mu_q.detach().float().norm(dim=-1).mean()))
        out["posterior"] = {"sigma_mean": s_mean, "sigma_posnorm": s_norm,
                            "mu_posnorm": m_norm}
    return out


# --------------------------------------------------------------------------- #
# mode: lambda
# --------------------------------------------------------------------------- #
def _fit_loglambda(d, lo=0, hi=None):
    """log d_t 의 최소자승 기울기.  d<=0 인 항은 버린다.

    lo/hi 로 창을 자를 수 있다.  d_t 가 단일 지수가 아니면(초반 급감 후 평탄)
    전 구간 fit 의 r2 가 낮게 나오므로 창별로 따로 본다.
    """
    seg = d[lo:hi]
    y = [math.log(v) for v in seg if v > 0]
    if len(y) < 3:
        return None, None, len(y)
    x = list(range(len(y)))
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    slope = sxy / sxx
    ss_res = sum((yi - (my + slope * (xi - mx))) ** 2 for xi, yi in zip(x, y))
    ss_tot = sum((yi - my) ** 2 for yi in y)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, r2, len(y)


EARLY = 6      # 초반 창 (transition 0..5)
TAIL = 12      # 평탄 구간 판정용 꼬리 길이


def _summarise(d):
    """d_t 곡선 하나에 대한 요약값 묶음.  판단이 아니라 수치만 낸다."""
    slope, r2, used = _fit_loglambda(d)
    e_slope, e_r2, e_used = _fit_loglambda(d, 0, EARLY)
    t_slope, t_r2, t_used = _fit_loglambda(d, len(d) - TAIL)
    tail_mean = float(np.mean(d[-TAIL:]))
    return {
        "d": d,
        "d_over_d0": [v / d[0] if d[0] > 0 else None for v in d],
        "d0": d[0],
        "dT": d[-1],
        "d_min": min(d),
        "d_argmin": int(np.argmin(d)),
        "log_lambda_fit": slope,
        "lambda_fit": math.exp(slope) if slope is not None else None,
        "fit_r2": r2,
        "fit_points": used,
        "lambda_fit_early": math.exp(e_slope) if e_slope is not None else None,
        "fit_r2_early": e_r2,
        "early_window": EARLY,
        "lambda_fit_tail": math.exp(t_slope) if t_slope is not None else None,
        "fit_r2_tail": t_r2,
        "tail_window": TAIL,
        "tail_mean": tail_mean,
        "tail_mean_over_d0": tail_mean / d[0] if d[0] > 0 else None,
        "lambda_endpoint": (d[-1] / d[0]) ** (1.0 / (len(d) - 1)) if d[0] > 0 and d[-1] > 0 else None,
    }


@torch.inference_mode()
def measure_lambda(net, batch, N, ratios, seed, dir_seed):
    inner = net.model.inner

    # --- 궤적 A (기준) --------------------------------------------------- #
    recA = Recorder(inner, seed=seed)
    with recA:
        run_trajectory(net, batch, N, recA)
    hA = [h.clone() for h in recA.h_post]
    eps1_norm = recA.eps_posnorm[0]

    # δ 방향: position 별 단위벡터.  ratio 사이에서 방향을 공유해 크기만 바꾼다.
    gdir = torch.Generator(device="cuda"); gdir.manual_seed(dir_seed)
    ref = recA.eps[0]
    u = torch.randn(ref.shape, generator=gdir, device=ref.device, dtype=torch.float32)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    res = {"eps1_posnorm": eps1_norm,
           "h_post_posnorm": [float(h.norm(dim=-1).mean()) for h in hA],
           "ratios": {}}

    for r in ratios:
        delta = u * (r * eps1_norm)
        recB = Recorder(inner, seed=seed, delta=delta, delta_at=0)
        with recB:
            run_trajectory(net, batch, N, recB)
        d = [float((a - b).norm(dim=-1).mean()) for a, b in zip(hA, recB.h_post)]
        res["ratios"][str(r)] = {"delta_posnorm": r * eps1_norm, **_summarise(d)}
        del recB
    return res


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sigma", "lambda"], required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-puzzles", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ratios", default="0.01,0.1,1.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir-seed", type=int, default=12345)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float64"])
    ap.add_argument("--posterior", action="store_true", help="sigma 모드에서 posterior 축도 잰다")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.dtype == "float64":
        promote_rms_norm_to_float64()
    net, config = load_model(args.ckpt_dir, args.step, forward_dtype=args.dtype)
    inner = net.model.inner
    H = int(inner.config.H_cycles)
    N = int(inner.config.halt_max_steps)
    print(f"[cfg] H_cycles={H} halt_max_steps={N} -> transitions={H*N} "
          f"L_cycles={inner.config.L_cycles} eps_scale={inner.config.eps_scale} "
          f"dtype={args.dtype}", flush=True)

    inputs = np.load(f"{args.data}/all__inputs.npy").astype(np.int64)
    labels = np.load(f"{args.data}/all__labels.npy").astype(np.int64)
    n = min(args.n_puzzles, len(inputs))
    ratios = [float(x) for x in args.ratios.split(",")]

    per_batch = []
    for lo in range(0, n, args.batch_size):
        hi = min(lo + args.batch_size, n)
        batch = _batch(inputs, labels, lo, hi, args.batch_size)
        if args.mode == "sigma":
            rec = Recorder(inner, seed=args.seed + lo)
            with rec:
                per_batch.append(measure_sigma(net, batch, H, N, rec, args.posterior))
            del rec
        else:
            per_batch.append(measure_lambda(net, batch, N, ratios,
                                            seed=args.seed + lo, dir_seed=args.dir_seed))
        print(f"  batch {lo}:{hi} done", flush=True)
        torch.cuda.empty_cache()

    # --- batch 평균 --------------------------------------------------------- #
    def mean_curves(key_path):
        acc = None
        for b in per_batch:
            v = b
            for k in key_path:
                v = v[k]
            v = np.asarray(v, dtype=np.float64)
            acc = v if acc is None else acc + v
        return (acc / len(per_batch)).tolist()

    out = {
        "mode": args.mode,
        "ckpt_dir": args.ckpt_dir,
        "step": args.step,
        "data": args.data,
        "n_puzzles": n,
        "batch_size": args.batch_size,
        "n_batches": len(per_batch),
        "dtype": args.dtype,
        "seed": args.seed,
        "H_cycles": H,
        "halt_max_steps": N,
        "n_transitions": H * N,
        "L_cycles": int(inner.config.L_cycles),
        "eps_scale": float(inner.config.eps_scale),
        # index i -> (n, t) = (i//H + 1, i%H + 1)
        "axis_note": "index i of every curve maps to supervision n=i//H_cycles+1, transition t=i%H_cycles+1",
    }

    if args.mode == "sigma":
        out["prior"] = {k: mean_curves(["prior", k]) for k in per_batch[0]["prior"]}
        if args.posterior:
            out["posterior"] = {k: mean_curves(["posterior", k]) for k in per_batch[0]["posterior"]}
    else:
        out["eps1_posnorm"] = float(np.mean([b["eps1_posnorm"] for b in per_batch]))
        out["h_post_posnorm"] = mean_curves(["h_post_posnorm"])
        out["ratios"] = {}
        for r in ratios:
            k = str(r)
            d = mean_curves(["ratios", k, "d"])
            out["ratios"][k] = {
                "delta_posnorm": float(np.mean([b["ratios"][k]["delta_posnorm"] for b in per_batch])),
                **_summarise(d),
                "per_batch_lambda_fit_early": [b["ratios"][k]["lambda_fit_early"] for b in per_batch],
                "per_batch_tail_mean_over_d0": [b["ratios"][k]["tail_mean_over_d0"] for b in per_batch],
            }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
