"""수축성 진단 probe -- 추론 전용, 재학습 없음.

이번 이식의 핵심 질문은 성능이 아니라
  ★"pre-norm + residual scaling 이 GRAM 재귀를 실제로 수축적으로 만들었는가"★ 다.
수축이면 FPRM Theorem 2 의 전제가 충족되고, GRAM 이 TRM 에서 물려받은 1-step
gradient 근사(IFT)가 사후 정당화된다 -- TRM 논문 §3.1 이 HRM 을 비판한 그 지점.

측정 A (contraction factor)
    같은 입력·★같은 z_H 고정★ 아래 서로 다른 두 z_L 로 출발시켜 inner loop 를
    돌리며 거리가 줄어드는지 잰다.  z_H 가 바뀌면 map 이 바뀌어 수축 개념 자체가
    성립하지 않으므로 z_H 고정이 필수다.  transition 별로 따로 잰다.
      factor = (d_K / d_0) ** (1/K)     <1 수축 / ~1 중립 / >1 발산

측정 B (불변 측도)
    같은 입력에 N 개 궤적을 돌리고 transition 후 z_H 의 샘플 간 분산 추이를 본다.
    var 수렴 -> 불변 측도 존재 (수축 + 유계 노이즈 -> 유일 불변 측도)
    var 단조 증가 -> ε 무계수 주입이 실전 문제라는 증거
    ★ H_cycles 를 config 값(보통 3) 대신 --extrap-T(기본 12) 로 외삽한다.
      3 step 으로는 수렴 판정이 불가능하다.  결과는 ★학습 분포 밖 외삽 측정★ 이다.

★ experiments/sweep_harness.py 는 건드리지 않는다.  load_model/_batch 를 재사용만 한다.

사용:
  python experiments/contraction_probe.py --ckpt-dir <dir> --step step_N.live \
      --data data/nqueens-10/test --out reports/contraction/<run>_<step>_probe.json
"""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.gram_harness import load_model, _batch   # noqa: E402


def _seq_info(inner):
    return dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)


def _init_carry(inner, batch_size):
    """empty_carry + 전건 reset = 학습 시 첫 transition 의 z_H/z_L 초기값."""
    # empty_carry 는 device 를 지정하지 않는다 -- pretrain.py 처럼 device 컨텍스트가 필요하다.
    with torch.device("cuda"):
        carry = inner.empty_carry(batch_size)
    flag = torch.ones(batch_size, dtype=torch.bool, device="cuda")
    carry = inner.reset_carry(flag, carry)
    return carry.z_H, carry.z_L


def _dist(a, b):
    """토큰별 L2 거리의 배치·시퀀스 평균 (fp32 로 올려서 잰다)."""
    return (a.float() - b.float()).norm(dim=-1).mean().item()


def _sample_eps(inner, z_H):
    """prior 경로 ε (라벨을 쓰지 않는다 -- 추론 전용 진단)."""
    mu_p = inner.prior_mu(z_H)
    logvar_p = inner.prior_logvar(z_H) + inner.prior_logvar_bias
    return inner.reparameterize(mu_p, logvar_p)


def _eps_scale(inner):
    """eps_scale 미도입 checkpoint(A0 등) 도 읽을 수 있게 기본 1.0 으로 떨어뜨린다."""
    return float(getattr(inner.config, "eps_scale", 1.0))


def _inner_loop(inner, z_L, inj, seq_info):
    """gram.py 의 z_L inner loop 1회분 (damping_eta 는 checkpoint 값을 그대로 쓴다)."""
    eta = float(getattr(inner.config, "damping_eta", 1.0))
    z_tilde = inner.L_level(z_L, inj, **seq_info)
    if eta != 1.0:
        return eta * z_tilde + (1.0 - eta) * z_L
    return z_tilde


@torch.inference_mode()
def measure_A(net, batch, delta_std, seed):
    """transition 별 contraction factor.  z_H 와 input injection 을 고정한다."""
    inner = net.model.inner
    cfg = inner.config
    seq_info = _seq_info(inner)
    x = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    B = batch["inputs"].shape[0]
    z_H, z_L = _init_carry(inner, B)

    out = {}
    torch.random.manual_seed(seed)
    for t in range(1, cfg.H_cycles + 1):
        inj = z_H + x                      # ★ 이 transition 동안 고정 ★
        a = z_L.clone()
        b = z_L + torch.randn_like(z_L.float()).to(z_L.dtype) * delta_std
        d = [_dist(a, b)]
        for _ in range(cfg.L_cycles):
            a = _inner_loop(inner, a, inj, seq_info)
            b = _inner_loop(inner, b, inj, seq_info)
            d.append(_dist(a, b))
        K = cfg.L_cycles
        factor = (d[-1] / d[0]) ** (1.0 / K) if d[0] > 0 and d[-1] > 0 else float("nan")
        out[f"t{t}"] = dict(
            d0=d[0], dK=d[-1], contraction_factor=factor,
            ratio_curve=[d[i + 1] / d[i] if d[i] > 0 else float("nan") for i in range(K)],
            dist_curve=d,
        )
        # --- 실제 상태 전진 (측정과 무관하게 정규 경로를 그대로 탄다) ---
        for _ in range(cfg.L_cycles):
            z_L = _inner_loop(inner, z_L, z_H + x, seq_info)
        z_H = inner.L_level(z_H, z_L, **seq_info)
        z_H = z_H + _eps_scale(inner) * _sample_eps(inner, z_H)
    return out


@torch.inference_mode()
def measure_B(net, batch, n_samples, extrap_T, seed):
    """N 궤적의 transition 별 z_H 샘플간 분산.  Welford 대신 sum/sumsq 누적."""
    inner = net.model.inner
    cfg = inner.config
    seq_info = _seq_info(inner)
    x = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    B = batch["inputs"].shape[0]

    s = [None] * extrap_T
    ss = [None] * extrap_T
    for k in range(n_samples):
        torch.random.manual_seed(seed * 100003 + k)
        z_H, z_L = _init_carry(inner, B)
        for t in range(extrap_T):
            for _ in range(cfg.L_cycles):
                z_L = _inner_loop(inner, z_L, z_H + x, seq_info)
            z_H = inner.L_level(z_H, z_L, **seq_info)
            z_H = z_H + _eps_scale(inner) * _sample_eps(inner, z_H)
            f = z_H.float()
            s[t] = f.clone() if s[t] is None else s[t] + f
            ss[t] = f * f if ss[t] is None else ss[t] + f * f

    var_curve, norm_curve = [], []
    for t in range(extrap_T):
        mean = s[t] / n_samples
        var = ss[t] / n_samples - mean * mean          # 샘플간 분산 (모분산)
        var_curve.append(float(var.clamp_min(0).mean().item()))
        norm_curve.append(float(mean.norm(dim=-1).mean().item()))
    return dict(var_curve=var_curve, zh_mean_norm_curve=norm_curve,
                n_samples=n_samples, extrap_T=extrap_T,
                config_H_cycles=cfg.H_cycles,
                note="★ 학습 분포 밖 외삽 측정 -- config H_cycles 를 넘겨 돌린 구간이 포함된다")


def verdict(factors):
    """3분류만. 성능과 엮지 않는다."""
    vals = [v for v in factors if np.isfinite(v)]
    if not vals:
        return "측정불가"
    m = float(np.mean(vals))
    if m < 0.95:
        return "수축"
    if m > 1.05:
        return "발산"
    return "중립"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", default="data/nqueens-10/test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta-std", type=float, default=0.01)
    ap.add_argument("--delta-mult", type=float, default=10.0,
                    help="선형영역 확인용 2차 delta 배율 ((b) 검증)")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--extrap-T", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    net, _ = load_model(args.ckpt_dir, args.step)
    inner = net.model.inner
    inputs = np.load(f"{args.data}/all__inputs.npy").astype(np.int64)
    labels = np.load(f"{args.data}/all__labels.npy").astype(np.int64)

    per_batch, per_batch_x10 = [], []
    for i in range(args.n_batches):
        lo = i * args.batch_size
        hi = min(lo + args.batch_size, len(inputs))
        if lo >= len(inputs):
            break
        b = _batch(inputs, labels, lo, hi, args.batch_size)
        per_batch.append(measure_A(net, b, args.delta_std, args.seed + i))
        per_batch_x10.append(measure_A(net, b, args.delta_std * args.delta_mult, args.seed + i))

    keys = sorted(per_batch[0].keys(), key=lambda s: int(s[1:]))
    cf = {k: float(np.mean([r[k]["contraction_factor"] for r in per_batch])) for k in keys}
    cf_x10 = {k: float(np.mean([r[k]["contraction_factor"] for r in per_batch_x10])) for k in keys}
    ratio = {k: [float(np.mean([r[k]["ratio_curve"][j] for r in per_batch]))
                 for j in range(len(per_batch[0][k]["ratio_curve"]))] for k in keys}

    b0 = _batch(inputs, labels, 0, min(args.batch_size, len(inputs)), args.batch_size)
    B = measure_B(net, b0, args.n_samples, args.extrap_T, args.seed)

    res = dict(
        checkpoint=os.path.join(args.ckpt_dir, args.step),
        contraction_factor=cf,
        contraction_factor_delta_x=dict(mult=args.delta_mult, factor=cf_x10),
        ratio_curve=ratio,
        var_curve=B["var_curve"],
        zh_mean_norm_curve=B["zh_mean_norm_curve"],
        measure_B=B,
        verdict=verdict(list(cf.values())),
        n_batches=len(per_batch), batch_size=args.batch_size, seed=args.seed,
        delta_std=args.delta_std,
        config=dict(
            H_cycles=inner.config.H_cycles, L_cycles=inner.config.L_cycles,
            use_fp_transformer=getattr(inner.config, "use_fp_transformer", None),
            norm_type=getattr(inner.config, "norm_type", None),
            residual_scale=getattr(inner.config, "residual_scale", None),
            alpha_1_init=getattr(inner.config, "alpha_1_init", None),
            alpha_2_init=getattr(inner.config, "alpha_2_init", None),
            eps_scale=_eps_scale(inner),
            damping_eta=float(getattr(inner.config, "damping_eta", 1.0)),
        ),
    )
    print(f"{res['checkpoint']}  factor={cf}  verdict={res['verdict']}")
    print(f"  var_curve[0]={B['var_curve'][0]:.4g} -> [-1]={B['var_curve'][-1]:.4g}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
