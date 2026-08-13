"""[50] Fig 18/19 용 궤적 덤프 (GRAM).  추론 전용, models/ 를 건드리지 않는다.

같은 입력 1개에 대해 노이즈 시드만 바꿔 n_traj 개 궤적을 굴리고 z_H 를 전 transition 저장.

GRAM 의 transition 정의:
  - 1 supervision step = H_cycles(3) 회의 z_H 갱신 (gram.py:468-478 의 H_cycles-1 회 +
    512 의 마지막 1회).  각 z_H 갱신 앞에 L_cycles(4) 회의 z_L 갱신이 있다.
  - supervision step 끝에서 z_H = z_H + eps_scale * eps 로 노이즈가 주입된다 (gram.py:550).
  - 따라서 T = H_cycles x halt_max_steps = 3 x 16 = 48.

저장: np.savez(path, h=h)   h.shape = (n_traj, T, D), float32
  D 축은 puzzle_emb_len 접두를 뺀 토큰 위치들의 평균이다 (FPRM 덤프와 같은 규약).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.recursive_reasoning.gram import TinyRecursiveReasoningModel_ACTV1  # noqa: E402


def load_model(ckpt_dir, step, device, dtype):
    cfg_all = yaml.safe_load(open(os.path.join(ckpt_dir, "all_config.yaml")))
    arch = dict(cfg_all["arch"])
    arch.pop("name")
    arch.pop("loss")
    meta = json.load(open(os.path.join(cfg_all["data_paths"][0], "train", "dataset.json")))
    arch.update(
        batch_size=cfg_all["global_batch_size"],
        vocab_size=meta["vocab_size"],
        seq_len=meta["seq_len"],
        num_puzzle_identifiers=meta["num_puzzle_identifiers"],
    )
    arch["forward_dtype"] = dtype

    model = TinyRecursiveReasoningModel_ACTV1(arch)
    sd = torch.load(os.path.join(ckpt_dir, step), map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.", "").replace("model.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [warn] missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
    model.to(device).eval()
    return model, meta, cfg_all


@torch.inference_mode()
def one_trajectory(model, batch, n_sup, seed):
    inner = model.inner
    cfg = inner.config
    torch.manual_seed(seed)

    x = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    # gram.py:454 와 같은 형태 -- cos_sin 하나뿐이다 (puzzle_emb_len 은 안 받는다).
    seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)

    carry = inner.empty_carry(batch["inputs"].shape[0])
    carry = type(carry)(z_H=carry.z_H.to(x.device), z_L=carry.z_L.to(x.device))
    flag = torch.ones(batch["inputs"].shape[0], dtype=torch.bool, device=x.device)
    carry = inner.reset_carry(flag, carry)
    z_H, z_L = carry.z_H, carry.z_L

    pel = inner.puzzle_emb_len
    rows = []
    for _ in range(n_sup):
        for _h in range(cfg.H_cycles):
            for _l in range(cfg.L_cycles):
                z_L = inner.L_level(z_L, z_H + x, **seq_info)
            z_H = inner.L_level(z_H, z_L, **seq_info)
            rows.append(z_H[0, pel:].float().mean(dim=0).cpu().numpy())
        # supervision step 끝: prior 에서 eps 를 뽑아 주입 (gram.py:508-550 의 eval 경로)
        y = z_H
        if hasattr(inner, "prior_mu"):
            mu_p = inner.prior_mu(y)
            logvar_p = inner.prior_logvar(y) + inner.prior_logvar_bias
            eps = inner.reparameterize(mu_p, logvar_p)
            z_H = z_H + cfg.eps_scale * eps
    return np.stack(rows).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", default="step_138534")
    ap.add_argument("--data", default="data/nqueens-10/test")
    ap.add_argument("--n-traj", type=int, default=50)
    ap.add_argument("--input-index", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, meta, cfg_all = load_model(args.ckpt_dir, args.step, args.device, args.dtype)
    cfg = model.inner.config
    n_sup = int(cfg.halt_max_steps)

    inputs = np.load(os.path.join(args.data, "all__inputs.npy"))
    pid = np.load(os.path.join(args.data, "all__puzzle_identifiers.npy"))
    i = args.input_index
    batch = {
        "inputs": torch.from_numpy(inputs[i : i + 1].astype(np.int64)).to(args.device),
        "puzzle_identifiers": torch.from_numpy(pid[i : i + 1].astype(np.int64)).to(args.device),
    }

    h = np.stack([one_trajectory(model, batch, n_sup, seed) for seed in range(args.n_traj)])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, h=h)

    meta_out = {
        "ckpt_dir": args.ckpt_dir,
        "step": args.step,
        "weight": "EMA (bare step file; .live 는 live weight)",
        "data": args.data,
        "input_index": i,
        "n_rows_in_split": int(inputs.shape[0]),
        "n_traj": args.n_traj,
        "T": int(h.shape[1]),
        "D": int(cfg.hidden_size),
        "shape": list(h.shape),
        "transition_def": "1 transition = L_cycles(%d)회 z_L 갱신 뒤 z_H 갱신 1회" % cfg.L_cycles,
        "T_source": "H_cycles x halt_max_steps = %d x %d" % (cfg.H_cycles, cfg.halt_max_steps),
        "D_reduction": "puzzle_emb_len(%d) 접두 제외 토큰 위치 평균" % model.inner.puzzle_emb_len,
        "arch": {
            "H_cycles": int(cfg.H_cycles),
            "L_cycles": int(cfg.L_cycles),
            "halt_max_steps": int(cfg.halt_max_steps),
            "hidden_size": int(cfg.hidden_size),
            "eps_scale": float(cfg.eps_scale),
            "kl_coef": float(getattr(cfg, "kl_coef", float("nan"))),
        },
        "note": "supervision step 끝에서 prior 로부터 eps 를 뽑아 z_H 에 주입한다 (eval 경로).",
    }
    with open(args.out.replace(".npz", ".meta.json"), "w") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    print(f"[out] {args.out}  shape={h.shape}  {os.path.getsize(args.out):,} bytes")


if __name__ == "__main__":
    main()
