"""[50 ②] traj_*.npz 에 final_valid 추가 (GRAM 셀).

experiments/residual_probe.run_cell 과 ★같은 forward★ 를 같은 seed·sigma 로 재현하고,
최종 transition 상태(carry.inner_carry.z_H)를 lm_head 로 디코드해 판정만 뽑는다.
기존 h / loss / loss_stablemax / h_mean / t_index 는 재생성하지 않고 키만 더한다.

산출 키
  final_valid  (N,) int8   최종 출력이 유효한 해인가
  final_strict (N,) int8   valid ∧ consistent
  final_iter   ()   int32  재현된 forward 의 최종 transition 번호 T

  python experiments/final_valid_probe.py --key nq10_sig1 \
      --ckpt-dir checkpoints/Nqueens-10-ACT-torch/nq10_repro_s0 --step step_138534 \
      --data data/nqueens-10/test --sigma 1.0 --n 10
"""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eps_propagation_probe import load_model               # noqa: E402
from models.losses import IGNORE_LABEL_ID                              # noqa: E402
from evaluators.nqueens import verify_boards                           # noqa: E402

OUT_DIR = "results/figdata"


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--n-probe", type=int, default=256)
    ap.add_argument("--n-traj", type=int, default=50)
    ap.add_argument("--traj-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

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
        "inputs": torch.from_numpy(inp.astype(np.int32)).cuda(),
        "labels": torch.from_numpy(lab.astype(np.int32)).cuda(),
        "puzzle_identifiers": torch.from_numpy(pids.astype(np.int32)).cuda(),
    }
    batch["labels"] = torch.where(batch["labels"] == 0, IGNORE_LABEL_ID, batch["labels"])
    pel = inner.puzzle_emb_len

    # ResidualRecorder 는 난수를 전용 generator 에서 뽑는다 -- 재현하려면 같은 방식이어야
    # 하므로 recorder 를 그대로 붙이되 궤적 누적만 끈다 (keep_traj=False).
    from experiments.residual_probe import ResidualRecorder            # noqa: E402
    rec = ResidualRecorder(inner, seed=args.seed, n_probe=n_probe,
                           labels_traj=batch["labels"][n_probe:], keep_traj=False)
    B, seq_len = batch["inputs"].shape
    rec.h_prev = torch.zeros(B, seq_len + pel, int(cfg.hidden_size),
                             device="cuda", dtype=torch.float32) + inner.H_init.float()

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

    n_calls = len(rec.rows_all)
    assert n_calls == T, f"transition {n_calls} != {T}"
    z_ref = carry.inner_carry.z_H.float()
    max_dev = float((rec.h_last - z_ref).abs().max())
    assert max_dev == 0.0, f"h 재구성 불일치: max|Δ| = {max_dev}"

    logits = inner.lm_head(z_ref)[:, pel:].contiguous()
    preds = torch.argmax(logits, dim=-1).cpu().numpy()[n_probe:]
    puz = np.repeat(inputs[i:i + 1].astype(np.int64), args.n_traj, 0)
    r = verify_boards(preds, puz, args.n)
    fv = r["valid"].astype(np.int8)
    fs = (r["valid"] & r["consistent"]).astype(np.int8)

    path = os.path.join(OUT_DIR, f"traj_{args.key}.npz")
    old = dict(np.load(path))
    keep = {k: old[k] for k in old}
    keep["final_valid"] = fv
    keep["final_strict"] = fs
    keep["final_iter"] = np.int32(T)
    np.savez(path, **keep)

    mp = path.replace(".npz", ".meta.json")
    meta = json.load(open(mp))
    meta["final_valid"] = {
        "shape": list(fv.shape), "dtype": "int8",
        "definition": "재현 forward 의 최종 transition 상태(carry.inner_carry.z_H)를 lm_head 로 "
                      "디코드해 evaluators.nqueens.verify_boards 로 판정한 valid (0/1)",
        "final_strict": "valid ∧ consistent (입력 퀸 보존)",
        "final_iter": int(T),
        "last_saved_frame_t": int(old["t_index"][-1]),
        "reconstruction_check": f"마지막 transition h 와 carry.z_H 의 max|Δ| = {max_dev}",
        "note": ("h / loss / loss_stablemax / h_mean / t_index 는 재생성하지 않고 그대로 보존했다."),
        "n_valid": int(fv.sum()), "n_strict": int(fs.sum()), "n_traj": int(len(fv)),
        "sigma_scale": float(args.sigma), "seed": args.seed,
    }
    with open(mp, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[out] {path}  key={args.key} final_iter={T} "
          f"last_saved_frame_t={int(old['t_index'][-1])} "
          f"final_valid={int(fv.sum())}/{len(fv)} final_strict={int(fs.sum())}/{len(fs)}")


if __name__ == "__main__":
    main()
