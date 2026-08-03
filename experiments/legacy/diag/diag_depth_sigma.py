"""Task 3+5: prior-path depth x sigma 2x2 grid, with failure-reason breakdown.

For a checkpoint, run the PRIOR path (K=1, test set) under every combination of
  halt_max_steps in {1, 16}   (learning saw depth ~1; eval normally forces 16)
  sigma_scale    in {0.0, 1.0} (0.0 = mean-only, no noise; 1.0 = trained noise)
and report valid/consistent/queen_count plus, for the invalid boards, which
constraint failed (count / row / col / diag) -- "wrong queen count" and
"right count but colliding" are different pathologies.

Depth is set by overriding inner model config.halt_max_steps after load; the
script prints the observed number of inference steps to confirm it took effect.
seq_len comes from the data (no SEQ_LEN=81 hardcoding).
"""
import argparse, json, os, sys
import numpy as np, torch
# experiments/legacy/diag/ 로 내려왔으므로 repo root 는 부모의 부모의 부모의 부모다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from experiments.gram_harness import load_model, sigma_scale as sigma_scale_ctx
from evaluators.nqueens import verify_boards
from experiments.legacy.diag.diag_posterior_prior import load_test


@torch.inference_mode()
def run_prior(net, inputs, labels, n, sigma, batch_size=512, seed=0):
    """Prior path, K=1, to halting. Returns (preds [N,L], observed_steps)."""
    N, L = inputs.shape
    preds = np.zeros((N, L), dtype=np.int64)
    steps_seen = 0
    torch.random.manual_seed(seed)
    with sigma_scale_ctx(net, sigma):
        for lo in range(0, N, batch_size):
            hi = min(lo + batch_size, N)
            bs = hi - lo
            inp = np.zeros((batch_size, L), dtype=np.int32)
            inp[:bs] = inputs[lo:hi]
            batch = {
                "inputs": torch.from_numpy(inp).cuda(),
                "labels": torch.from_numpy(np.zeros((batch_size, L), np.int32)).cuda(),
                "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.int32).cuda(),
            }
            with torch.device("cuda"):
                carry = net.initial_carry(batch)
            p_final = torch.zeros(batch_size, L, dtype=torch.long, device="cuda")
            done = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
            nsteps = 0
            while True:
                carry, _, _, out, all_finish = net(
                    carry=carry, batch=batch, return_keys=["preds"], use_posterior=False)
                nsteps += 1
                newly = carry.halted & (~done)
                if newly.any():
                    p_final[newly] = out["preds"][newly].long()
                done |= carry.halted
                if all_finish:
                    break
            steps_seen = max(steps_seen, nsteps)
            preds[lo:hi] = p_final[:bs].cpu().numpy()
    return preds, steps_seen


def summarize(preds, inputs, n):
    r = verify_boards(preds, inputs, n)
    qc = r["queen_count"]
    inval = ~r["valid"]
    ninv = int(inval.sum())
    # failure-reason breakdown among INVALID boards (a board can fail several)
    reasons = {k: int((~r[k] & inval).sum()) for k in
               ("count_ok", "row_ok", "col_ok", "d1_ok", "d2_ok")}
    # among boards with the RIGHT count, how many still collide
    right_count = r["count_ok"]
    collide_with_right_count = int((right_count & inval).sum())
    return {
        "valid_rate": float(r["valid"].mean()),
        "consistent_rate": float(r["consistent"].mean()),
        "strict_rate": float(r["strict"].mean()),
        "queen_count_mean": float(qc.mean()),
        "queen_count_min": int(qc.min()),
        "queen_count_max": int(qc.max()),
        "n_invalid": ninv,
        "fail_count_ok": reasons["count_ok"],   # invalid boards whose queen count != n
        "fail_row": reasons["row_ok"],
        "fail_col": reasons["col_ok"],
        "fail_diag1": reasons["d1_ok"],
        "fail_diag2": reasons["d2_ok"],
        "right_count_but_invalid": collide_with_right_count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", default="data/nqueens-8")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    net, config = load_model(args.ckpt_dir, args.step)
    inner = net.model.inner
    inputs, labels = load_test(args.data)
    orig_depth = inner.config.halt_max_steps
    print(f"loaded {args.ckpt_dir}/{args.step}  test={inputs.shape}  orig halt_max_steps={orig_depth}")

    grid = {}
    for depth in (1, 16):
        inner.config.halt_max_steps = depth
        for sigma in (0.0, 1.0):
            preds, steps_seen = run_prior(net, inputs, labels, args.n, sigma)
            s = summarize(preds, inputs, args.n)
            s["observed_steps"] = steps_seen
            key = f"depth{depth}_sigma{sigma}"
            grid[key] = s
            print(f"\n[{key}]  observed_steps={steps_seen}")
            print(f"  valid={s['valid_rate']*100:.2f}%  consistent={s['consistent_rate']*100:.2f}%  "
                  f"queen_count mean={s['queen_count_mean']:.2f} [{s['queen_count_min']},{s['queen_count_max']}]")
            print(f"  invalid={s['n_invalid']}  fail(count={s['fail_count_ok']} row={s['fail_row']} "
                  f"col={s['fail_col']} d1={s['fail_diag1']} d2={s['fail_diag2']})  "
                  f"right_count_but_invalid={s['right_count_but_invalid']}")
    inner.config.halt_max_steps = orig_depth

    res = {"checkpoint": os.path.join(args.ckpt_dir, args.step), "n": args.n, "grid": grid}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
