"""Task 4: q_halt firing rate per supervision step (posterior path).

train/steps is filtered out of stdout, so recover the halting behaviour from the
checkpoint: run the posterior path, and at each supervision step record the
fraction of samples with (q_halt_logits > 0) -- the exact condition gram.py uses
to halt in training (gram.py:458).  A rate near 1.0 at step 1 means the model
halts immediately (learning only ever saw depth ~1).

Works for both sudoku (seq_len 81) and nqueens (64); test data + labels come from
config.data_paths[0]/test.
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.gram_harness import load_model


def load_test_from_config(config, n_max):
    d = os.path.join(config.data_paths[0], "test")
    inp = np.load(os.path.join(d, "all__inputs.npy")).astype(np.int64)[:n_max]
    lab = np.load(os.path.join(d, "all__labels.npy")).astype(np.int64)[:n_max]
    return inp, lab


@torch.inference_mode()
def firing(net, inputs, labels, n_steps=3, use_posterior=True, seed=0):
    L = inputs.shape[1]
    B = len(inputs)
    batch = {
        "inputs": torch.from_numpy(inputs.astype(np.int32)).cuda(),
        "labels": torch.from_numpy(labels.astype(np.int32)).cuda(),
        "puzzle_identifiers": torch.zeros(B, dtype=torch.int32).cuda(),
    }
    torch.random.manual_seed(seed)
    with torch.device("cuda"):
        carry = net.initial_carry(batch)
    rates, logit_means = [], []
    for _ in range(n_steps):
        carry, _, _, out, _ = net(carry=carry, batch=batch,
                                  return_keys=["q_halt_logits"], use_posterior=use_posterior)
        q = out["q_halt_logits"].float()
        rates.append(float((q > 0).float().mean()))
        logit_means.append(float(q.mean()))
    return rates, logit_means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--num-puzzles", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    net, config = load_model(args.ckpt_dir, args.step)
    inputs, labels = load_test_from_config(config, args.num_puzzles)
    rates, lmeans = firing(net, inputs, labels, args.steps, use_posterior=True)
    res = {
        "checkpoint": os.path.join(args.ckpt_dir, args.step),
        "data": config.data_paths[0], "n": len(inputs), "path": "posterior",
        "halt_max_steps": net.model.config.halt_max_steps,
        "q_halt_fire_rate_by_step": rates,       # fraction with q_halt_logits > 0
        "q_halt_logit_mean_by_step": lmeans,
    }
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
