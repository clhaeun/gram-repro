"""B3: valid-rate bucketed by the input's number of solutions (sidecar).

Answers "does prior fail even on unique-solution inputs?".  Runs both paths at
K=1 and reports per-bucket valid rate.  seq_len from data (no 81 hardcoding).
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.gram_harness import load_model
from evaluators.nqueens import verify_boards
from experiments.diag_posterior_prior import run_path, load_test
import hashlib

def ihash(row):
    return hashlib.sha1(np.ascontiguousarray(row, dtype=np.uint8).tobytes()).hexdigest()[:16]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True); ap.add_argument("--step", required=True)
    ap.add_argument("--data", default="data/nqueens-8"); ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    side = json.load(open(os.path.join(args.data, f"sidecar_{args.n}.json")))["inputs"]
    net, config = load_model(args.ckpt_dir, args.step)
    inputs, labels = load_test(args.data)
    nsol = np.array([len(side[ihash(r)]["solutions"]) for r in inputs])

    print(f"{args.ckpt_dir}/{args.step}  test={len(inputs)}")
    for mode, up in [("posterior", True), ("prior", False)]:
        preds = run_path(net, inputs, labels, up, args.n)
        valid = verify_boards(preds, inputs, args.n)["valid"]
        print(f"\n[{mode}]  #sol : n_inputs  valid_rate")
        for lo, hi in [(1,1),(2,2),(3,4),(5,8),(9,18)]:
            m = (nsol>=lo)&(nsol<=hi)
            if m.any():
                lbl = f"{lo}" if lo==hi else f"{lo}-{hi}"
                print(f"    {lbl:>5} : {int(m.sum()):>7}  {valid[m].mean()*100:6.2f}%")

if __name__ == "__main__":
    main()
