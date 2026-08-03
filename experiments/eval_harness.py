"""Task-agnostic test-time-scaling eval harness for GRAM checkpoints.

★ 상태 (2026-08-03): **결과 산출에서는 은퇴, contract 검증 라이브러리로만 현역.**
  지표 산출은 sweep_harness.py(valid 축)가 맡는다 -- 이쪽 NQueensTask.verify 는
  strict(valid AND consistent)를 정답 기준으로 반환해 논문 헤드라인 축과 다르다.
  루트에 남는 이유: dataset/verify_nqueens_dataset.py:210 이 `_input_hash` /
  `NQueensTask` 를 import 해 [12] 항목(빌더-harness 해시/coverage contract)을 검증한다.
  옮기거나 지우면 그쪽 try/except 에 먹혀 검증이 조용히 SKIP 된다.

Generalises the e1/e3 one-off scripts (which stay untouched) into a single
entry point with a pluggable task:

    sample_candidates(model, batch, N, sigma_scale=1.0) -> [B, N, L]
    verify(candidates, inputs, task)                    -> [B, N] bool
    compute_metrics(...)                                -> dict

Correctness judgement is deliberately split in two, because the two task
families disagree about what "right" means:

  * unique-solution tasks (sudoku): a candidate is correct iff it equals the
    label.  The verifier still exists, but only as a *selection* signal.
  * multi-solution tasks (nqueens): the label is one witness among many, so
    correctness is the strict verifier (valid board AND consistent with the
    given clues).  Comparing against the label would undercount.

RNG note: sample_dataset() reproduces experiments/gram_harness.rollout()'s
stream exactly -- candidate-index-outer, batch-inner, reseeded per candidate
as seed*100003+k, fixed-size padded batches.  That is what makes the
halt_only regression anchor land on its recorded value instead of merely
within sampling noise (n=1000 gives ~1%p of noise per re-draw, which is
wider than the +-0.5%p gate).
"""
import argparse
import json
import os
import sys
from collections import Counter
from math import comb

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.gram_harness import load_model, load_puzzles, sigma_scale as _sigma_scale_ctx

PAD = 0


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #
class Task:
    """Plugin interface.  All methods take/return numpy, shapes [B, N, L]."""

    name = "base"
    unique_solution = True

    def verify(self, cand, inputs):
        """[B, N, L], [B, L] -> [B, N] bool.  Strict, self-contained validity."""
        raise NotImplementedError

    def is_correct(self, cand, inputs, labels):
        """[B, N] bool -- the metric's notion of a right answer."""
        if self.unique_solution:
            return (cand == labels[:, None, :]).all(-1)
        return self.verify(cand, inputs)

    def coverage(self, cand, inputs):
        """Optional: fraction of the known solution set hit.  None if N/A."""
        return None


class SudokuTask(Task):
    """Tokens: 0=pad, 1=blank, 2..10 = digits 1..9 (build_sudoku_dataset +1 shift)."""

    name = "sudoku"
    unique_solution = True

    def verify(self, cand, inputs):
        B, N, L = cand.shape
        assert L == 81, f"sudoku expects seq_len 81, got {L}"
        g = cand.reshape(B * N, 9, 9)

        ok = ((g >= 2) & (g <= 10)).all(-1).all(-1)

        def _all_distinct(x):
            # x [M, 9, 9] -> [M] : every one of the 9 groups is a permutation of 2..10
            m = x.shape[0]
            oh = np.zeros((m, 9, 11), dtype=bool)
            idx = np.clip(x, 0, 10)
            np.put_along_axis(oh, idx, True, axis=2)
            return oh[:, :, 2:11].sum(-1).min(-1) == 9

        ok &= _all_distinct(g)                                   # rows
        ok &= _all_distinct(g.transpose(0, 2, 1))                # cols
        boxes = g.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
        ok &= _all_distinct(boxes)                               # 3x3 boxes

        # consistency with the clues (input token 1 == blank)
        clue = inputs[:, None, :] != 1
        ok &= np.where(clue, cand == inputs[:, None, :], True).reshape(B * N, L).all(-1)
        return ok.reshape(B, N)


class NQueensTask(Task):
    """Tokens: 0=pad, 1=empty, 2=queen.  Multi-solution -> verifier is truth."""

    name = "nqueens"
    unique_solution = False

    def __init__(self, n=8, sidecar=None):
        self.n = n
        # sidecar_{n}.json: {"meta": {...}, "inputs": {hash: {"solutions": [[col per row], ...]}}}
        # solutions are column-index encoded; expand to flat token boards once, up front.
        self.sidecar = {}
        if sidecar:
            from dataset.nqueens_verify import boards_to_tokens

            meta = sidecar.get("meta", {})
            assert meta.get("encoding", "columns") == "columns", meta
            self.n = meta.get("n", n)
            for h, entry in sidecar["inputs"].items():
                cols = np.asarray(entry["solutions"], dtype=np.int64)
                self.sidecar[h] = boards_to_tokens(cols, self.n)

    def verify(self, cand, inputs):
        from dataset.nqueens_verify import verify as nq_verify

        B, N, L = cand.shape
        flat_in = np.repeat(inputs[:, None, :], N, axis=1).reshape(B * N, L)
        out = nq_verify(cand.reshape(B * N, L), flat_in, self.n)
        return np.asarray(out["strict"]).reshape(B, N)

    def coverage(self, cand, inputs):
        """Mean fraction of each puzzle's full solution set present in the candidates."""
        if not self.sidecar:
            return None
        B, N, L = cand.shape
        cov = []
        for b in range(B):
            key = _input_hash(inputs[b])
            sols = self.sidecar.get(key)
            if sols is None or len(sols) == 0:
                continue
            sols = {tuple(s) for s in sols}
            got = {tuple(c) for c in cand[b]} & sols
            cov.append(len(got) / len(sols))
        return float(np.mean(cov)) if cov else None


def _input_hash(row):
    """Must match build_nqueens_dataset: sha1(flat token board uint8 bytes)[:16]."""
    import hashlib

    return hashlib.sha1(np.ascontiguousarray(row, dtype=np.uint8).tobytes()).hexdigest()[:16]


def make_task(name, **kw):
    if name == "sudoku":
        return SudokuTask()
    if name == "nqueens":
        return NQueensTask(**kw)
    raise ValueError(f"unknown task {name!r}")


def verify(candidates, inputs, task):
    """Spec-level entry point: [B, N, L], [B, L], Task -> [B, N] bool."""
    return task.verify(candidates, inputs)


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def _make_batch(inputs, labels, lo, hi, batch_size, seq_len):
    """Fixed-size padded batch so shapes stay static across the tail batch."""
    inp = np.zeros((batch_size, seq_len), dtype=np.int32)
    lab = np.zeros((batch_size, seq_len), dtype=np.int32)
    inp[: hi - lo] = inputs[lo:hi]
    lab[: hi - lo] = labels[lo:hi]
    return {
        "inputs": torch.from_numpy(inp).cuda(),
        "labels": torch.from_numpy(lab).cuda(),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.int32).cuda(),
    }


@torch.inference_mode()
def _forward_once(model, batch, seq_len):
    """One stochastic rollout to halting.  Returns (preds [B, L], v [B])."""
    bs = batch["inputs"].shape[0]
    with torch.device("cuda"):
        carry = model.initial_carry(batch)

    p_final = torch.zeros(bs, seq_len, dtype=torch.long, device="cuda")
    v_final = torch.zeros(bs, dtype=torch.float32, device="cuda")
    done = torch.zeros(bs, dtype=torch.bool, device="cuda")
    while True:
        carry, _, _, out, all_finish = model(
            carry=carry, batch=batch, return_keys=["preds", "lrpm_logits"],
            use_posterior=False)
        newly = carry.halted & (~done)
        if newly.any():
            p_final[newly] = out["preds"][newly].long()
            v_final[newly] = out["lrpm_logits"][newly][:, -1].float()
        done |= carry.halted
        if all_finish:
            break
    return p_final.cpu().numpy(), v_final.cpu().numpy()


def sample_candidates(model, batch, N, sigma_scale=1.0, *, seed=0, seq_len=None):
    """N independent prior-path rollouts for ONE batch -> ([B, N, L], [B, N]).

    GRAM's transition is stochastic, so repetition alone gives distinct answers;
    no test-time augmentation is involved.  For multi-batch datasets use
    sample_dataset(), which owns the RNG ordering.
    """
    seq_len = seq_len or batch["inputs"].shape[1]
    B = batch["inputs"].shape[0]
    cand = np.zeros((B, N, seq_len), dtype=np.int64)
    v = np.zeros((B, N), dtype=np.float32)
    with _sigma_scale_ctx(model, sigma_scale):
        for k in range(N):
            torch.random.manual_seed(seed * 100003 + k)
            p, vv = _forward_once(model, batch, seq_len)
            cand[:, k], v[:, k] = p, vv
    return cand, v


def sample_dataset(model, inputs, labels, N, sigma_scale=1.0, batch_size=512, seed=0,
                   verbose=True):
    """[n, L] -> ([n, N, L], [n, N]).  RNG stream matches gram_harness.rollout."""
    n, seq_len = inputs.shape
    cand = np.zeros((n, N, seq_len), dtype=np.int64)
    v = np.zeros((n, N), dtype=np.float32)

    with _sigma_scale_ctx(model, sigma_scale):
        for k in range(N):
            torch.random.manual_seed(seed * 100003 + k)
            for lo in range(0, n, batch_size):
                hi = min(lo + batch_size, n)
                batch = _make_batch(inputs, labels, lo, hi, batch_size, seq_len)
                p, vv = _forward_once(model, batch, seq_len)
                cand[lo:hi, k] = p[: hi - lo]
                v[lo:hi, k] = vv[: hi - lo]
            if verbose:
                print(f"  sample sigma={sigma_scale} {k + 1}/{N}", flush=True)
    return cand, v


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def pass_at_k(num_correct, n_samples, k):
    """Unbiased pass@k from num_correct hits out of n_samples iid draws."""
    if n_samples - num_correct < k:
        return 1.0
    return 1.0 - comb(n_samples - num_correct, k) / comb(n_samples, k)


def _majority(rows):
    """[K, L] -> most common exact row; ties broken by first appearance."""
    counts = Counter(tuple(r) for r in rows)
    return np.array(counts.most_common(1)[0][0], dtype=np.int64)


def compute_metrics(cand, v, inputs, labels, task, ks=(1, 5, 10, 20)):
    """cand [n, N, L], v [n, N] -> metric dict."""
    n, N, L = cand.shape
    correct = task.is_correct(cand, inputs, labels)          # [n, N]
    passes = task.verify(cand, inputs)                       # [n, N]
    num_correct = correct.sum(1)

    out = {
        "n": int(n),
        "N": int(N),
        "task": task.name,
        "single": float(correct[:, 0].mean()),
        "mean_sample": float(correct.mean()),
        "by_k": {},
    }

    for k in sorted({k for k in ks if k <= N} | {N}):
        maj_ok = np.zeros(n, dtype=bool)
        ver_ok = np.zeros(n, dtype=bool)
        lprm_ok = np.zeros(n, dtype=bool)
        distinct = np.zeros(n, dtype=np.float64)

        for i in range(n):
            sub = cand[i, :k]
            maj = _majority(sub)
            maj_idx = int(np.flatnonzero((sub == maj).all(-1))[0])
            maj_ok[i] = correct[i, maj_idx]

            hit = np.flatnonzero(passes[i, :k])
            ver_ok[i] = correct[i, int(hit[0])] if hit.size else maj_ok[i]

            lprm_ok[i] = correct[i, int(np.argmax(v[i, :k]))]
            distinct[i] = len({tuple(r) for r in sub})

        row = {
            "majority": float(maj_ok.mean()),
            "pass": float(np.mean([pass_at_k(int(c), N, k) for c in num_correct])),
            "pass_first_k": float(correct[:, :k].any(1).mean()),
            "verifier_selected": float(ver_ok.mean()),
            "lprm_best": float(lprm_ok.mean()),
            "distinct": float(distinct.mean()),
            "verifier_hit_rate": float((passes[:, :k].any(1)).mean()),
        }
        cov = task.coverage(cand[:, :k], inputs)
        if cov is not None:
            row["coverage"] = cov
        out["by_k"][str(k)] = row

    # verifier soundness: does "passes the verifier" imply "correct"?
    if passes.any():
        out["verifier_precision"] = float(correct[passes].mean())
    out["verifier_pass_rate"] = float(passes.mean())
    out["hits_histogram"] = {str(c): int(x) for c, x in sorted(Counter(num_correct.tolist()).items())}
    return out


def print_report(m):
    print(f"\n  task={m['task']}  n={m['n']}  N={m['N']}")
    print(f"  single           = {m['single'] * 100:.2f}%")
    print(f"  mean_sample      = {m['mean_sample'] * 100:.2f}%")
    print(f"  verifier_pass    = {m['verifier_pass_rate'] * 100:.2f}%"
          + (f"   precision={m['verifier_precision'] * 100:.2f}%" if "verifier_precision" in m else ""))
    hdr = f"\n  {'K':>3} {'majority':>9} {'pass':>8} {'passfirst':>10} {'verif_sel':>10} {'lprm_best':>10} {'distinct':>9}"
    cov = any("coverage" in r for r in m["by_k"].values())
    print(hdr + (f" {'coverage':>9}" if cov else ""))
    for k, r in sorted(m["by_k"].items(), key=lambda kv: int(kv[0])):
        line = (f"  {k:>3} {r['majority'] * 100:>8.2f}% {r['pass'] * 100:>7.2f}% "
                f"{r['pass_first_k'] * 100:>9.2f}% {r['verifier_selected'] * 100:>9.2f}% "
                f"{r['lprm_best'] * 100:>9.2f}% {r['distinct']:>9.2f}")
        if cov:
            line += f" {r.get('coverage', float('nan')) * 100:>8.2f}%"
        print(line)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)
    print(f"\nsaved -> {path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", default="step_65100")
    ap.add_argument("--task", default="sudoku", choices=["sudoku", "nqueens"])
    ap.add_argument("--nq-n", type=int, default=8, help="board size for --task nqueens")
    ap.add_argument("--sidecar", default=None, help="sidecar_{n}.json for coverage@K")
    ap.add_argument("--num-puzzles", type=int, default=1000)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--ks", default="1,5,10,20")
    ap.add_argument("--sigma-scale", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None, help="free-form label copied into the json")
    args = ap.parse_args()

    sidecar = None
    if args.sidecar:
        with open(args.sidecar) as f:
            sidecar = json.load(f)
    task = make_task(args.task, **({"n": args.nq_n, "sidecar": sidecar} if args.task == "nqueens" else {}))

    print(f"loading {args.ckpt_dir}/{args.step}", flush=True)
    net, config = load_model(args.ckpt_dir, args.step)
    inputs, labels = load_puzzles(config, args.num_puzzles, seed=args.seed)
    print(f"  {len(inputs)} puzzles, seq_len={inputs.shape[1]}", flush=True)

    cand, v = sample_dataset(net, inputs, labels, args.k, sigma_scale=args.sigma_scale,
                             batch_size=args.batch_size, seed=args.seed)
    m = compute_metrics(cand, v, inputs, labels, task,
                        ks=tuple(int(x) for x in args.ks.split(",")))
    m["checkpoint"] = os.path.join(args.ckpt_dir, args.step)
    m["seed"] = args.seed
    m["sigma_scale"] = args.sigma_scale
    if args.tag:
        m["tag"] = args.tag
    print_report(m)

    out = args.out or os.path.join("experiments", "results",
                                   f"{args.task}_{os.path.basename(args.ckpt_dir)}_{args.step}_k{args.k}.json")
    save_json(m, out)


if __name__ == "__main__":
    main()
