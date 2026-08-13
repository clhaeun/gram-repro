"""Acceptance check for a generated N-Queens dataset.

``build_nqueens_dataset.py`` is kept thin on purpose; the guarantees it used to
assert inline live here instead.  This script does NOT import the builder --
it re-enumerates the solutions, re-derives every clue set, and recomputes each
input's full solution set from scratch, then compares that against what is on
disk.  A builder bug therefore cannot hide itself.

    python dataset/verify_nqueens_dataset.py --data data/nqueens-8 --n 8
    python dataset/verify_nqueens_dataset.py --data data/nqueens-10 --n 10

Exit code 0 = every check passed.  Run it after (re)generating a dataset.
"""

import argparse
import hashlib
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nqueens_verify import verify, boards_to_tokens  # noqa: E402


SOLUTION_ANCHORS = {4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724, 11: 2680, 12: 14200}
# (n, removals) -> number of (puzzle, solution) pairs
PAIR_ANCHORS = {(8, (5, 6, 7)): 8464, (10, (7, 8, 9)): 126700}

FAILURES = []


def chk(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)


def enumerate_solutions(n):
    out, cols = [], [0] * n
    uc, ud, ua = [False] * n, [False] * (2 * n - 1), [False] * (2 * n - 1)

    def place(r):
        if r == n:
            out.append(tuple(cols))
            return
        for c in range(n):
            if uc[c] or ud[r - c + n - 1] or ua[r + c]:
                continue
            uc[c] = ud[r - c + n - 1] = ua[r + c] = True
            cols[r] = c
            place(r + 1)
            uc[c] = ud[r - c + n - 1] = ua[r + c] = False

    place(0)
    return np.array(sorted(out), dtype=np.int32)


def board_hash(flat_tokens):
    return hashlib.sha1(np.ascontiguousarray(flat_tokens, dtype=np.uint8).tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nqueens-8")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=0.85)
    args = ap.parse_args()
    n, root = args.n, args.data

    D = {}
    for sp in ("train", "test"):
        D[sp] = {k: np.load(f"{root}/{sp}/all__{k}.npy")
                 for k in ("inputs", "labels", "puzzle_indices", "group_indices", "puzzle_identifiers")}
        D[sp]["meta"] = json.load(open(f"{root}/{sp}/dataset.json"))
    side = json.load(open(f"{root}/sidecar_{n}.json"))
    removals = tuple(side["meta"]["removals"])
    print(f"n={n}  root={root}  removals={removals}")

    # --- independent recomputation -----------------------------------------
    sols = enumerate_solutions(n)
    chk(sols.shape[0] == SOLUTION_ANCHORS.get(n, sols.shape[0]),
        f"[1] complete solutions {sols.shape[0]} == anchor {SOLUTION_ANCHORS.get(n)}")

    keep_sets = [tuple(sorted(set(range(n)) - set(d)))
                 for k in removals for d in itertools.combinations(range(n), k)]
    ref_clues, ref_rows, num_pairs = {}, {}, 0
    for si, sol in enumerate(sols):
        for keep in keep_sets:
            clues = tuple((r, int(sol[r])) for r in keep)
            b = np.ones(n * n, np.uint8)
            for r, c in clues:
                b[r * n + c] = 2
            h = board_hash(b)
            ref_clues.setdefault(h, clues)
            ref_rows.setdefault(h, []).append(si)
            num_pairs += 1
    anchor = PAIR_ANCHORS.get((n, removals))
    chk(num_pairs == (anchor if anchor is not None else num_pairs),
        f"[2] (puzzle, solution) pairs {num_pairs} == anchor {anchor}")
    chk(len(ref_clues) == len({v for v in ref_clues.values()}), "[2] no sha1 collision among clue sets")

    # every solution containing all of an input's clues, computed by broadcast
    indep, by_m = {}, {}
    for h, cl in ref_clues.items():
        by_m.setdefault(len(cl), []).append(h)
    for m, hs in by_m.items():
        for s in range(0, len(hs), 2048):
            blk = hs[s:s + 2048]
            cl = np.array([ref_clues[h] for h in blk], np.int32)
            match = (sols[:, cl[..., 0]] == cl[..., 1][None]).all(axis=2)
            for j, h in enumerate(blk):
                indep[h] = sorted(np.nonzero(match[:, j])[0].tolist())

    # --- tokens / shapes ---------------------------------------------------
    for sp in ("train", "test"):
        x, y, m = D[sp]["inputs"], D[sp]["labels"], D[sp]["meta"]
        chk(x.shape[1] == n * n == m["seq_len"], f"[3] {sp}: seq_len {m['seq_len']} == n*n")
        chk(set(np.unique(x)) <= {1, 2} and set(np.unique(y)) <= {1, 2},
            f"[3] {sp}: tokens in {{1, 2}} only (PAD 0 unused)")
        chk(m["vocab_size"] == 3 and m["pad_id"] == 0, f"[3] {sp}: vocab_size=3, pad_id=0")
        chk(x.dtype == np.int32 and y.dtype == np.int32, f"[3] {sp}: dtype int32")

    # --- index bookkeeping -------------------------------------------------
    for sp in ("train", "test"):
        pi, gi, pid = D[sp]["puzzle_indices"], D[sp]["group_indices"], D[sp]["puzzle_identifiers"]
        x, m = D[sp]["inputs"], D[sp]["meta"]
        chk(np.array_equal(pi, np.arange(len(x) + 1)), f"[4] {sp}: puzzle_indices == arange(N+1)")
        chk(gi[0] == 0 and gi[-1] == len(x) and np.all(np.diff(gi) > 0),
            f"[4] {sp}: group_indices 0..N strictly increasing")
        chk(len(gi) - 1 == m["total_groups"], f"[4] {sp}: total_groups {m['total_groups']} == len(group_indices)-1")
        chk(m["total_puzzles"] == len(x), f"[4] {sp}: total_puzzles {m['total_puzzles']} == rows {len(x)}")
        chk(len(pid) == len(x) and set(np.unique(pid)) == {0}, f"[4] {sp}: puzzle_identifiers all 0, length N")
        chk(m["num_puzzle_identifiers"] == 1 and m["mean_puzzle_examples"] == 1,
            f"[4] {sp}: num_puzzle_identifiers=1, mean_puzzle_examples=1")

    for sp in ("train", "test"):
        gi, x = D[sp]["group_indices"], D[sp]["inputs"]
        chk(np.array_equal(gi, np.arange(len(x) + 1)), f"[5] {sp}: group_indices == arange(N+1) (행 1개 = group 1개)")
        D[sp]["ghash"] = [board_hash(v) for v in x]

    for sp in ("train", "test"):
        cnt = {}
        for h in D[sp]["ghash"]:
            cnt[h] = cnt.get(h, 0) + 1
        bad = [h for h, c in cnt.items() if c != len(indep[h])]
        chk(not bad, f"[6] {sp}: 입력별 행 수 == 그 입력의 전체 해 개수 (불일치 {len(bad)})")
        y = D[sp]["labels"]
        pairs = {(h, y[i].tobytes()) for i, h in enumerate(D[sp]["ghash"])}
        chk(len(pairs) == len(y), f"[6] {sp}: 중복 (입력, 라벨) 행 0 ({len(y) - len(pairs)}건 중복)")
    tot = len(D["train"]["inputs"]) + len(D["test"]["inputs"])
    chk(tot == (anchor if anchor is not None else tot),
        f"[6] train+test 행 수 {tot} == pair anchor {anchor}")

    # --- split --------------------------------------------------------------
    a, b = set(D["train"]["ghash"]), set(D["test"]["ghash"])
    chk(len(a & b) == 0, "[7] train/test input hashes disjoint (no leakage)")
    chk(a | b == set(ref_clues), f"[7] train u test == all {len(ref_clues)} unique inputs")
    frac = len(a) / (len(a) + len(b))
    chk(abs(frac - args.train_frac) < 0.002, f"[7] train fraction {frac:.4f} ~= {args.train_frac}")

    # --- labels are real solutions -----------------------------------------
    for sp in ("train", "test"):
        r = verify(D[sp]["labels"], D[sp]["inputs"], n)
        chk(r["strict"].all(), f"[8] {sp}: all {len(D[sp]['labels'])} labels strict PASS")

    # --- the verifier is not vacuous ---------------------------------------
    rng = np.random.default_rng(0)
    y, x = D["test"]["labels"].copy(), D["test"]["inputs"]
    for i in range(len(y)):
        # snapshot both index sets first: vacating a queen would otherwise make
        # its own cell a candidate target and could restore the original board
        queens, empties = np.nonzero(y[i] == 2)[0], np.nonzero(y[i] == 1)[0]
        y[i, rng.choice(queens)] = 1
        y[i, rng.choice(empties)] = 2
    chk((y != D["test"]["labels"]).any(-1).all(), "[9] every corruption actually changed the board")
    chk((~verify(y, x, n)["valid"]).all(), f"[9] all {len(y)} one-queen-moved corruptions are invalid")

    # --- sidecar ------------------------------------------------------------
    si = side["inputs"]
    chk(set(si) == set(ref_clues), f"[10] sidecar keys == all unique inputs ({len(si)})")
    bad = [h for h in si if [list(sols[i]) for i in indep[h]] != si[h]["solutions"]]
    chk(not bad, f"[10] sidecar solutions == independently computed solution set (mismatch {len(bad)})")
    bad = [h for h in si if [list(map(int, c)) for c in ref_clues[h]] != si[h]["clues"]]
    chk(not bad, f"[10] sidecar clues == independently computed (mismatch {len(bad)})")
    bad = [h for h in si if si[h]["split"] != ("train" if h in a else "test")]
    chk(not bad, f"[10] sidecar split == actual npy split (mismatch {len(bad)})")
    chk(side["meta"]["num_unique_inputs"] == len(ref_clues)
        and side["meta"]["num_solutions_total"] == len(sols), "[10] sidecar meta counts correct")

    ins, outs = [], []
    for h in list(si)[:3000]:
        xin = np.ones(n * n, np.int32)
        for r, c in si[h]["clues"]:
            xin[r * n + c] = 2
        for s in si[h]["solutions"]:
            ins.append(xin)
            outs.append(boards_to_tokens(np.array(s), n))
    chk(verify(np.stack(outs), np.stack(ins), n)["strict"].all(),
        f"[11] {len(outs)} sidecar (input, solution) pairs strict PASS")

    # --- eval harness contract ---------------------------------------------
    try:
        from experiments.legacy import eval_harness as EH
    except Exception as e:  # torch missing etc. -- not a dataset problem
        print(f"  SKIP  [12] eval_harness unavailable ({type(e).__name__})")
    else:
        hx = D["test"]["inputs"]
        chk([board_hash(v) for v in hx] == [EH._input_hash(v) for v in hx],
            f"[12] eval_harness._input_hash matches builder hash on all {len(hx)} rows")
        task = EH.NQueensTask(n=n, sidecar=side)
        hit = sum(1 for g in D["test"]["ghash"] if g in task.sidecar)
        chk(hit == len(D["test"]["ghash"]), f"[12] sidecar lookup by hash {hit}/{len(D['test']['ghash'])}")
        # coverage() wants [B, N, L]; batch inputs that share a solution count
        by_cnt = {}
        for i, g in enumerate(D["test"]["ghash"]):
            by_cnt.setdefault(len(task.sidecar[g]), []).append(i)
        cnt, idx = max(by_cnt.items(), key=lambda kv: len(kv[1]) * kv[0])
        idx = idx[:64]
        full = np.stack([np.stack(task.sidecar[D["test"]["ghash"][i]]) for i in idx])
        chk(task.coverage(full, hx[idx]) == 1.0,
            f"[12] coverage@K == 1.0 when all {cnt} solutions are supplied")
        half = full[:, :max(1, cnt // 2)]
        want = max(1, cnt // 2) / cnt
        chk(abs(task.coverage(half, hx[idx]) - want) < 1e-9,
            f"[12] coverage@K == {want:.4f} when half are supplied")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("ALL ACCEPTANCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
