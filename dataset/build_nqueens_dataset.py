"""Build an N-Queens dataset in the TRM/GRAM ``PuzzleDataset`` format.

Task
----
The model is given a *partial* n-queens board (a subset of the queens of some
complete solution) and must output a complete, legal n-queens placement that
keeps every given queen.

Token vocabulary (``vocab_size = 3``)::

    0 = PAD  (never actually emitted by this builder, reserved by convention)
    1 = empty cell
    2 = queen

Following ``build_sudoku_dataset.py`` we assemble *raw* boards (0 = empty,
1 = queen) and add ``+1`` on the way out, so raw 0 becomes token 1 and the PAD
id 0 stays unused.  ``seq_len = n * n``, row-major, cell ``(r, c)`` at ``r*n+c``.

Pipeline
--------
1. Enumerate *all* complete solutions by backtracking
   (n=8 -> 92, n=10 -> 724; asserted).
2. For every complete solution and every k in ``removals`` (default: {5,6,7}
   for n=8, {7,8,9} for n=10), enumerate *every* way of deleting k of the n
   queens (``itertools.combinations``).  The surviving queens are the input
   clues, the full solution is the label.
   (n=8 -> 92*92 = 8464 pairs, n=10 -> 724*175 = 126700 pairs; asserted.)
   Duplicate (input, label) pairs are impossible, but the same *input* legally
   appears with several different labels -- these are NOT deduplicated.
3. Rows are grouped by unique input hash; the 85:15 train/test split is taken
   over *groups*, so no input ever straddles the split (asserted disjoint).
4. Test keeps exactly one row per unique input (no augmentation); train keeps
   all of its rows.

Sidecar
-------
``sidecar_{n}.json`` in ``output_dir`` maps every unique input hash to the FULL
set of complete solutions consistent with that input (computed from the
enumerated solution list, not merely from the generated pairs).  Format::

    {
      "meta": {"n": 8, "encoding": "columns", "num_solutions_total": 92},
      "inputs": {
        "<hash>": {
          "split": "train" | "test",
          "clues": [[row, col], ...],          # the given queens
          "solutions": [[c_0, ..., c_{n-1}], ...]
        },
        ...
      }
    }

``encoding: "columns"`` means each solution is a list of n column indices, one
per row: ``solution[r] = c`` is a queen at ``(r, c)``.  Convert to the flat
token board with ``dataset.nqueens_verify.boards_to_tokens(sol, n)``.  The
input hash is ``sha1(flat_token_input_board.astype(uint8).tobytes())[:16]``,
recomputable from a batch of inputs by an eval harness, which makes
``coverage@K`` a dict lookup.
"""

import hashlib
import itertools
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

# Support both `python dataset/build_nqueens_dataset.py` and
# `python -m dataset.build_nqueens_dataset`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import PuzzleDatasetMetadata  # noqa: E402
from nqueens_verify import verify, boards_to_tokens  # noqa: E402


cli = ArgParser()

# Known-good anchors: number of distinct complete solutions.
SOLUTION_ANCHORS = {4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724, 11: 2680, 12: 14200}
# Default removal counts k (queens deleted from the complete solution).
DEFAULT_REMOVALS = {8: (5, 6, 7), 10: (7, 8, 9)}


class DataProcessConfig(BaseModel):
    n: int = 8
    output_dir: str = "data/nqueens-8"
    seed: int = 0
    train_frac: float = 0.85
    # Comma-separated k values; empty -> DEFAULT_REMOVALS[n]
    removals: str = ""
    self_check: bool = True
    check_samples: int = 1000


# ---------------------------------------------------------------------------
# 1. Enumerate all complete solutions
# ---------------------------------------------------------------------------

def enumerate_solutions(n: int) -> np.ndarray:
    """All complete n-queens solutions as ``[S, n]`` column indices (sorted)."""
    solutions: List[Tuple[int, ...]] = []
    cols = [0] * n
    used_col = [False] * n
    used_diag = [False] * (2 * n - 1)   # r - c + (n - 1)
    used_anti = [False] * (2 * n - 1)   # r + c

    def place(r: int):
        if r == n:
            solutions.append(tuple(cols))
            return
        for c in range(n):
            d = r - c + n - 1
            a = r + c
            if used_col[c] or used_diag[d] or used_anti[a]:
                continue
            used_col[c] = used_diag[d] = used_anti[a] = True
            cols[r] = c
            place(r + 1)
            used_col[c] = used_diag[d] = used_anti[a] = False

    place(0)
    solutions.sort()
    arr = np.array(solutions, dtype=np.int32)

    if n in SOLUTION_ANCHORS:
        assert arr.shape[0] == SOLUTION_ANCHORS[n], (
            f"solver bug: n={n} produced {arr.shape[0]} solutions, expected {SOLUTION_ANCHORS[n]}"
        )
    return arr


# ---------------------------------------------------------------------------
# 2. Generate (puzzle, solution) pairs
# ---------------------------------------------------------------------------

def input_hash(clues: Tuple[Tuple[int, int], ...], n: int) -> str:
    """Stable hash of a clue set, identical to hashing its flat token board."""
    board = np.ones(n * n, dtype=np.uint8)  # token 1 = empty
    for r, c in clues:
        board[r * n + c] = 2
    return hashlib.sha1(board.tobytes()).hexdigest()[:16]


def generate_pairs(sols: np.ndarray, n: int, removals: Tuple[int, ...]):
    """Return (pair_sol_idx, pair_key, key_to_clues).

    ``pair_sol_idx[i]`` indexes ``sols``; ``pair_key[i]`` is the input hash of
    pair ``i``; ``key_to_clues`` maps hash -> clue tuple.
    """
    pair_sol_idx: List[int] = []
    pair_key: List[str] = []
    key_to_clues: Dict[str, Tuple[Tuple[int, int], ...]] = {}

    all_rows = tuple(range(n))
    keep_sets = [
        tuple(sorted(set(all_rows) - set(drop)))
        for k in removals
        for drop in itertools.combinations(all_rows, k)
    ]

    for si in range(sols.shape[0]):
        sol = sols[si]
        for keep in keep_sets:
            clues = tuple((r, int(sol[r])) for r in keep)
            h = input_hash(clues, n)
            if h not in key_to_clues:
                key_to_clues[h] = clues
            pair_sol_idx.append(si)
            pair_key.append(h)

    return np.array(pair_sol_idx, dtype=np.int64), pair_key, key_to_clues


# ---------------------------------------------------------------------------
# 5. Solution-set sidecar
# ---------------------------------------------------------------------------

def consistent_solutions(sols: np.ndarray, keys: List[str],
                         key_to_clues: Dict[str, Tuple[Tuple[int, int], ...]],
                         chunk: int = 2048) -> Dict[str, np.ndarray]:
    """For each unique input hash, the indices of every complete solution
    containing all of its clue queens.  Computed from the full solution list."""
    out: Dict[str, np.ndarray] = {}
    # Group by number of clues so the clue arrays are rectangular.
    by_m: Dict[int, List[str]] = {}
    for h in keys:
        by_m.setdefault(len(key_to_clues[h]), []).append(h)

    for m, hs in by_m.items():
        for start in range(0, len(hs), chunk):
            block = hs[start:start + chunk]
            clue = np.array([key_to_clues[h] for h in block], dtype=np.int32)  # [U, m, 2]
            rows, cols = clue[..., 0], clue[..., 1]                            # [U, m]
            # sols[:, rows] -> [S, U, m]
            match = (sols[:, rows] == cols[None]).all(axis=2)                  # [S, U]
            for j, h in enumerate(block):
                out[h] = np.nonzero(match[:, j])[0]
    return out


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _raw_board(clues, n: int) -> np.ndarray:
    b = np.zeros(n * n, dtype=np.int32)
    for r, c in clues:
        b[r * n + c] = 1
    return b


def save_split(set_name: str, output_dir: str, n: int,
               inputs_raw: np.ndarray, labels_raw: np.ndarray,
               group_sizes: List[int]) -> Dict[str, int]:
    """Write the five ``all__*.npy`` arrays plus ``dataset.json``.

    One group per unique input; one puzzle per row; one example per puzzle.
    """
    assert inputs_raw.shape == labels_raw.shape
    N = inputs_raw.shape[0]
    assert sum(group_sizes) == N

    assert np.all((inputs_raw >= 0) & (inputs_raw <= 1))
    assert np.all((labels_raw >= 0) & (labels_raw <= 1))
    inputs = (inputs_raw + 1).astype(np.int32)   # 1 = empty, 2 = queen
    labels = (labels_raw + 1).astype(np.int32)

    puzzle_indices = np.arange(N + 1, dtype=np.int32)          # each puzzle = 1 example
    group_indices = np.concatenate([[0], np.cumsum(group_sizes)]).astype(np.int32)
    puzzle_identifiers = np.zeros(N, dtype=np.int32)

    metadata = PuzzleDatasetMetadata(
        seq_len=n * n,
        vocab_size=3,          # PAD + empty + queen
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(group_indices) - 1,
        mean_puzzle_examples=1,
        total_puzzles=N,
        sets=["all"],
    )

    save_dir = os.path.join(output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    for k, v in {
        "inputs": inputs,
        "labels": labels,
        "puzzle_indices": puzzle_indices,
        "group_indices": group_indices,
        "puzzle_identifiers": puzzle_identifiers,
    }.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    return {"rows": N, "groups": len(group_indices) - 1}


# ---------------------------------------------------------------------------
# Acceptance self-check
# ---------------------------------------------------------------------------

def text_histogram(values: np.ndarray, title: str, width: int = 50):
    print(f"\n{title}")
    if values.size == 0:
        print("  (empty)")
        return
    vals, counts = np.unique(values, return_counts=True)
    top = counts.max()
    for v, c in zip(vals, counts):
        bar = "#" * max(1, int(round(width * c / top)))
        print(f"  {int(v):>4} solution(s) | {bar:<{width}} {c:>8}  ({100.0 * c / values.size:5.2f}%)")
    print(f"  total unique inputs = {values.size}, mean = {values.mean():.3f}, max = {int(vals.max())}")


def run_self_check(n: int, sols: np.ndarray, sidecar_idx: Dict[str, np.ndarray],
                   key_to_clues, train_keys, test_keys, num_pairs: int,
                   samples: int, seed: int):
    print("\n" + "=" * 72)
    print(f"ACCEPTANCE SELF-CHECK  (n={n})")
    print("=" * 72)

    exp_sol = SOLUTION_ANCHORS.get(n)
    print(f"[anchor] complete solutions : actual={sols.shape[0]:>7}  expected={exp_sol}")
    exp_pairs = {8: 8464, 10: 126700}.get(n)
    print(f"[anchor] (puzzle,solution) pairs: actual={num_pairs:>7}  expected={exp_pairs}")
    if exp_pairs is not None:
        assert num_pairs == exp_pairs, f"pair anchor mismatch: {num_pairs} != {exp_pairs}"

    inter = set(train_keys) & set(test_keys)
    print(f"[split ] |train hashes|={len(train_keys)}  |test hashes|={len(test_keys)}  "
          f"|intersection|={len(inter)}  disjoint={len(inter) == 0}")
    assert len(inter) == 0, "train/test input hashes overlap -- leakage!"

    rng = np.random.default_rng(seed + 12345)
    all_keys = list(sidecar_idx.keys())

    # (a) every sidecar solution is strict==True against its input
    pick_k = rng.choice(len(all_keys), size=min(samples, len(all_keys)), replace=False)
    inp_list, out_list = [], []
    for ki in pick_k:
        h = all_keys[ki]
        clues = key_to_clues[h]
        inp = _raw_board(clues, n) + 1
        for si in sidecar_idx[h]:
            inp_list.append(inp)
            out_list.append(boards_to_tokens(sols[si], n))
    res = verify(np.stack(out_list), np.stack(inp_list), n)
    n_a = len(out_list)
    print(f"\n[verify a] sidecar (input, solution) pairs checked = {n_a}")
    print(f"           strict PASS={int(res['strict'].sum())}  FAIL={int((~res['strict']).sum())}")
    assert res["strict"].all(), "sidecar solution failed strict verification"

    # (b) move exactly one queen to a random empty cell -> valid must be False
    m = min(samples, n_a)
    sel = rng.choice(n_a, size=m, replace=False)
    good = np.stack([out_list[i] for i in sel]).copy()
    corrupt_inp = np.stack([inp_list[i] for i in sel])
    for row in range(m):
        qpos = np.nonzero(good[row] == 2)[0]
        epos = np.nonzero(good[row] == 1)[0]
        good[row, rng.choice(qpos)] = 1
        good[row, rng.choice(epos)] = 2
    cres = verify(good, corrupt_inp, n)
    n_invalid = int((~cres["valid"]).sum())
    print(f"\n[verify b] one-queen-moved corruptions checked = {m}")
    print(f"           valid==False PASS={n_invalid}  FAIL(still valid)={m - n_invalid}")
    assert n_invalid == m, "a one-queen corruption verified as valid"

    text_histogram(np.array([len(v) for v in sidecar_idx.values()]),
                   "[hist  ] solutions per unique input:")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    n = config.n
    if config.removals.strip():
        removals = tuple(int(x) for x in config.removals.split(","))
    else:
        assert n in DEFAULT_REMOVALS, f"no default removals for n={n}; pass --removals"
        removals = DEFAULT_REMOVALS[n]
    print(f"n={n}  removals(k)={removals}  seed={config.seed}  out={config.output_dir}")

    # 1. solutions
    sols = enumerate_solutions(n)
    print(f"complete solutions: {sols.shape[0]}")

    # 2. pairs
    pair_sol_idx, pair_key, key_to_clues = generate_pairs(sols, n, removals)
    num_pairs = pair_sol_idx.shape[0]
    print(f"(puzzle, solution) pairs: {num_pairs}   unique inputs: {len(key_to_clues)}")
    exp_pairs = {8: 8464, 10: 126700}.get(n)
    if exp_pairs is not None:
        assert num_pairs == exp_pairs, f"pair anchor mismatch: {num_pairs} != {exp_pairs}"

    # 5. sidecar (computed independently from the full solution list)
    unique_keys = sorted(key_to_clues.keys())
    sidecar_idx = consistent_solutions(sols, unique_keys, key_to_clues)

    # cross-check: sidecar must agree with the labels actually generated
    from collections import defaultdict
    observed = defaultdict(set)
    for si, h in zip(pair_sol_idx, pair_key):
        observed[h].add(int(si))
    for h in unique_keys:
        assert observed[h] == set(int(x) for x in sidecar_idx[h]), (
            f"sidecar disagrees with generated labels for input {h}"
        )
    print("sidecar cross-check vs generated labels: OK")

    # 4. split by group
    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(len(unique_keys))
    n_train = int(round(config.train_frac * len(unique_keys)))
    train_keys = [unique_keys[i] for i in perm[:n_train]]
    test_keys = [unique_keys[i] for i in perm[n_train:]]
    train_set, test_set = set(train_keys), set(test_keys)
    assert not (train_set & test_set)
    print(f"split: {len(train_keys)} train inputs / {len(test_keys)} test inputs")

    # ---- assemble train: all rows, grouped by input ----
    rows_by_key = defaultdict(list)
    for si, h in zip(pair_sol_idx, pair_key):
        rows_by_key[h].append(int(si))

    def build(keys, one_row_per_input: bool):
        inp_rows, lab_rows, gsizes = [], [], []
        for h in sorted(keys):
            clues = key_to_clues[h]
            inp = _raw_board(clues, n)
            # deterministic: sorted solution indices; test takes the first
            sidxs = sorted(rows_by_key[h])
            if one_row_per_input:
                sidxs = sidxs[:1]
            for si in sidxs:
                inp_rows.append(inp)
                lab_rows.append(_raw_board([(r, int(sols[si][r])) for r in range(n)], n))
            gsizes.append(len(sidxs))
        return np.stack(inp_rows), np.stack(lab_rows), gsizes

    tr_inp, tr_lab, tr_groups = build(train_keys, one_row_per_input=False)
    te_inp, te_lab, te_groups = build(test_keys, one_row_per_input=True)

    os.makedirs(config.output_dir, exist_ok=True)
    tr_stats = save_split("train", config.output_dir, n, tr_inp, tr_lab, tr_groups)
    te_stats = save_split("test", config.output_dir, n, te_inp, te_lab, te_groups)

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    # sidecar file
    split_of = {h: ("train" if h in train_set else "test") for h in unique_keys}
    sidecar = {
        "meta": {
            "n": n,
            "encoding": "columns",
            "removals": list(removals),
            "num_solutions_total": int(sols.shape[0]),
            "num_unique_inputs": len(unique_keys),
            "hash": "sha1(flat token board uint8 bytes).hexdigest()[:16]",
        },
        "inputs": {
            h: {
                "split": split_of[h],
                "clues": [[int(r), int(c)] for r, c in key_to_clues[h]],
                "solutions": [[int(x) for x in sols[si]] for si in sidecar_idx[h]],
            }
            for h in unique_keys
        },
    }
    sidecar_path = os.path.join(config.output_dir, f"sidecar_{n}.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f)
    print(f"wrote sidecar: {sidecar_path}")

    print(f"\nTRAIN rows={tr_stats['rows']} groups={tr_stats['groups']}")
    print(f"TEST  rows={te_stats['rows']} groups={te_stats['groups']}")
    print(f"seq_len={n * n} vocab_size=3 mean_puzzle_examples=1")

    if config.self_check:
        run_self_check(n, sols, sidecar_idx, key_to_clues, train_keys, test_keys,
                       num_pairs, config.check_samples, config.seed)
        print("\nALL ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    cli()
