"""Build an N-Queens dataset in the TRM/GRAM ``PuzzleDataset`` format.

The model gets a *partial* board (a subset of the queens of some complete
solution) and must output a complete legal placement keeping every given queen.
Tokens: 0 = PAD (unused), 1 = empty, 2 = queen.  ``seq_len = n * n``, row-major.

Rows are grouped by unique input; the train/test split is over *groups*, so no
input straddles the split.  Test keeps one row per input, train keeps all.

``sidecar_{n}.json`` maps each input hash to its split, clues, and the FULL set
of solutions consistent with it -- ``experiments/eval_harness.py`` (coverage@K)
and ``experiments/diag/diag_bucket.py`` read it.  Solutions are column encodings:
``solution[r] = c``.  The hash is sha1 of the flat uint8 token board, [:16].
"""

from typing import Optional
from collections import defaultdict
import math
import os
import json
import hashlib
import itertools
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata


SOLUTION_ANCHORS = {4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724, 11: 2680, 12: 14200}
# 논문 p19 값이다 (우리가 고른 값이 아니다): 8x8 은 k in {5,6,7}, 10x10 은 k in {7,8,9} 를
# 제거한다.  removals 를 바꾸면 해-개수 분포가 통째로 달라진다 -- 10x10 에서 단서 3 개짜리
# (k=7)가 C(10,7)=120 으로 전체 인스턴스의 92% 를 차지하고 평균 해 2.18 개인 반면,
# 단서 1 개짜리(k=9)는 100 개뿐인데 평균 해 72.4 개다.  coverage 를 논문과 맞대려면
# 이 값이 일치해야 한다.
DEFAULT_REMOVALS = {8: (5, 6, 7), 10: (7, 8, 9)}
PAIR_ANCHORS = {8: 8464, 10: 126700}


cli = ArgParser()


class DataProcessConfig(BaseModel):
    n: int = 8
    output_dir: str = "data/nqueens-8"
    seed: int = 0
    train_frac: float = 0.85

    removals: Optional[str] = None


def build_pairs(config: DataProcessConfig):
    """Enumerate solutions, generate (input, label) pairs, split by input."""
    n = config.n
    assert config.removals or n in DEFAULT_REMOVALS, f"no default removals for n={n}; pass --removals"
    removals = tuple(int(x) for x in config.removals.split(",")) if config.removals else DEFAULT_REMOVALS[n]

    # All complete solutions, by backtracking
    sols = []
    cols = [0] * n
    used_c, used_d, used_a = [False] * n, [False] * (2 * n - 1), [False] * (2 * n - 1)

    def place(r):
        if r == n:
            sols.append(tuple(cols))
            return
        for c in range(n):
            if not (used_c[c] or used_d[r - c + n - 1] or used_a[r + c]):
                used_c[c] = used_d[r - c + n - 1] = used_a[r + c] = True
                cols[r] = c
                place(r + 1)
                used_c[c] = used_d[r - c + n - 1] = used_a[r + c] = False

    place(0)
    sols = np.array(sorted(sols), dtype=np.int32)
    assert sols.shape[0] == SOLUTION_ANCHORS.get(n, sols.shape[0]), f"solver bug: {sols.shape[0]} solutions for n={n}"

    # Every way of deleting k of the n queens -> the survivors are the clues
    keep_sets = [tuple(sorted(set(range(n)) - set(drop)))
                 for k in removals for drop in itertools.combinations(range(n), k)]
    assert keep_sets, f"removals={removals} produce no clue sets for n={n}"

    key_to_clues = {}
    rows_by_key = defaultdict(list)

    for si, sol in enumerate(tqdm(sols, desc="Generating pairs")):
        for keep in keep_sets:
            clues = tuple((r, int(sol[r])) for r in keep)

            board = np.ones(n * n, dtype=np.uint8)
            for r, c in clues:
                board[r * n + c] = 2
            h = hashlib.sha1(board.tobytes()).hexdigest()[:16]

            assert key_to_clues.setdefault(h, clues) == clues, f"sha1 collision on {h}"
            rows_by_key[h].append(si)

    num_pairs = sum(len(v) for v in rows_by_key.values())
    assert num_pairs == PAIR_ANCHORS.get(n, num_pairs), f"pair anchor mismatch: {num_pairs}"

    # Split unique inputs 85:15
    unique_keys = sorted(key_to_clues)
    perm = np.random.default_rng(config.seed).permutation(len(unique_keys))
    n_train = int(round(config.train_frac * len(unique_keys)))
    train_keys = [unique_keys[i] for i in perm[:n_train]]
    test_keys = [unique_keys[i] for i in perm[n_train:]]

    return sols, removals, key_to_clues, rows_by_key, train_keys, test_keys


def convert_subset(set_name: str, keys, sols, key_to_clues, rows_by_key, config: DataProcessConfig):
    n = config.n
    grid_size = (n, n)

    # Generate dataset
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for h in tqdm(sorted(keys), desc=f"Processing {set_name}"):
        inp = np.zeros(n * n, dtype=np.int32)
        for r, c in key_to_clues[h]:
            inp[r * n + c] = 1

        # One row per consistent solution; test keeps only the first
        sidxs = sorted(rows_by_key[h])
        if set_name == "test":
            sidxs = sidxs[:1]

        for si in sidxs:
            lab = np.zeros(n * n, dtype=np.int32)
            for r, c in enumerate(sols[si]):
                lab[r * n + c] = 1

            results["inputs"].append(inp)
            results["labels"].append(lab)
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        # Push group
        results["group_indices"].append(puzzle_id)

    # To Numpy, shifting raw {0, 1} to tokens {1, 2} so PAD id 0 stays unused
    results = {
        "inputs": np.vstack(results["inputs"]).astype(np.int32) + 1,
        "labels": np.vstack(results["labels"]).astype(np.int32) + 1,

        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=int(math.prod(grid_size)),  # type: ignore
        vocab_size=3,  # PAD + empty + queen
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["inputs"]),
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


def save_sidecar(sols, removals, key_to_clues, rows_by_key, train_keys, test_keys, config: DataProcessConfig):
    """Input hash -> split, clues, and every solution consistent with it."""
    split_of = {h: "train" for h in train_keys}
    split_of.update({h: "test" for h in test_keys})
    assert len(split_of) == len(train_keys) + len(test_keys), "train/test overlap -- leakage!"

    sidecar = {
        "meta": {
            "n": config.n,
            "encoding": "columns",
            "removals": list(removals),
            "num_solutions_total": int(sols.shape[0]),
            "num_unique_inputs": len(key_to_clues),
            "hash": "sha1(flat token board uint8 bytes).hexdigest()[:16]",
        },
        "inputs": {
            h: {
                "split": split_of[h],
                "clues": [[int(r), int(c)] for r, c in key_to_clues[h]],
                "solutions": [[int(x) for x in sols[si]] for si in sorted(rows_by_key[h])],
            }
            for h in sorted(key_to_clues)
        },
    }

    with open(os.path.join(config.output_dir, f"sidecar_{config.n}.json"), "w") as f:
        json.dump(sidecar, f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    # Unlike the other builders, both splits come from one enumeration pass:
    # the split is over unique inputs, so it cannot be decided per-subset.
    sols, removals, key_to_clues, rows_by_key, train_keys, test_keys = build_pairs(config)

    convert_subset("train", train_keys, sols, key_to_clues, rows_by_key, config)
    convert_subset("test", test_keys, sols, key_to_clues, rows_by_key, config)
    save_sidecar(sols, removals, key_to_clues, rows_by_key, train_keys, test_keys, config)


if __name__ == "__main__":
    cli()
