"""Build a 3-Graph-Coloring dataset in the TRM/GRAM ``PuzzleDataset`` format.

Task
----
The model is given an undirected graph (upper triangle of its adjacency matrix)
and must output a proper 3-coloring: one color per node, adjacent nodes must
differ.

Token vocabulary (``vocab_size = 6``)::

    0 = PAD / ignore
    1 = no edge
    2 = edge
    3 = red
    4 = blue
    5 = green

Encoding (``seq_len = n*(n-1)/2``; n=8 -> 28, n=10 -> 45)
--------------------------------------------------------
``inputs``  : the strict upper triangle of the adjacency matrix, flattened in
              ``np.triu_indices(n, 1)`` order (row-major over pairs
              (0,1), (0,2), ..., (n-2,n-1)).  Tokens 1/2 only.
``labels``  : node colors in the FIRST ``n`` positions (tokens 3/4/5); the
              remaining ``seq_len - n`` positions are ``0``.

The label is shorter than the sequence (n colors vs n(n-1)/2 positions), so the
tail is filled with ``ignore_label_id = 0``.  ``puzzle_dataset.py`` rewrites
that to ``IGNORE_LABEL_ID = -100`` and ``models/losses.py`` masks it out, so the
padding contributes nothing to the loss.  This mirrors how the paper specifies
"input = adjacency upper triangle, output = length-N color vector" inside a
fixed-width sequence model.

Color permutation canonical form
--------------------------------
A proper coloring stays proper under any permutation of the 3 colors, so each
graph would otherwise appear with up to 3! = 6 equivalent labels.  We keep only
colorings in *first-appearance normal form*: scanning nodes 0..n-1, the first
color seen is 0, the next new color is 1, then 2.  Equivalently ``c[0] == 0``
and ``c[j] <= max(c[:j]) + 1``.  Exactly one member of each permutation orbit
survives, so no graph is duplicated by a relabeling of its colors.

Pipeline
--------
1. Sample Erdos-Renyi graphs G(n, p): each of the n(n-1)/2 possible edges
   independently with probability ``p``, symmetric, no self-loops.
2. Keep a graph only if it is 3-colorable (>= 1 canonical proper coloring).
   Duplicate graphs (identical upper triangle) are discarded, so every graph in
   the dataset is unique.  Sampling continues until ``num_graphs`` are kept.
3. Split the unique graphs 85:15 into train/test.  The split is over *graphs*,
   so no graph ever straddles the split (asserted disjoint) -- identical
   discipline to ``build_nqueens_dataset.py``.
4. Train keeps every canonical coloring of its graphs; test keeps exactly one
   (the lexicographically first), matching the N-Queens builder.

``p`` is not given in the paper.  It is chosen so that the resulting dataset
sizes reproduce the published anchors (see ``SIZE_ANCHORS``); the number of
colorings per graph falls steeply with p, which is what pins it down.

Sidecar
-------
``sidecar_gc_{n}.json`` maps each graph hash to its split, edge list, and the
FULL list of canonical proper colorings -- the analogue of the N-Queens
solution sidecar, for coverage@K style harnesses.
"""
import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import PuzzleDatasetMetadata  # noqa: E402

cli = ArgParser()

NO_EDGE, EDGE = 1, 2
COLOR0 = 3          # 3=red, 4=blue, 5=green

# Published dataset sizes (GRAM paper, appendix C.2.2).
#   n -> (train_pairs, test_pairs)
SIZE_ANCHORS = {8: (7002, 255), 10: (13465, 192)}
# Unique 3-colorable graphs implied by the anchors: test keeps 1 row per graph
# and the split is 85:15, so num_graphs = test_pairs / 0.15.
DEFAULT_NUM_GRAPHS = {8: 1700, 10: 1280}
# Edge probability tuned to reproduce the train-pair anchor (see module docstring).
# Scanned at seed 0 in steps of 0.0005; these minimise |actual - anchor|:
#   n=8  p=0.5455 -> 6996 train pairs (anchor 7002, -6)
#   n=10 p=0.4250 -> 13461 train pairs (anchor 13465, -4)
# The count is a random variable: neighbouring p values and neighbouring seeds
# move it by +-200 (n=8) / +-1000 (n=10), so the residual is sampling noise, not
# a wrong p or a wrong 3-colorability filter.  Hitting the anchor exactly would
# mean fitting the seed to the noise, which is not evidence of anything.
DEFAULT_P = {8: 0.5455, 10: 0.4250}


class DataProcessConfig(BaseModel):
    n: int = 8
    output_dir: str = "data/graph-coloring-8"
    seed: int = 0
    p: float = -1.0             # <0 -> DEFAULT_P[n]
    num_graphs: int = -1        # <0 -> DEFAULT_NUM_GRAPHS[n]
    train_frac: float = 0.85
    self_check: bool = True
    check_samples: int = 1000


# ---------------------------------------------------------------------------
# 1. Canonical coloring tables
# ---------------------------------------------------------------------------

def canonical_colorings(n: int) -> np.ndarray:
    """All first-appearance-normalized color assignments, ``[M, n]`` in 0..2."""
    C = np.indices((3,) * n).reshape(n, -1).T.astype(np.int8)
    running_max = np.maximum.accumulate(C, axis=1)
    ok = C[:, 0] == 0
    for j in range(1, n):
        ok &= C[:, j] <= running_max[:, j - 1] + 1
    return np.ascontiguousarray(C[ok])


def edge_ok_table(Ccan: np.ndarray, iu: np.ndarray, iv: np.ndarray) -> np.ndarray:
    """``D[e, m]`` = canonical coloring m gives different colors to edge e's ends."""
    return np.ascontiguousarray((Ccan[:, iu] != Ccan[:, iv]).T)


def proper_mask(bits: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Which canonical colorings are proper for the graph given by ``bits``."""
    e = np.nonzero(bits)[0]
    if len(e) == 0:
        return np.ones(D.shape[1], dtype=bool)
    return D[e].all(0)


# ---------------------------------------------------------------------------
# 2. Sampling
# ---------------------------------------------------------------------------

def graph_hash(bits: np.ndarray) -> str:
    tokens = np.where(bits, EDGE, NO_EDGE).astype(np.uint8)
    return hashlib.sha1(tokens.tobytes()).hexdigest()[:16]


def sample_graphs(n: int, p: float, num_graphs: int, seed: int, D: np.ndarray,
                  P: int, batch: int = 4096, max_tries: int = 40_000_000):
    """Sample unique 3-colorable ER graphs. Returns (bits[G,P], colorings list)."""
    rng = np.random.default_rng(seed)
    seen: Dict[str, int] = {}
    bits_out: List[np.ndarray] = []
    cols_out: List[np.ndarray] = []
    tries = kept_dup = not3 = 0

    while len(bits_out) < num_graphs:
        if tries >= max_tries:
            raise RuntimeError(f"max_tries exceeded with {len(bits_out)}/{num_graphs} graphs")
        block = rng.random((batch, P)) < p
        tries += batch
        for bits in block:
            h = graph_hash(bits)
            if h in seen:
                kept_dup += 1
                continue
            m = proper_mask(bits, D)
            if not m.any():
                not3 += 1
                continue
            seen[h] = len(bits_out)
            bits_out.append(bits.copy())
            cols_out.append(np.nonzero(m)[0])
            if len(bits_out) >= num_graphs:
                break

    stats = dict(tries=tries, duplicates=kept_dup, not_3colorable=not3,
                 accept_rate=len(bits_out) / max(tries, 1))
    return np.stack(bits_out), cols_out, stats


# ---------------------------------------------------------------------------
# 3. Encoding / serialization
# ---------------------------------------------------------------------------

def encode_input(bits: np.ndarray) -> np.ndarray:
    return np.where(bits, EDGE, NO_EDGE).astype(np.int32)


def encode_label(coloring: np.ndarray, seq_len: int) -> np.ndarray:
    lab = np.zeros(seq_len, dtype=np.int32)          # 0 = ignore
    lab[:len(coloring)] = coloring.astype(np.int32) + COLOR0
    return lab


def save_split(set_name: str, output_dir: str, n: int, seq_len: int,
               inputs: np.ndarray, labels: np.ndarray, group_sizes: List[int]):
    assert inputs.shape == labels.shape
    N = inputs.shape[0]
    assert sum(group_sizes) == N
    assert set(np.unique(inputs).tolist()) <= {NO_EDGE, EDGE}
    assert set(np.unique(labels).tolist()) <= {0, 3, 4, 5}

    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len,
        vocab_size=6,           # PAD + no_edge + edge + 3 colors
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(group_sizes),
        mean_puzzle_examples=1,
        total_puzzles=N,
        sets=["all"],
    )
    save_dir = os.path.join(output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    arrays = {
        "inputs": inputs,
        "labels": labels,
        "puzzle_indices": np.arange(N + 1, dtype=np.int32),
        "group_indices": np.concatenate([[0], np.cumsum(group_sizes)]).astype(np.int32),
        "puzzle_identifiers": np.zeros(N, dtype=np.int32),
    }
    for k, v in arrays.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    return {"rows": N, "groups": len(group_sizes)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    n = config.n
    p = config.p if config.p >= 0 else DEFAULT_P.get(n, 0.5)
    num_graphs = config.num_graphs if config.num_graphs > 0 else DEFAULT_NUM_GRAPHS.get(n, 1000)
    seq_len = n * (n - 1) // 2
    iu, iv = np.triu_indices(n, 1)
    P = len(iu)
    assert P == seq_len

    print(f"n={n}  p={p}  num_graphs={num_graphs}  seq_len={seq_len}  seed={config.seed}")

    Ccan = canonical_colorings(n)
    D = edge_ok_table(Ccan, iu, iv)
    print(f"canonical color assignments: {Ccan.shape[0]}  (all 3^n = {3**n})")

    bits, col_idx, stats = sample_graphs(n, p, num_graphs, config.seed, D, P)
    per_graph = np.array([len(c) for c in col_idx])
    print(f"sampled: tries={stats['tries']} duplicates={stats['duplicates']} "
          f"not_3colorable={stats['not_3colorable']} accept={stats['accept_rate']:.4f}")
    print(f"canonical colorings per graph: mean={per_graph.mean():.4f} "
          f"min={per_graph.min()} max={per_graph.max()}")
    print(f"edges per graph: mean={bits.sum(1).mean():.3f}")

    hashes = [graph_hash(b) for b in bits]
    assert len(set(hashes)) == len(hashes), "duplicate graph slipped through"

    # ---- split over graphs ----
    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(num_graphs)
    n_train = int(round(config.train_frac * num_graphs))
    train_ids = sorted(perm[:n_train].tolist())
    test_ids = sorted(perm[n_train:].tolist())
    assert not (set(train_ids) & set(test_ids))
    print(f"split: {len(train_ids)} train graphs / {len(test_ids)} test graphs")

    def build(ids: List[int], one_row_per_graph: bool):
        inp_rows, lab_rows, gsizes = [], [], []
        for gi in ids:
            x = encode_input(bits[gi])
            cs = col_idx[gi]                       # sorted -> lexicographically first is cs[0]
            if one_row_per_graph:
                cs = cs[:1]
            for ci in cs:
                inp_rows.append(x)
                lab_rows.append(encode_label(Ccan[ci], seq_len))
            gsizes.append(len(cs))
        return np.stack(inp_rows), np.stack(lab_rows), gsizes

    tr_x, tr_y, tr_g = build(train_ids, False)
    te_x, te_y, te_g = build(test_ids, True)

    os.makedirs(config.output_dir, exist_ok=True)
    tr = save_split("train", config.output_dir, n, seq_len, tr_x, tr_y, tr_g)
    te = save_split("test", config.output_dir, n, seq_len, te_x, te_y, te_g)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    print(f"train rows={tr['rows']} groups={tr['groups']}   test rows={te['rows']} groups={te['groups']}")
    anchor = SIZE_ANCHORS.get(n)
    if anchor is not None:
        print(f"[anchor] train pairs actual={tr['rows']} expected={anchor[0]}  "
              f"({tr['rows']-anchor[0]:+d})")
        print(f"[anchor] test  pairs actual={te['rows']} expected={anchor[1]}  "
              f"({te['rows']-anchor[1]:+d})")

    # ---- sidecar ----
    sidecar = {"meta": {"n": n, "p": p, "num_graphs": num_graphs, "seed": config.seed,
                        "encoding": "upper_triangle", "canonical": "first_appearance",
                        "vocab": {"pad": 0, "no_edge": 1, "edge": 2,
                                  "red": 3, "blue": 4, "green": 5},
                        "hash": "sha1(upper-triangle token bytes).hexdigest()[:16]"},
               "graphs": {}}
    split_of = {}
    for gi in train_ids:
        split_of[gi] = "train"
    for gi in test_ids:
        split_of[gi] = "test"
    for gi in range(num_graphs):
        e = np.nonzero(bits[gi])[0]
        sidecar["graphs"][hashes[gi]] = {
            "split": split_of[gi],
            "edges": [[int(iu[k]), int(iv[k])] for k in e],
            "colorings": [Ccan[ci].tolist() for ci in col_idx[gi]],
        }
    with open(os.path.join(config.output_dir, f"sidecar_gc_{n}.json"), "w") as f:
        json.dump(sidecar, f)

    if config.self_check:
        run_self_check(n, seq_len, tr_x, tr_y, te_x, te_y, hashes,
                       train_ids, test_ids, config.seed, config.check_samples)


def run_self_check(n, seq_len, tr_x, tr_y, te_x, te_y, hashes,
                   train_ids, test_ids, seed, samples):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from evaluators.graph_coloring import verify_colorings  # noqa: E402

    print("\n" + "=" * 72)
    print(f"ACCEPTANCE SELF-CHECK  (graph coloring, n={n})")
    print("=" * 72)

    for name, x, y in (("train", tr_x, tr_y), ("test", te_x, te_y)):
        r = verify_colorings(y.astype(np.int64), x.astype(np.int64), n)
        print(f"[labels {name:5}] valid={r['valid'].mean():.6f}  "
              f"conflicts total={int(r['conflicts'].sum())}  "
              f"colors_ok={r['colors_ok'].mean():.6f}  rows={len(y)}")
        assert r["valid"].all(), f"{name} label is not a proper coloring"
        assert r["conflicts"].sum() == 0, f"{name} label has conflicting edges"

    def keyset(a):
        return set(map(bytes, np.ascontiguousarray(a.astype(np.int8))))
    ktr, kte = keyset(tr_x), keyset(te_x)
    print(f"[split ] distinct train x={len(ktr)}  test x={len(kte)}  "
          f"intersection={len(ktr & kte)}")
    assert not (ktr & kte), "train/test graph overlap -- leakage!"

    ptr = keyset(np.concatenate([tr_x, tr_y], 1))
    pte = keyset(np.concatenate([te_x, te_y], 1))
    print(f"[dup   ] train duplicate rows={len(tr_x)-len(ptr)}  test duplicate rows={len(te_x)-len(pte)}")
    assert len(ptr) == len(tr_x) and len(pte) == len(te_x), "duplicate (x,y) rows"

    # corrupting one node's color into a neighbor's color must break validity
    rng = np.random.default_rng(seed + 999)
    iu, iv = np.triu_indices(n, 1)
    m = min(samples, len(tr_x))
    sel = rng.choice(len(tr_x), size=m, replace=False)
    bad = tr_y[sel].copy()
    nbad = 0
    for r_i, row in enumerate(sel):
        e = np.nonzero(tr_x[row] == EDGE)[0]
        if len(e) == 0:
            continue
        k = e[rng.integers(len(e))]
        u, v = iu[k], iv[k]
        bad[r_i, u] = bad[r_i, v]          # force a conflict on edge (u,v)
        nbad += 1
    r = verify_colorings(bad.astype(np.int64), tr_x[sel].astype(np.int64), n)
    print(f"[negative] forced one conflicting edge on {nbad} rows -> "
          f"valid={r['valid'].sum()} (expect 0)  mean conflicts={r['conflicts'].mean():.3f}")
    assert r["valid"].sum() == 0, "verifier accepted a conflicting coloring"
    print("=" * 72)
    print("SELF-CHECK PASSED")


if __name__ == "__main__":
    cli()
